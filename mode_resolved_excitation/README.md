# Mode-resolved excitation: hybrid RF/BNN + SHAP (benzene dimer)

This folder contains the code and data for the analysis "Prediction of
full-mode dissociation rates using mode-resolved excitation". A hybrid
Random Forest / Bayesian neural network (BNN) model is fitted to reconstruct the
dissociation profile of a full-excitation simulation from the profiles of
simpler component simulations, and Kernel SHAP is applied to the hybrid model
to attribute the reconstruction to each component. The attribution gives the
"Predicted Contribution (%)" column of Table 3.

One script handles both cases; everything that differs is in a config file.

| Case | Full simulation (target) | Component simulations (features) | Features | Config |
|---|---|---|---|---|
| Intermolecular | Simu3 (6 intermolecular modes, 142.3 kcal/mol) | Simu13-Simu15 (mode pairs 1-2, 3-4, 5-6) | 3 | `configs/intermolecular.json` |
| Intramolecular | Simu2 (60 intramolecular modes, 215.1 kcal/mol, 1500 K) | Simu4-Simu12 (nine frequency windows) | 9 | `configs/intramolecular.json` |

## Folder contents

```
mode_resolved_shap.py          the analysis script (--case inter | intra)
configs/                       one JSON file per case (files, labels, network size, ...)
data/inter/                    Simu3 and Simu13-Simu15 survival curves
data/intra/                    Simu2 and Simu4-Simu12 survival curves
expected_output/               contribution values from the paper / original run
```

The Python packages are listed in `requirements.txt` in the repository root.

## Data format

Each data file is the survival curve of one simulation (1000 trajectories).
Column 0 is the time in ps at which the survival fraction N(t)/N(0) in column 1
is reached; column 1 falls from 1.000 to 0 in steps of 0.001. The first row is
the t = 0 point, so each file has 1001 rows. Rows are sorted in time, so row *i*
of every file refers to the same survival level. The script reads only column 0.
The model therefore learns the time at which the full simulation reaches a given
survival level from the times at which the component simulations reach the same
level. Tab-separated values.

### Intermolecular files (`data/inter/`)

| Simulation | File | Label |
|---|---|---|
| Simu3 (target) | `all-inter.txt` | all intermolecular modes |
| Simu13 | `alldata_1-2_inter.txt` | Mode 1-2 |
| Simu14 | `alldata_3_4_inter.txt` | Mode 3-4 |
| Simu15 | `alldata_5_6_inter.txt` | Mode 5-6 |

### Intramolecular files (`data/intra/`)

| Simulation | File | Frequency window (cm-1, as in Table 1) |
|---|---|---|
| Simu2 (target) | `all_intra_1500K.txt` | all intramolecular modes |
| Simu4 | `400_range.txt` | 400 |
| Simu5 | `612_range.txt` | 612-653 |
| Simu6 | `696_range.txt` | 696-846 |
| Simu7 | `987_range.txt` | 987-1097 |
| Simu8 | `1154_range.txt` | 1154 |
| Simu9 | `1300_range.txt` | 1374-1387 |
| Simu10 | `1560_range.txt` | 1500 |
| Simu11 | `1700_range.txt` | 1708 |
| Simu12 | `3000_range.txt` | 3174-3192 |

The file names for Simu9-Simu12 are short names for the windows listed in Table 1.

## Installation and usage

```
# once, from the repository root
pip install -r requirements.txt

# then, from this folder
cd mode_resolved_excitation
python mode_resolved_shap.py --case inter
python mode_resolved_shap.py --case intra

# quick installation check (3 epochs, few MC passes, 10 explained rows;
# the numbers it produces are not results)
python mode_resolved_shap.py --case inter --smoke-test
```

Options: `--config`, `--data-dir`, `--out-dir` (default `outputs/<case>`), `--show`.
Kernel SHAP is the slow step: the saved intramolecular notebook run
(9 features, 1001 explained points) took about 2 h 14 min.

### Outputs (in `outputs/<case>/`)

| File | Content |
|---|---|
| `Hybrid_SHAP_Contribution.csv` | mean abs(SHAP) and percent contribution per simulation (Table 3 values) |
| `model_performance.csv` | MAE, RMSE and R2 of the RF, BNN and hybrid models |
| `predictions.csv` | RF, BNN mean, BNN uncertainty, weights and hybrid prediction for every point |
| `shap_summary.png`, `shap_bar.png`, `shap_contribution.png`, `parity_hybrid.png` | figures |
| `x_scaler.pkl`, `y_scaler.pkl` | fitted standard scalers |
| `run_summary.json` | settings, package versions and run time of this run |

## Method summary

1. Features and target are standardized (zero mean, unit variance).
2. Random Forest: 500 trees, maximum depth 20, minimum leaf size 2.
3. BNN: Dense layers with ReLU, BatchNormalization and dropout (0.10) after the
   first layer, and a Bayesian `DenseVariational` output layer. Hidden units are
   [6, 6, 3] for the intermolecular case and [64, 32, 16] for the intramolecular
   case. Huber loss, Adam (learning rate 3e-4), batch size 32, up to 1000 epochs,
   early stopping on the training loss with patience 50.
4. Uncertainty: 200 stochastic forward passes give the predictive mean and the
   standard deviation sigma.
5. Hybrid: BNN confidence 1/(1 + alpha*sigma) with alpha = 1, RF confidence 1;
   the two confidences are normalized into weights and used to combine the BNN
   mean and the RF prediction.
6. Kernel SHAP is applied to the hybrid prediction function with 100 background
   points. Inside that function the BNN uses 100 stochastic passes (intermolecular)
   or 50 (intramolecular). The percent contribution of a feature is its mean
   absolute SHAP value divided by the sum over all features.

The SHAP methodology is written out in full in the Supporting Information.

## Expected results

`expected_output/expected_contributions.csv` lists the values of Table 3 and, for
the intramolecular case, the values from the original notebook run. The two
differ by at most 0.03 percentage points. A new run can differ slightly from both
because of random-number and hardware differences between machines.

## Notes

- The models are fitted to, and explained on, all 1001 points. This is an
  attribution analysis, not a held-out prediction. The contribution percentages
  are model-attributed importances, not fractions of deposited vibrational energy.
- The Monte Carlo passes use `training=True`, so dropout, the variational weight
  sampling and the BatchNormalization batch statistics are active in every pass.
- A fixed seed (42) is used throughout. The intramolecular config re-seeds
  TensorFlow before the Monte Carlo passes (`reseed_before_mc`), as in the
  original notebook; the intermolecular config does not, as in the original script.
