# Transferability across temperature

Hybrid ANN-BNN-RF prediction of the dissociation time (`tdiss`) of the dimer at a
temperature that was never seen during training. The model is trained on
all-mode-excitation trajectories at **1500 K** and tested on independent
all-mode-excitation trajectories at **1000 K**. Only the normal-mode coordinates and
momenta are used as input, so the feature representation is identical at both temperatures.

| System | Config | Train (1500 K) | Test (1000 K) | Normal modes |
|---|---|---|---|---|
| Benzene dimer | `configs/benzene.json` | 400 trajectories | 400 trajectories | 66 |
| Phenol dimer | `configs/phenol.json` | 400 trajectories | 400 trajectories | 72 |

## Repository layout

```
temperature_transfer/
├── temperature_transfer.py      workflow script (RF, BNN, hybrid, survival curves)
├── configs/                     benzene.json, phenol.json: all settings of each system
├── sample_data/<system>/        two trajectories per temperature, for a quick test
└── expected_output/<system>/    reference results of the manuscript run
```

The full data sets are large (about 1.8–1.9 GB per file) and are archived on
Zenodo: **[DOI to be added]**. Download them and place them in `data/<system>/`
(or point the script to them with `--data-dir`).

## Data

Frame-level processed machine-learning data; one row per retained frame.

| Column | Content |
|---|---|
| `trajectory` | trajectory index |
| `time` | time of the frame; every fifth frame of each trajectory is retained, starting from t = 0 |
| `Q1 … Qn` | normal-mode coordinates (n = 66 benzene dimer, 72 phenol dimer) |
| `P1 … Pn` | normal-mode momenta |
| `tdiss` | dissociation time of the trajectory (ps), the prediction target |

The Cartesian coordinates and momenta of the selected frames were transformed into
normal-mode coordinates and momenta before being stored. Raw trajectories and
simulation inputs are not distributed; they are not needed to reproduce the
machine-learning results.

| System | Role | File (in `data/<system>/`) | Trajectories | Frames per trajectory |
|---|---|---|---|---|
| Benzene | train, 1500 K | `normal-mode-dataset-random-1500K-400-traj-tdiss.csv.gz` | 400 | 1800–1801 |
| Benzene | test, 1000 K | `normal-mode-dataset-random-1000K-400-traj-tdiss.csv.gz` | 400 | 1801 |
| Phenol | train, 1500 K | `phenol-normal-mode-dataset-random-1500K-400-traj-tdiss.csv.gz` | 400 | 1601 |
| Phenol | test, 1000 K | `phenol-normal-mode-dataset-random-1000K-400-traj-tdiss.csv.gz` | 400 | 1601–3201 |

The number of frames per trajectory follows the length of the simulation, so it is
not constant in the phenol test set.

## Run

```bash
pip install -r ../requirements.txt

# quick check on the bundled sample (two trajectories per temperature; the
# numbers are not meaningful, the run only verifies the installation)
python temperature_transfer.py --config configs/benzene.json \
    --data-dir sample_data/benzene --rf-only \
    --train-file sample-1500K-2-traj.csv.gz --test-file sample-1000K-2-traj.csv.gz

# full run (after downloading the data from Zenodo into data/<system>/)
python temperature_transfer.py --config configs/benzene.json
python temperature_transfer.py --config configs/phenol.json
```

`--rf-only` runs the random-forest branch without TensorFlow. The full run trains
the BNN and needs TensorFlow 2.15 and TensorFlow Probability 0.23.

Outputs are written to `outputs/<system>/`:

| File | Content |
|---|---|
| `trajectory_predictions.csv` | per-trajectory RF and BNN predictions, BNN uncertainty, weights, hybrid and final prediction |
| `predicted_survival_curve.txt` | N(t)/N(0) of the final prediction (the ML-predicted curve of the manuscript) |
| `predicted_survival_curve_plain_hybrid.txt` | the same curve for the plain confidence-weighted hybrid, for comparison |
| `reference_survival_curve_test_temperature.txt` | N(t)/N(0) of the simulated 1000 K trajectories |
| `run_summary.json` | MAE, RMSE and R² of every prediction type, trajectory counts |

## Method

1. **Features and target.** `time`, `Q1…Qn`, `P1…Pn` are the inputs (133 features for
   benzene, 145 for phenol); `tdiss` is the target.
2. **Random forest.** 50 trees, maximum depth 10, `min_samples_leaf` 2, seed 42.
3. **Bayesian neural network.** Inputs → 512 → 256 → 128 → 64 → 32 → Bayesian output
   (`DenseVariational`), with batch normalisation and dropout, Huber loss, Adam
   (learning rate 3·10⁻⁴), validation split 0.2, batch size 1024, early stopping
   (patience 20) and learning-rate reduction. 200 stochastic forward passes give
   the mean prediction and the uncertainty σ. The remaining settings differ
   between the two systems (table below).
4. **Trajectory level.** Frame predictions are reduced per trajectory: median for the
   predictions, mean for σ.
5. **Confidence-weighted hybrid.** `c_bnn = 1 / (1 + α σ)` with α = 1, `c_rf = 1`,
   normalised to weights; the hybrid is the weighted sum of the BNN and RF predictions.
6. **Final prediction.** For trajectories whose BNN uncertainty exceeds
   `sigma_threshold` the RF prediction is used, otherwise the hybrid.
   Setting `"sigma_threshold": null` in the config gives the plain hybrid.
7. **Survival curve.** The predicted dissociation times are sorted and plotted
   against the fraction of trajectories remaining, `1 − k/n`.

| Setting | Benzene dimer | Phenol dimer |
|---|---|---|
| `kl_weight` | 10⁻⁷ | 1 / (number of training frames) |
| Maximum epochs | 200 | 100 |
| σ threshold (ps) | 0.04 | 0.45 |

### Note on the uncertainty threshold

The thresholds were chosen manually by trial and error on the 1000 K results; they
are not derived from the training data, and they differ between the two systems. In the benzene
test the BNN uncertainty exceeds 0.04 for 358 of the 400 trajectories, so the final
curve is dominated by the RF prediction. The script reports how many trajectories
fall on each branch and writes both the final and the plain-hybrid curve, so that the
effect of this choice can be checked.

## Expected results

**Benzene dimer** (manuscript run, `expected_output/benzene/`): metrics per trajectory
against the simulated 1000 K `tdiss`.

| Prediction | MAE (ps) | RMSE (ps) | R² |
|---|---|---|---|
| Final (threshold rule) | 0.726 | 2.157 | 0.166 |
| Plain hybrid | 0.680 | 2.062 | 0.238 |
| RF alone | 0.725 | 2.157 | 0.166 |
| BNN alone | 0.713 | 2.042 | 0.252 |

**Phenol dimer** (`expected_output/phenol/`): the ML-predicted N(t)/N(0) curve of the
manuscript.

`ML-predicted-1000K-*.dat` contain the ML-predicted N(t)/N(0) curve of the manuscript
(time in ps, fraction remaining). For benzene it equals the sorted `Final_Prediction`
column of the saved results file `results_threshold_0.04.csv`. The BNN is stochastic
(weight initialisation, dropout, sampling), so a new run reproduces these values
closely but not bit for bit. The agreement of the predicted and simulated curves is
judged by the biexponential fits reported in the manuscript.
