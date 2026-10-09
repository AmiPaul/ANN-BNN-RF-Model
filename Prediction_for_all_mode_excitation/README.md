# Transferability to all-mode excitation

Hybrid ANN-BNN-RF prediction of the dissociation time (`tdiss`) of the dimer under
random excitation of **all** vibrational modes, using a model that was trained **only**
on trajectories from mode-specific excitation: intermolecular modes in one data set,
intramolecular modes in the other. The all-mode trajectories are never used for training
or model selection; they serve only as the test set (1500 K).

| System | Config | Train: inter + intra | Test: all-mode | Normal modes |
|---|---|---|---|---|
| Benzene dimer | `configs/benzene.json` | 1000 + 1000 trajectories | 1000 trajectories | 66 |
| Phenol dimer | `configs/phenol.json` | 400 + 400 trajectories | 400 trajectories | 72 |

## Repository layout

```
all_mode_excitation/
├── all_mode_excitation.py       workflow script (RF, BNN, hybrid, survival curves)
├── configs/                     benzene.json, phenol.json: all settings of each system
├── sample_data/<system>/        two trajectories of every data set, for a quick test
└── expected_output/<system>/    ML-predicted N(t)/N(0) curve of the manuscript
```

The full data sets are large (several GB each) and are archived on Zenodo:
**https://doi.org/10.5281/zenodo.23232903**. Download them and place them in `data/<system>/`
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
| Benzene | train, intermolecular excitation | `benzene-normal-mode-dataset-inter-1500K-1000-traj-tdiss.csv.gz` | 1000 | 2601 |
| Benzene | train, intramolecular excitation | `benzene-normal-mode-dataset-intra-1500K-1000-traj-tdiss.csv.gz` | 1000 | 2601 |
| Benzene | test, all-mode excitation | `benzene-normal-mode-dataset-all-mode-1500K-1000-traj-tdiss.csv.gz` | 1000 | 1801 |
| Phenol | train, intermolecular excitation | `phenol-normal-mode-dataset-inter-1500K-100-traj-tdiss.csv.gz` | 400 | 1601 |
| Phenol | train, intramolecular excitation | `phenol-normal-mode-dataset-intra-1500K-100-traj-tdiss.csv.gz` | 400 | 3249 |
| Phenol | test, all-mode excitation | `phenol-normal-mode-dataset-all-mode-1500K-100-traj-tdiss.csv.gz` | 400 | 1601 |

The trajectory numbers in each file start at 1. The script shifts the numbers of the
second training file, so that the 2000 (benzene) or 200 (phenol) training trajectories
are distinct.

## Run

```bash
pip install -r ../requirements.txt

# quick check on the bundled sample (two trajectories per data set; the numbers are
# not meaningful, the run only verifies the installation)
python all_mode_excitation.py --config configs/benzene.json \
    --data-dir sample_data/benzene --rf-only \
    --train-files sample-inter-2-traj.csv.gz sample-intra-2-traj.csv.gz \
    --test-file sample-all-mode-2-traj.csv.gz

# full run (after downloading the data from Zenodo into data/<system>/)
python all_mode_excitation.py --config configs/benzene.json
python all_mode_excitation.py --config configs/phenol.json
```

`--rf-only` runs the random-forest branch without TensorFlow. The full run trains the
BNN and needs TensorFlow 2.15 and TensorFlow Probability 0.23.

**Resources.** The benzene training set has about 5.2 million frames with 133 features.
The script reads the descriptors as float32; the full run nevertheless needs a large
amount of memory and several hours (500-tree random forest, BNN with up to 200 epochs).

Outputs are written to `outputs/<system>/`:

| File | Content |
|---|---|
| `trajectory_predictions.csv` | per-trajectory RF and BNN predictions, BNN uncertainty, weights, hybrid prediction |
| `predicted_survival_curve.txt` | N(t)/N(0) of the hybrid prediction (the ML-predicted curve of the manuscript) |
| `reference_survival_curve_all_mode.txt` | N(t)/N(0) of the simulated all-mode trajectories |
| `run_summary.json` | MAE, RMSE and R² of every prediction type, trajectory counts |

## Method

1. **Features and target.** `time`, `Q1…Qn`, `P1…Pn` are the inputs (133 features for
   benzene, 145 for phenol); `tdiss` is the target.
2. **Random forest.** 500 trees, maximum depth 20, `min_samples_leaf` 2, seed 42.
3. **Bayesian neural network.** Inputs (standardised) → 512 → 256 → 128 → 64 → 32 →
   Bayesian output (`DenseVariational`, Laplace prior with scale 0.1, normal
   posterior, KL weight 10⁻⁷). Mean-squared-error loss, Adam (learning rate 3·10⁻⁴),
   batch size 256, at most 200 epochs, validation split 0.2, early stopping on the
   validation MAE (patience 10) and learning-rate reduction (patience 5). The target is
   standardised for training. 200 stochastic forward passes give the mean prediction
   and the uncertainty σ.
4. **Trajectory level.** Frame predictions are reduced per trajectory: median for the
   predictions, mean for σ.
5. **Confidence-weighted hybrid.** `c_bnn = 1 / (1 + α σ)` with α = 1, `c_rf = 1`,
   normalised to weights; the prediction is the weighted sum of the BNN and RF
   predictions (equations 4–9 of the manuscript).
6. **Survival curve.** The predicted dissociation times are sorted and plotted against
   the fraction of trajectories remaining, `1 − k/n`.

The same settings are used for both systems. The BNN is stochastic (weight
initialisation, sampling), so a new run reproduces the manuscript curves closely but
not bit for bit.

## Expected output

`expected_output/<system>/ML-predicted-all-mode-1500K-*.dat` contain the ML-predicted
N(t)/N(0) curve of the manuscript (time in ps, fraction remaining).
