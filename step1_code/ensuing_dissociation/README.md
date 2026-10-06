# Prediction of ensuing dissociation

Hybrid ANN-BNN-RF prediction of the dissociation time (`tdiss`) of dimers formed by
association collisions, trained only on unimolecular-dissociation trajectories and
tested at four impact parameters (b = 0, 2, 4, 6).

| System | Config | Train | Test |
|---|---|---|---|
| Benzene dimer | `configs/benzene.json` | 1000 trajectories, 1500 K | b0, b2, b4, b6 |
| Phenol dimer | `configs/phenol.json` | 597 trajectories, 1500 K | b0, b2, b4, b6 |

## Data (`data/<system>/`)

Frame-level processed ML data, columns `trajectory,time,angle,com,ke,pe,tdiss`
(time in ps, COM separation in Angstrom, angle in degrees, ke/pe in kcal/mol).

- `train_unimolecular_1500K_com-le-16A.csv.gz` contains only the frames with COM <= 16 Å,
  i.e. exactly the frames the model uses (the complete training file is 832 MB).
  The filter is a plain row selection; no values were changed.
- All data files are gzip-compressed (`.csv.gz`) to keep the repository small; pandas reads them directly, and `gunzip` gives the plain CSV. The phenol training set (597 trajectories) is split into three parts (`..._part1/2/3.csv.gz`, split at trajectory boundaries) so that each file is small; the script concatenates them.
- `test_ensuing_b{0,2,4,6}.csv.gz` are the association trajectories at each impact parameter.

Raw trajectories and simulation inputs are not distributed; they are not needed to reproduce the results.

## Run

```bash
pip install -r ../requirements.txt
python ensuing_dissociation.py --config configs/benzene.json            # all b
python ensuing_dissociation.py --config configs/benzene.json --b 0 6    # selected b
python ensuing_dissociation.py --config configs/benzene.json --rf-only  # RF branch only, no TensorFlow
```

Outputs are written to `outputs/<system>/`:
`b<b>_trajectory_predictions.csv` (per-trajectory RF, BNN, uncertainty, weights, hybrid),
`b<b>_predicted_survival_curve.txt`, `b<b>_reference_survival_curve.txt` (time in ps, fraction remaining),
and `run_summary.json` (MAE, RMSE, R², counts of dropped rows/trajectories).
The predicted curves are the raw hybrid model output; no offset or post-processing is applied.

## Method

1. The test files are converted to numeric and invalid rows are dropped; frames with COM <= 16 Å are kept
   (the dissociation criterion is COM = 14 Å).
2. Features: `time, angle, com, ke, pe`; target: `tdiss`.
3. RF: 500 trees, depth 20, `min_samples_leaf` 2, seed 42.
4. BNN: 5→512→256→128→64→32→Bayesian (`DenseVariational`) output, batch norm and dropout, Huber loss,
   Adam 3e-4, validation split 0.2, batch 1024, early stopping (patience 20), learning-rate reduction;
   200 stochastic forward passes give the mean and σ.
5. Frame predictions are reduced per trajectory (median; mean σ).
6. Hybrid: `c_bnn = 1/(1+ασ)`, `c_rf = 1`, normalised weights, α = 1.
7. Survival curve: sorted predicted times versus `1 − k/n`.

## Notes on trajectory counts

Some test files contain repeated header rows or rows with missing values; these are removed by the cleaning step, and trajectories left without valid frames are excluded (benzene: b0 loses 1, b4 loses 2); the
numbers are printed at run time and stored in `run_summary.json`.
