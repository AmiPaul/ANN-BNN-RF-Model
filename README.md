# Hybrid ANN-BNN-RF Model

<img width="828" height="585" alt="image" src="https://github.com/user-attachments/assets/c54f50b1-dff5-4833-bd14-3d66b06d405b" />

Code, configuration files and processed machine-learning data accompanying the study

> **Transferable and Interpretable Machine Learning Model for Predicting Vibrationally Driven Dissociation Dynamics in Benzene Dimer and Phenol Dimer Complexes**
> Souvik Shaw\*, Basudha Deb\*, Marry Hazarika and Amit Kumar Paul (\* equal contribution)

Everything needed to repeat the machine-learning part of the study is in this repository:
one script and one set of configuration files per workflow, the processed data (or a small
sample of it, see [Data](#data)), and the reference output of the manuscript runs.

## Methodology

An **ANN–BNN–RF hybrid machine-learning framework** was developed to predict molecular dissociation time. The ANN–BNN model provides both the **predicted mean and associated uncertainty**, with uncertainty estimated using **200 Monte Carlo stochastic predictions**. An independently trained **Random Forest (RF)** model uses the same input features and target variable to provide an additional nonlinear prediction.

The ANN–BNN uncertainty is converted into a **confidence score**, where lower uncertainty corresponds to higher confidence. The confidence values are normalized to determine the respective weights of the ANN–BNN and RF models. The final dissociation-time prediction is obtained through a **confidence-weighted combination** of the ANN–BNN mean prediction and RF prediction.

With σ the BNN uncertainty of a trajectory, the confidences are `c_bnn = 1 / (1 + α σ)` and `c_rf = 1` (α = 1); they are normalized to weights and the hybrid prediction is the weighted sum of the BNN and RF predictions.

The models are trained independently, and the hybrid prediction is generated only after obtaining the individual model predictions. The performance of the **ANN–BNN, RF, and hybrid models** is evaluated using **MAE, RMSE, and R²** on an independent test set to assess predictive accuracy and generalization capability. The mode-resolved excitation workflow is different in purpose: the models are fitted to, and explained on, all data points to attribute a full simulation to its component simulations, so no held-out test set is used there.

## Workflows

The study consists of four workflows. Each one lives in its own folder with its own README
(data description, run instructions, method details, expected results).

| # | Workflow | Systems | Training data | Test data | Folder |
|---|---|---|---|---|---|
| 1 | **Mode-resolved excitation**: attribution of a full simulation to its component simulations (hybrid model + Kernel SHAP) | benzene dimer | survival curves of the component simulations | none (fit and explain on all points) | [`mode_resolved_excitation/`](mode_resolved_excitation/) |
| 2 | **Ensuing dissociation**: dimers formed by association collisions | benzene dimer, phenol dimer | unimolecular-dissociation trajectories, 1500 K | association trajectories at impact parameters b = 0, 2, 4, 6 | [`ensuing_dissociation/`](ensuing_dissociation/) |
| 3 | **Transferability across temperature** | benzene dimer, phenol dimer | all-mode excitation, 1500 K | all-mode excitation, 1000 K | [`Prediction_at_different_temparature/`](Prediction_at_different_temparature/) |
| 4 | **Transferability to all-mode excitation** | benzene dimer, phenol dimer | intermolecular-mode and intramolecular-mode excitation, 1500 K | all-mode excitation, 1500 K | [`Prediction_for_all_mode_excitation/`](Prediction_for_all_mode_excitation/) |

Workflows 2–4 use the same hybrid ANN-BNN-RF scheme; they differ in the input features, the
training and test sets and a few model settings, which are all stored in the `configs/`
folder of the workflow (and listed in its README). Benzene and phenol share one script per
workflow and differ only in the configuration file.

| Workflow | Script | Input features | Target |
|---|---|---|---|
| 1 | `mode_resolved_shap.py` | times at which the component simulations reach a given survival level | time at which the full simulation reaches the same level |
| 2 | `ensuing_dissociation.py` | `time, angle, com, ke, pe` | trajectory dissociation time `tdiss` |
| 3 | `temperature_transfer.py` | `time`, normal-mode coordinates `Q1…Qn` and momenta `P1…Pn` | `tdiss` |
| 4 | `all_mode_excitation.py` | `time`, normal-mode coordinates `Q1…Qn` and momenta `P1…Pn` | `tdiss` |

## Repository structure

```
.
├── README.md
├── LICENSE                              MIT License (code)
├── requirements.txt                     Python packages
│
├── mode_resolved_excitation/            workflow 1
│   ├── mode_resolved_shap.py
│   ├── configs/                         intermolecular.json, intramolecular.json
│   ├── data/                            survival curves (inter/, intra/)
│   ├── expected_output/                 contributions of the manuscript
│   └── README.md
├── ensuing_dissociation/                workflow 2
│   ├── ensuing_dissociation.py
│   ├── configs/                         benzene.json, phenol.json
│   ├── data/                            benzene/, phenol/ (training and test files)
│   └── README.md
├── Prediction_at_different_temparature/ workflow 3
│   ├── temperature_transfer.py
│   ├── configs/                         benzene.json, phenol.json
│   ├── sample_data/                     two trajectories per temperature and system
│   ├── expected_output/                 ML-predicted curves of the manuscript
│   └── README.md
└── Prediction_for_all_mode_excitation/  workflow 4
    ├── all_mode_excitation.py
    ├── configs/                         benzene.json, phenol.json
    ├── sample_data/                     two trajectories of every data set
    ├── expected_output/                 ML-predicted curves of the manuscript
    └── README.md
```

## Data

All data are processed machine-learning data (normal-mode coordinates and momenta, or
trajectory-level quantities such as COM separation and energies, with the dissociation time
as target). **Raw trajectories and the simulation input files are not distributed**; they
are not needed to reproduce the machine-learning results.

| Workflow | Where the data are |
|---|---|
| 1 Mode-resolved excitation | complete, in [`mode_resolved_excitation/data/`](mode_resolved_excitation/data/) |
| 2 Ensuing dissociation | complete, in [`ensuing_dissociation/data/`](ensuing_dissociation/data/) (gzip-compressed; the phenol training set is split into three parts) |
| 3 Transferability across temperature | small sample (two trajectories per file) in `sample_data/`, for a quick test of the installation; the data sets of the workflow are on Zenodo |
| 4 Transferability to all-mode excitation | small sample (two trajectories per file) in `sample_data/`, for a quick test of the installation; the data sets of the workflow are on Zenodo |

Zenodo record for workflows 3 and 4: **https://doi.org/10.5281/zenodo.23232903** (data
license: CC BY 4.0). Download the files, place them in `data/<system>/` of the workflow
folder, or pass `--data-dir` to the script. The record's description lists every file with
its role, trajectory numbers and SHA-256 checksum.

## Installation

Python 3.10 is recommended. The packages are listed in [`requirements.txt`](requirements.txt)
(TensorFlow 2.15.0, TensorFlow Probability 0.23.0, NumPy < 2, pandas, scikit-learn, joblib,
matplotlib, SHAP).

```bash
git clone https://github.com/AmiPaul/ANN-BNN-RF-Model.git
cd ANN-BNN-RF-Model
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

## Quick start

Each script has a fast check that verifies the installation without TensorFlow or a long
run. The numbers it produces are not results.

```bash
# workflow 3: random-forest branch on the bundled two-trajectory sample
cd Prediction_at_different_temparature
python temperature_transfer.py --config configs/benzene.json \
    --data-dir sample_data/benzene --rf-only \
    --train-file sample-1500K-2-traj.csv.gz --test-file sample-1000K-2-traj.csv.gz

# workflow 1: 3 epochs, few Monte Carlo passes, 10 explained rows
cd ../mode_resolved_excitation
python mode_resolved_shap.py --case inter --smoke-test
```

Full runs, per workflow:

```bash
python mode_resolved_excitation/mode_resolved_shap.py --case inter        # and --case intra
python ensuing_dissociation/ensuing_dissociation.py --config ensuing_dissociation/configs/benzene.json
python Prediction_at_different_temparature/temperature_transfer.py --config Prediction_at_different_temparature/configs/benzene.json
python Prediction_for_all_mode_excitation/all_mode_excitation.py --config Prediction_for_all_mode_excitation/configs/benzene.json
```

(use `phenol.json` for the phenol dimer; each workflow README lists all options, such as
`--rf-only`, `--data-dir` and `--out-dir`). Outputs are written to an `outputs/` folder
inside the workflow folder, which is not tracked by git.

## Reproducibility notes

- All random seeds are fixed (42). The BNN is nevertheless stochastic (weight initialisation,
  sampling, and on some hardware non-deterministic operations), so a new run reproduces the
  manuscript curves closely but not bit for bit.
- The reference output of the manuscript runs (ML-predicted curves, contribution values) is
  in the `expected_output/` folders.
- Some settings differ between systems and are therefore kept in the configuration files,
  not in the code. Examples are the BNN settings of the temperature workflow, whose
  uncertainty threshold for falling back to the random forest was chosen by trial and
  error for each system (see the README of that workflow).
- Resources: the full all-mode run for the benzene dimer uses about 5.2 million training
  frames and needs a large amount of memory and several hours; the Kernel SHAP step of the
  intramolecular mode-resolved analysis took about 2 h 14 min.

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The data deposited on
Zenodo are released under CC BY 4.0.

## Contact

Corresponding author: Amit Kumar Paul, Department of Chemical Sciences, Bose Institute,
EN 80, Sector V, Kolkata 700091, India. Questions about the code can also be raised as
[GitHub issues](https://github.com/AmiPaul/ANN-BNN-RF-Model/issues).
