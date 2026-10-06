# Hybrid ANN-BNN-RF Model

<img width="828" height="585" alt="image" src="https://github.com/user-attachments/assets/c54f50b1-dff5-4833-bd14-3d66b06d405b" />

Code and data accompanying the study on machine-learning prediction of
vibrationally driven dissociation dynamics in benzene dimer and phenol dimer
complexes.

## Methodology

An **ANN–BNN–RF hybrid machine-learning framework** was developed to predict molecular dissociation time. The ANN–BNN model provides both the **predicted mean and associated uncertainty**, with uncertainty estimated using **200 Monte Carlo stochastic predictions**. An independently trained **Random Forest (RF)** model uses the same input features and target variable to provide an additional nonlinear prediction.

The ANN–BNN uncertainty is converted into a **confidence score**, where lower uncertainty corresponds to higher confidence. The confidence values are normalized to determine the respective weights of the ANN–BNN and RF models. The final dissociation-time prediction is obtained through a **confidence-weighted combination** of the ANN–BNN mean prediction and RF prediction.

The models are trained independently, and the hybrid prediction is generated only after obtaining the individual model predictions. The performance of the **ANN–BNN, RF, and hybrid models** is evaluated using **MAE, RMSE, and R²** on an independent test set to assess predictive accuracy and generalization capability. The mode-resolved excitation workflow is different in purpose: the models are fitted to, and explained on, all data points to attribute a full simulation to its component simulations, so no held-out test set is used there.

## Workflows

| Workflow | Systems | Location |
|---|---|---|
| Prediction of ensuing dissociation | benzene dimer, phenol dimer | `Prediction of ensuing dissociation` |
| Prediction at different temperatures | benzene dimer, phenol dimer | `Prediction-at-different-temperature` |
| Prediction for all-mode excitation | benzene dimer, phenol dimer | `Prediction-of-all-mode-excitation` |
| Mode-resolved excitation (SHAP attribution), intramolecular and intermolecular | benzene dimer | [`mode_resolved_excitation/`](mode_resolved_excitation/) |

The mode-resolved excitation workflow has its own README with the data
description, configuration files, run instructions and expected results.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── mode_resolved_excitation/      mode-resolved excitation workflow (code, configs, data)
├── Prediction of ensuing dissociation
├── Prediction-at-different-temperature
└── Prediction-of-all-mode-excitation
```

## Installation

Python 3.10 is recommended.

```
pip install -r requirements.txt
```

## Citation

A. K. Paul, ANN-BNN-RF Model codes, GitHub (2026),
https://github.com/AmiPaul/ANN-BNN-RF-Model

## License

Released under the MIT License; see [LICENSE](LICENSE).
