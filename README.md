<div align="center">

# Hybrid ANN-BNN-RF Model


# Methodology
An ANN–BNN–RF hybrid machine-learning framework was developed to predict molecular dissociation time. The ANN–BNN model was trained to provide both the predicted mean and associated uncertainty, with uncertainty estimated from 200 Monte Carlo stochastic predictions. A Random Forest (RF) model was trained independently using the same input features and target variable to provide an additional nonlinear prediction.
The uncertainty obtained from the ANN–BNN was converted into a confidence score, where lower uncertainty corresponds to higher confidence. The ANN–BNN and RF confidence values were then normalized to obtain their respective weights. The final dissociation-time prediction was calculated as a weighted combination of the ANN–BNN mean prediction and RF prediction, allowing the contribution of each model to vary according to the ANN–BNN uncertainty.
The individual models were trained independently, and the hybrid prediction was constructed only after obtaining their predictions. The performance of the ANN–BNN, RF, and hybrid models was evaluated using MAE, RMSE, and \(R^2\) on an independent test set to assess their predictive accuracy and generalization capability.
