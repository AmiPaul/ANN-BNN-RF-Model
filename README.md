# Hybrid ANN-BNN-RF Model

## Methodology

An **ANN–BNN–RF hybrid machine-learning framework** was developed to predict molecular dissociation time. The ANN–BNN model provides both the **predicted mean and associated uncertainty**, with uncertainty estimated using **200 Monte Carlo stochastic predictions**. An independently trained **Random Forest (RF)** model uses the same input features and target variable to provide an additional nonlinear prediction.

The ANN–BNN uncertainty is converted into a **confidence score**, where lower uncertainty corresponds to higher confidence. The confidence values are normalized to determine the respective weights of the ANN–BNN and RF models. The final dissociation-time prediction is obtained through a **confidence-weighted combination** of the ANN–BNN mean prediction and RF prediction.

The models are trained independently, and the hybrid prediction is generated only after obtaining the individual model predictions. The performance of the **ANN–BNN, RF, and hybrid models** is evaluated using **MAE, RMSE, and R²** on an independent test set to assess predictive accuracy and generalization capability.
