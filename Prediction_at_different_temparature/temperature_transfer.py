#!/usr/bin/env python3
"""
Transferability across temperature with the hybrid ANN-BNN-RF model.

Train : all-mode-excitation trajectories at the training temperature (1500 K).
Test  : independent all-mode-excitation trajectories at another temperature (1000 K).
Input : frame-level normal-mode coordinates Q1..Qn and momenta P1..Pn plus `time`
        (every fifth frame of each trajectory); target: trajectory dissociation time `tdiss`.

Workflow (as in the notebooks used for the manuscript)
  1. RF on frames, BNN (Bayesian output layer, 200 stochastic passes) on frames
  2. per-trajectory aggregation: median prediction, mean BNN uncertainty
  3. confidence-weighted hybrid, alpha = 1:  c_bnn = 1/(1 + alpha*sigma), c_rf = 1
  4. "Final" prediction: the RF prediction is used whenever the BNN uncertainty
     sigma > sigma_threshold; otherwise the hybrid is used.  The threshold was set
     manually by trial and error; set it to null in the config to use the plain hybrid.
  5. predicted survival curve = sorted predicted dissociation times vs fraction remaining.

Usage
  python temperature_transfer.py --config configs/benzene.json
  python temperature_transfer.py --config configs/benzene.json --data-dir /path/to/zenodo/data
  python temperature_transfer.py --config configs/benzene.json --rf-only   # no TensorFlow needed
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ID_COL, TARGET = "trajectory", "tdiss"


def build_bnn(n_features, n_train, cfg):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd, tfpl = tfp.distributions, tfp.layers

    def prior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=tf.zeros(n), scale=1.0), reinterpreted_batch_ndims=1))])

    def posterior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n)])

    kl = cfg["kl_weight"]
    kl = 1.0 / n_train if kl == "1/N" else float(kl)
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=(n_features,)),
        L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.30),
        L.Dense(256, activation="relu"), L.BatchNormalization(), L.Dropout(0.25),
        L.Dense(128, activation="relu"), L.BatchNormalization(),
        L.Dense(64, activation="relu"),
        L.Dense(32, activation="relu"),
        tfpl.DenseVariational(units=1, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn,
                              kl_weight=kl),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model


def train_bnn(X, y, cfg, seed):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build_bnn(X.shape[1], X.shape[0], cfg)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                           restore_best_weights=True, verbose=1),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                               patience=cfg["reduce_lr_patience"], min_lr=1e-6, verbose=1)]
    model.fit(X, y, validation_split=cfg["validation_split"], epochs=cfg["epochs"],
              batch_size=cfg["batch_size"], shuffle=True, callbacks=cb, verbose=2)
    return model


def survival_curve(values):
    t = np.sort(np.asarray(values))
    n = len(t)
    return np.column_stack([np.concatenate([[0.0], t]),
                            np.concatenate([[1.0], 1.0 - np.arange(1, n + 1) / n])])


def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--train-file", default=None, help="override train_file of the config")
    ap.add_argument("--test-file", default=None, help="override test_file of the config")
    ap.add_argument("--rf-only", action="store_true", help="RF branch only (no TensorFlow)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg["train_file"] = args.train_file or cfg["train_file"]
    cfg["test_file"] = args.test_file or cfg["test_file"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    data_dir = args.data_dir or os.path.join(root, cfg["data_dir"])
    out_dir = args.out_dir or os.path.join(root, "outputs", cfg["system"])
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg["seed"]

    train = pd.read_csv(os.path.join(data_dir, cfg["train_file"]), low_memory=False)
    test = pd.read_csv(os.path.join(data_dir, cfg["test_file"]), low_memory=False)
    print(f"train {train.shape} ({train[ID_COL].nunique()} trajectories), "
          f"test {test.shape} ({test[ID_COL].nunique()} trajectories)")

    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X_tr, y_tr, X_te = train[features], train[TARGET], test[features]

    rf = RandomForestRegressor(n_estimators=cfg["rf"]["n_estimators"], max_depth=cfg["rf"]["max_depth"],
                               min_samples_leaf=cfg["rf"]["min_samples_leaf"], random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    res = (pd.DataFrame({ID_COL: test[ID_COL].values, "RF_Prediction": rf.predict(X_te)})
           .groupby(ID_COL).agg(RF_Prediction=("RF_Prediction", "median")))
    res["Actual_tdiss"] = test.groupby(ID_COL)[TARGET].first()
    res = res.reset_index()

    final_col = "RF_Prediction"
    if not args.rf_only:
        xs, ys = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr.values.reshape(-1, 1))
        model = train_bnn(xs.transform(X_tr), ys.transform(y_tr.values.reshape(-1, 1)).ravel(), cfg["bnn"], seed)
        Xt = xs.transform(X_te)
        draws = np.array([model(Xt, training=True).numpy().ravel() for _ in range(cfg["bnn"]["mc_samples"])])
        mu = ys.inverse_transform(draws.mean(0).reshape(-1, 1)).ravel()
        sigma = draws.std(0) * ys.scale_[0]
        ann = (pd.DataFrame({ID_COL: test[ID_COL].values, "ANN_Prediction": mu, "ANN_Uncertainty": sigma})
               .groupby(ID_COL).agg(ANN_Prediction=("ANN_Prediction", "median"),
                                    ANN_Uncertainty=("ANN_Uncertainty", "mean")).reset_index())
        res = res.merge(ann, on=ID_COL, how="inner")
        alpha = cfg["alpha"]
        res["ANN_Confidence"] = 1.0 / (1.0 + alpha * res["ANN_Uncertainty"])
        res["RF_Confidence"] = 1.0
        s = res["ANN_Confidence"] + res["RF_Confidence"]
        res["ANN_Weight"], res["RF_Weight"] = res["ANN_Confidence"] / s, res["RF_Confidence"] / s
        res["Hybrid_Prediction"] = res["ANN_Weight"] * res["ANN_Prediction"] + res["RF_Weight"] * res["RF_Prediction"]
        thr = cfg.get("sigma_threshold")
        if thr is None:
            res["Final_Prediction"] = res["Hybrid_Prediction"]
        else:
            res["Final_Prediction"] = np.where(res["ANN_Uncertainty"] > thr,
                                               res["RF_Prediction"], res["Hybrid_Prediction"])
            print(f"sigma threshold {thr}: {int((res['ANN_Uncertainty'] > thr).sum())} of {len(res)} "
                  f"trajectories use the RF prediction, {int((res['ANN_Uncertainty'] <= thr).sum())} the hybrid")
        final_col = "Final_Prediction"

    res.to_csv(os.path.join(out_dir, "trajectory_predictions.csv"), index=False)
    np.savetxt(os.path.join(out_dir, "predicted_survival_curve.txt"), survival_curve(res[final_col]),
               fmt="%.8g", header="time_ps  fraction_remaining")
    np.savetxt(os.path.join(out_dir, "reference_survival_curve_test_temperature.txt"),
               survival_curve(res["Actual_tdiss"]), fmt="%.8g", header="time_ps  fraction_remaining")
    if not args.rf_only:
        np.savetxt(os.path.join(out_dir, "predicted_survival_curve_plain_hybrid.txt"),
                   survival_curve(res["Hybrid_Prediction"]), fmt="%.8g", header="time_ps  fraction_remaining")

    summary = {"system": cfg["system"], "train_temperature_K": cfg["train_temperature_K"],
               "test_temperature_K": cfg["test_temperature_K"], "rf_only": args.rf_only,
               "n_test_trajectories": int(len(res)), "n_features": len(features), "metrics": {}}
    for c in [c for c in ["RF_Prediction", "ANN_Prediction", "Hybrid_Prediction", "Final_Prediction"] if c in res]:
        summary["metrics"][c] = metrics(res["Actual_tdiss"], res[c])
        m = summary["metrics"][c]
        print(f"  {c:18s} MAE {m['MAE']:.4f}  RMSE {m['RMSE']:.4f}  R2 {m['R2']:.4f}")
    json.dump(summary, open(os.path.join(out_dir, "run_summary.json"), "w"), indent=2)
    print("Outputs written to", out_dir)


if __name__ == "__main__":
    sys.exit(main())
