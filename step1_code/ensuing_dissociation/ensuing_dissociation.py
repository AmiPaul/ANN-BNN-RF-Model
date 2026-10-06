#!/usr/bin/env python3
"""
Prediction of ensuing dissociation with the hybrid ANN-BNN-RF model.

Train : unimolecular dissociation trajectories (frame-level data).
Test  : association trajectories at one or more impact parameters (b).
Target: tdiss, the dissociation time of the trajectory a frame belongs to.

Workflow (identical to the notebooks used for the manuscript)
  1. clean the test set (numeric conversion, drop invalid rows)
  2. keep frames with COM <= com_threshold
  3. RF (500 trees) and BNN (Bayesian output layer, 200 stochastic passes)
  4. per-trajectory aggregation (median prediction; mean BNN uncertainty)
  5. confidence-weighted hybrid:  w_bnn = c_bnn/(c_bnn + c_rf),
        c_bnn = 1/(1 + alpha*sigma),  c_rf = 1
  6. predicted survival curve = sorted predicted dissociation times
     versus the fraction of trajectories remaining.

Usage
  python ensuing_dissociation.py --config configs/benzene.json
  python ensuing_dissociation.py --config configs/benzene.json --b 0 6
  python ensuing_dissociation.py --config configs/benzene.json --rf-only   # no TensorFlow needed
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["time", "angle", "com", "ke", "pe"]
TARGET = "tdiss"


def load_train(paths, com_threshold):
    if isinstance(paths, str):
        paths = [paths]
    train = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    frames = train[train["com"] <= com_threshold].copy()
    print(f"Training file: {train.shape[0]} rows -> {frames.shape[0]} frames with COM <= {com_threshold}")
    return frames


def load_test(path, com_threshold):
    test = pd.read_csv(path, low_memory=False)
    n_rows, n_traj = len(test), test["trajectory"].nunique()
    # header rows repeated inside the file and non-numeric entries become NaN
    for col in test.columns:
        test[col] = pd.to_numeric(test[col], errors="coerce")
    test = test.dropna().reset_index(drop=True)
    dropped = n_rows - len(test)
    frames = test[test["com"] <= com_threshold].copy()
    n_traj_kept = frames["trajectory"].nunique()
    lost = n_traj - n_traj_kept
    print(f"  test rows {n_rows} -> {len(test)} after cleaning ({dropped} invalid rows dropped)")
    print(f"  {frames.shape[0]} frames, {n_traj_kept} trajectories ({lost} trajectory(ies) with no valid frame)")
    return frames, {"rows_dropped": int(dropped), "trajectories_total": int(n_traj),
                    "trajectories_used": int(n_traj_kept)}


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

    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=(n_features,)),
        L.Dense(512, activation="relu"), L.BatchNormalization(), L.Dropout(0.30),
        L.Dense(256, activation="relu"), L.BatchNormalization(), L.Dropout(0.25),
        L.Dense(128, activation="relu"), L.BatchNormalization(),
        L.Dense(64, activation="relu"),
        L.Dense(32, activation="relu"),
        tfpl.DenseVariational(units=1, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn,
                              kl_weight=1 / n_train),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model


def train_bnn(X_train, y_train, cfg, seed):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build_bnn(X_train.shape[1], X_train.shape[0], cfg)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=cfg["reduce_lr_patience"], min_lr=1e-6, verbose=1),
    ]
    model.fit(X_train, y_train, validation_split=cfg["validation_split"], epochs=cfg["epochs"],
              batch_size=cfg["batch_size"], shuffle=True, callbacks=callbacks, verbose=2)
    return model


def hybrid_combine(ann, sigma, rf, alpha):
    conf_ann = 1.0 / (1.0 + alpha * sigma)
    conf_rf = np.ones_like(conf_ann)
    w_ann = conf_ann / (conf_ann + conf_rf)
    return w_ann * ann + (1.0 - w_ann) * rf, w_ann


def survival_curve(pred_times):
    """(0, 1) followed by sorted predicted times t_(k) with fraction 1 - k/n."""
    t = np.sort(np.asarray(pred_times))
    n = len(t)
    frac = 1.0 - np.arange(1, n + 1) / n
    return np.column_stack([np.concatenate([[0.0], t]), np.concatenate([[1.0], frac])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None, help="override data directory of the config")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--b", nargs="*", default=None, help="impact parameters to run (default: all in config)")
    ap.add_argument("--rf-only", action="store_true", help="run only the RF branch (no TensorFlow)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(args.config))
    with open(args.config) as f:
        cfg = json.load(f)
    root = os.path.dirname(base)
    data_dir = args.data_dir or os.path.join(root, cfg["data_dir"])
    out_dir = args.out_dir or os.path.join(root, "outputs", cfg["system"])
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg["seed"]
    com_thr = cfg["com_threshold"]
    alpha = cfg["alpha"]
    b_values = [str(b) for b in (args.b if args.b else cfg["tests"].keys())]

    train_files = cfg["train_file"] if isinstance(cfg["train_file"], list) else [cfg["train_file"]]
    train = load_train([os.path.join(data_dir, p) for p in train_files], com_thr)
    X_tr, y_tr = train[FEATURES], train[TARGET]

    rf = RandomForestRegressor(n_estimators=cfg["rf"]["n_estimators"], max_depth=cfg["rf"]["max_depth"],
                               min_samples_leaf=cfg["rf"]["min_samples_leaf"], random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    model = None
    if not args.rf_only:
        x_scaler = StandardScaler().fit(X_tr)
        y_scaler = StandardScaler().fit(y_tr.values.reshape(-1, 1))
        model = train_bnn(x_scaler.transform(X_tr), y_scaler.transform(y_tr.values.reshape(-1, 1)).ravel(),
                          cfg["bnn"], seed)

    summary = {"system": cfg["system"], "alpha": alpha, "com_threshold": com_thr, "seed": seed,
               "rf_only": args.rf_only, "tests": {}}
    for b in b_values:
        print(f"\n=== {cfg['system']}  b = {b} ===")
        frames, info = load_test(os.path.join(data_dir, cfg["tests"][b]), com_thr)
        rf_traj = (pd.DataFrame({"trajectory": frames["trajectory"].values,
                                 "RF_Prediction": rf.predict(frames[FEATURES])})
                   .groupby("trajectory").agg(RF_Prediction=("RF_Prediction", "median")))
        actual = frames.groupby("trajectory")[TARGET].first().rename("Actual_tdiss")
        res = pd.concat([actual, rf_traj], axis=1).reset_index()

        if model is not None:
            Xs = x_scaler.transform(frames[FEATURES])
            draws = np.array([model(Xs, training=True).numpy().ravel() for _ in range(cfg["bnn"]["mc_samples"])])
            mu = y_scaler.inverse_transform(draws.mean(0).reshape(-1, 1)).ravel()
            sigma = draws.std(0) * y_scaler.scale_[0]
            ann_traj = (pd.DataFrame({"trajectory": frames["trajectory"].values,
                                      "ANN_Prediction": mu, "ANN_Uncertainty": sigma})
                        .groupby("trajectory").agg(ANN_Prediction=("ANN_Prediction", "median"),
                                                   ANN_Uncertainty=("ANN_Uncertainty", "mean")))
            res = res.merge(ann_traj.reset_index(), on="trajectory", how="inner")
            res["Hybrid_Prediction"], res["ANN_Weight"] = hybrid_combine(
                res["ANN_Prediction"].values, res["ANN_Uncertainty"].values,
                res["RF_Prediction"].values, alpha)
            res["RF_Weight"] = 1.0 - res["ANN_Weight"]
            final = "Hybrid_Prediction"
        else:
            final = "RF_Prediction"

        res.to_csv(os.path.join(out_dir, f"b{b}_trajectory_predictions.csv"), index=False)
        np.savetxt(os.path.join(out_dir, f"b{b}_predicted_survival_curve.txt"),
                   survival_curve(res[final].values), fmt="%.7f", header="time_ps  fraction_remaining")
        np.savetxt(os.path.join(out_dir, f"b{b}_reference_survival_curve.txt"),
                   survival_curve(res["Actual_tdiss"].values), fmt="%.7f", header="time_ps  fraction_remaining")

        metrics = {}
        for col in [c for c in ["RF_Prediction", "ANN_Prediction", "Hybrid_Prediction"] if c in res]:
            metrics[col] = {"MAE": float(mean_absolute_error(res["Actual_tdiss"], res[col])),
                            "RMSE": float(np.sqrt(mean_squared_error(res["Actual_tdiss"], res[col]))),
                            "R2": float(r2_score(res["Actual_tdiss"], res[col]))}
            print(f"  {col:18s} MAE {metrics[col]['MAE']:.4f}  RMSE {metrics[col]['RMSE']:.4f}  R2 {metrics[col]['R2']:.4f}")
        summary["tests"][b] = {**info, "n_trajectories_evaluated": int(len(res)), "metrics": metrics}

    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
