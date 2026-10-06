#!/usr/bin/env python3
"""
Mode-resolved excitation analysis: hybrid RF / BNN model + Kernel SHAP.

One script for both cases studied for the benzene dimer:

  --case inter   all-intermolecular excitation (Simu3), explained by the three
                 mode-pair simulations Simu13-Simu15           (3 features)
  --case intra   all-intramolecular excitation (Simu2), explained by the nine
                 frequency-window simulations Simu4-Simu12     (9 features)

Everything that differs between the two cases (file names, labels, BNN width,
number of Monte Carlo passes inside the SHAP call, ...) lives in
configs/<case>.json. The pipeline itself is identical:

  1. load the survival-curve time columns (column 0 of every data file)
  2. standardize features and target
  3. fit a Random Forest
  4. fit a Bayesian neural network (BNN); 200 stochastic forward passes give a
     predictive mean and an uncertainty (standard deviation)
  5. confidence-weighted hybrid:  w_bnn = c_bnn / (c_bnn + c_rf),
     c_bnn = 1 / (1 + alpha * sigma),  c_rf = 1
  6. Kernel SHAP on the hybrid prediction function; mean |SHAP| per feature,
     normalized to a percentage contribution

The model is fitted to, and explained on, all N = 1001 points (this is an
attribution analysis, not a held-out prediction task).

Usage:
  python mode_resolved_shap.py --case inter
  python mode_resolved_shap.py --case intra
  python mode_resolved_shap.py --case inter --smoke-test   # quick install check
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CASE_TO_CONFIG = {"inter": "intermolecular", "intra": "intramolecular"}


# --------------------------------------------------------------------------
# command line / configuration
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid RF/BNN + SHAP analysis of mode-resolved excitation."
    )
    p.add_argument("--case", choices=sorted(CASE_TO_CONFIG), required=True,
                   help="inter = intermolecular (Simu3), intra = intramolecular (Simu2)")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON config (default: configs/<case>.json next to this script)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="folder with the data files (default: data/<case>)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output folder (default: outputs/<case>)")
    p.add_argument("--show", action="store_true",
                   help="also open the figures on screen (default: only save PNG files)")
    p.add_argument("--smoke-test", action="store_true",
                   help="tiny run (3 epochs, few MC passes, 10 explained rows) to check the "
                        "installation; the numbers it produces are NOT results")
    return p.parse_args()


def load_config(args):
    cfg_path = args.config or HERE / "configs" / f"{CASE_TO_CONFIG[args.case]}.json"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def apply_smoke_test(cfg):
    cfg["bnn"]["epochs"] = 3
    cfg["bnn"]["mc_samples"] = 5
    cfg["shap"]["mc_samples_per_call"] = 5
    cfg["shap"]["background_size"] = 20
    cfg["shap"]["max_explained_rows"] = 10
    return cfg


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def set_all_seeds(seed, tf):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(cfg, data_dir):
    """Column 0 of each file is the time (ps) at which the survival fraction
    N(t)/N(0) (column 1, not used) reaches a given level. Files are sorted in
    time, so row i of every file refers to the same survival level."""
    def col0(fname):
        return np.loadtxt(data_dir / fname, delimiter="\t", usecols=0)

    feats = cfg["features"]
    X = np.column_stack([col0(f["file"]) for f in feats])
    y = col0(cfg["full_simulation"]["file"])
    assert X.shape[0] == y.shape[0], "feature files and target file differ in length"
    assert X.shape[1] == len(feats)
    return X, y


def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def hybrid_combine(mean_pred, sigma, rf_pred, alpha):
    conf_bnn = 1.0 / (1.0 + alpha * sigma)
    conf_rf = np.ones_like(conf_bnn)
    w_bnn = conf_bnn / (conf_bnn + conf_rf)
    w_rf = conf_rf / (conf_bnn + conf_rf)
    return w_bnn * mean_pred + w_rf * rf_pred, w_bnn, w_rf


def build_bnn(n_features, n_train, bnn_cfg, tf, tfp):
    tfd = tfp.distributions
    tfpl = tfp.layers
    layers = tf.keras.layers

    def prior(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)))
        ])

    def posterior(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
            tfpl.MultivariateNormalTriL(n),
        ])

    h = bnn_cfg["hidden_units"]
    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(h[0], activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(bnn_cfg["dropout"]),
        *[layers.Dense(u, activation="relu") for u in h[1:]],
        tfpl.DenseVariational(
            units=1,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1.0 / n_train,
        ),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=bnn_cfg["learning_rate"]),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def mc_forward(model, X, n_passes):
    """Stochastic forward passes (training=True keeps dropout and the
    variational weight sampling active)."""
    out = np.zeros((n_passes, X.shape[0]))
    for i in range(n_passes):
        out[i] = model(X, training=True).numpy().flatten()
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args)
    if args.smoke_test:
        cfg = apply_smoke_test(cfg)
        print("=" * 60)
        print("SMOKE TEST: tiny run to check the installation.")
        print("The numbers produced are NOT results.")
        print("=" * 60)

    data_dir = args.data_dir or HERE / "data" / args.case
    out_dir = args.out_dir or HERE / "outputs" / (args.case + ("_smoke-test" if args.smoke_test else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # heavy imports only after the arguments are parsed (so --help is fast)
    import joblib
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import sklearn
    import tensorflow as tf
    import tensorflow_probability as tfp
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    seed = cfg["seed"]
    set_all_seeds(seed, tf)
    t_start = time.time()

    # 1. data ----------------------------------------------------------------
    X, y = load_data(cfg, data_dir)
    labels = [f["label"] for f in cfg["features"]]
    print(f"Case: {cfg['case']}  |  X shape: {X.shape}  |  y shape: {y.shape}")

    # 2. scaling -------------------------------------------------------------
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
    joblib.dump(x_scaler, out_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, out_dir / "y_scaler.pkl")

    # 3. random forest -------------------------------------------------------
    rf_cfg = cfg["random_forest"]
    rf_model = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1,
    )
    rf_model.fit(X_scaled, y)
    rf_prediction = rf_model.predict(X_scaled)
    rf_metrics = regression_metrics(y, rf_prediction)

    # 4. Bayesian neural network ----------------------------------------------
    bnn_cfg = cfg["bnn"]
    model = build_bnn(X_scaled.shape[1], X_scaled.shape[0], bnn_cfg, tf, tfp)
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=bnn_cfg["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        X_scaled, y_scaled,
        epochs=bnn_cfg["epochs"],
        batch_size=bnn_cfg["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    # 5. Monte Carlo predictive mean and uncertainty --------------------------
    if bnn_cfg["reseed_before_mc"]:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    draws = mc_forward(model, X_scaled, bnn_cfg["mc_samples"])
    mean_prediction = y_scaler.inverse_transform(
        draws.mean(axis=0).reshape(-1, 1)).flatten()
    uncertainty = draws.std(axis=0) * y_scaler.scale_[0]
    bnn_metrics = regression_metrics(y, mean_prediction)

    # 6. hybrid ---------------------------------------------------------------
    alpha = cfg["hybrid"]["alpha"]
    hybrid_prediction, w_bnn, w_rf = hybrid_combine(
        mean_prediction, uncertainty, rf_prediction, alpha)
    hybrid_metrics = regression_metrics(y, hybrid_prediction)

    performance = pd.DataFrame(
        [rf_metrics, bnn_metrics, hybrid_metrics],
        index=["Random Forest", "Bayesian Neural Network", "Hybrid"],
    )
    performance.index.name = "Model"
    performance.to_csv(out_dir / "model_performance.csv")
    print("\n" + performance.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nBNN uncertainty: mean={uncertainty.mean():.4f} "
          f"min={uncertainty.min():.4f} max={uncertainty.max():.4f}")
    print(f"BNN weight: min={w_bnn.min():.4f} max={w_bnn.max():.4f} mean={w_bnn.mean():.4f}")

    pd.DataFrame({
        "target_time": y,
        "RF_prediction": rf_prediction,
        "BNN_mean_prediction": mean_prediction,
        "BNN_uncertainty": uncertainty,
        "BNN_weight": w_bnn,
        "RF_weight": w_rf,
        "Hybrid_prediction": hybrid_prediction,
    }).to_csv(out_dir / "predictions.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y, hybrid_prediction, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Hybrid prediction")
    plt.title("Hybrid model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "parity_hybrid.png", dpi=300)
    if args.show:
        plt.show()
    plt.close()

    # 7. SHAP on the hybrid prediction function ---------------------------------
    shap_cfg = cfg["shap"]

    def hybrid_predict(X_input):
        rf_pred = rf_model.predict(X_input)
        # reseeded on every call so that SHAP sees a deterministic function
        np.random.seed(seed)
        tf.random.set_seed(seed)
        d = mc_forward(model, X_input, shap_cfg["mc_samples_per_call"])
        mean_pred = y_scaler.inverse_transform(d.mean(axis=0).reshape(-1, 1)).flatten()
        sigma = d.std(axis=0) * y_scaler.scale_[0]
        return hybrid_combine(mean_pred, sigma, rf_pred, alpha)[0]

    background = shap.sample(X_scaled, shap_cfg["background_size"], random_state=seed)
    explainer = shap.KernelExplainer(hybrid_predict, background)
    n_explain = shap_cfg.get("max_explained_rows", X_scaled.shape[0])
    X_explain = X_scaled[:n_explain]
    print(f"\nKernel SHAP on {X_explain.shape[0]} rows "
          f"(this is the slow step for the full run)...")
    shap_values = explainer.shap_values(X_explain)

    shap.summary_plot(shap_values, X_explain, feature_names=labels, show=False)
    plt.savefig(out_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()

    shap.summary_plot(shap_values, X_explain, feature_names=labels,
                      plot_type="bar", show=False)
    plt.savefig(out_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()

    # 8. contributions ---------------------------------------------------------
    importance = np.mean(np.abs(shap_values), axis=0)
    percentage = importance / importance.sum() * 100.0
    contribution = pd.DataFrame({
        "Simulation": [f["simulation"] for f in cfg["features"]],
        "Label": labels,
        "File": [f["file"] for f in cfg["features"]],
        "Mean |SHAP|": importance,
        "Contribution (%)": percentage,
    })
    contribution.to_csv(out_dir / "Hybrid_SHAP_Contribution.csv", index=False)
    print("\n" + contribution.sort_values("Contribution (%)", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plt.figure(figsize=(max(6, 0.7 * len(labels) + 3), 4.5))
    plt.bar(contribution["Label"], contribution["Contribution (%)"])
    plt.ylabel("Contribution (%)")
    plt.xlabel(cfg["plots"]["contribution_xlabel"])
    plt.title(cfg["plots"]["contribution_title"])
    if len(labels) > 4:
        plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_contribution.png", dpi=300)
    if args.show:
        plt.show()
    plt.close()

    # 9. run record -------------------------------------------------------------
    summary = {
        "case": cfg["case"],
        "smoke_test": bool(args.smoke_test),
        "config_file": cfg["_config_path"],
        "data_dir": str(data_dir),
        "n_points": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "runtime_seconds": round(time.time() - t_start, 1),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "shap": shap.__version__,
        },
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDone in {summary['runtime_seconds']} s. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
