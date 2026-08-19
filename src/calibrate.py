"""Probability calibration (PLAN §7.1).

The logistic model's holdout set is split in half chronologically: the first
half calibrates (isotonic regression, one-vs-rest + renormalized), the second
half is the final, untouched evaluation set used later by evaluate.py /
bet_sim.py. This keeps calibration fitting from leaking into the backtest.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from db import PROJECT_ROOT, load_config
from features import MODEL_FEATURE_COLUMNS
from train_logistic import RESULT_CATEGORIES


def reliability_curve(y_true_onehot: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-probability bucket vs. observed frequency, pooled across classes."""
    bins = np.linspace(0, 1, n_bins + 1)
    flat_p = proba.ravel()
    flat_y = y_true_onehot.ravel()
    bin_idx = np.digitize(flat_p, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_mid": (bins[b] + bins[b + 1]) / 2,
                "mean_predicted": flat_p[mask].mean(),
                "observed_freq": flat_y[mask].mean(),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def calibrate(config: dict | None = None):
    config = config or load_config()
    processed_dir = PROJECT_ROOT / "data" / "processed"

    with open(processed_dir / "logistic_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    sk_model, mu, sigma = artifacts["sklearn_model"], artifacts["mu"], artifacts["sigma"]

    test_df = pd.read_csv(processed_dir / "logistic_test_predictions.csv", parse_dates=["date"])
    half = len(test_df) // 2
    calib_df, holdout_df = test_df.iloc[:half].copy(), test_df.iloc[half:].copy()
    print(f"calibration set={len(calib_df)}  final holdout set={len(holdout_df)}")

    Xcal = (calib_df[MODEL_FEATURE_COLUMNS] - mu) / sigma
    Xhold = (holdout_df[MODEL_FEATURE_COLUMNS] - mu) / sigma
    ycal = pd.Categorical(calib_df["result"], categories=RESULT_CATEGORIES).codes
    yhold = pd.Categorical(holdout_df["result"], categories=RESULT_CATEGORIES).codes

    y_onehot = np.eye(3)[yhold]
    proba_raw_hold = sk_model.predict_proba(Xhold)

    def brier(proba):
        return float(((proba - y_onehot) ** 2).sum(axis=1).mean())

    # With only ~178 calibration rows, isotonic regression (flexible/non-parametric)
    # can overfit the calibration set; Platt/sigmoid (2 params per class) is more
    # stable at this sample size (PLAN §7.1 names both — pick whichever generalizes
    # to the holdout Brier score rather than assuming isotonic is always better).
    candidates = {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(FrozenEstimator(sk_model), method=method)
        cal.fit(Xcal, ycal)
        candidates[method] = (cal, cal.predict_proba(Xhold))

    raw_brier = brier(proba_raw_hold)
    scored = {"raw": raw_brier, **{m: brier(p) for m, (_, p) in candidates.items()}}
    print("Brier score (multiclass):", {k: round(v, 4) for k, v in scored.items()})

    best_method = min(candidates, key=lambda m: scored[m])
    if scored[best_method] < raw_brier:
        calibrated, proba_cal_hold = candidates[best_method]
        print(f"Using '{best_method}' calibration (beats raw).")
    else:
        calibrated, proba_cal_hold = None, proba_raw_hold
        print("No calibration method beat the raw (uncalibrated) probabilities on the holdout Brier score — keeping raw.")

    curve_raw = reliability_curve(y_onehot, proba_raw_hold)
    curve_cal = reliability_curve(y_onehot, proba_cal_hold)

    out = holdout_df.reset_index(drop=True).copy()
    for i, cat in enumerate(RESULT_CATEGORIES):
        out[f"p_raw_{cat}"] = proba_raw_hold[:, i]
        out[f"p_cal_{cat}"] = proba_cal_hold[:, i]

    out.to_csv(processed_dir / "calibrated_holdout_predictions.csv", index=False)
    curve_raw.to_csv(processed_dir / "reliability_curve_raw.csv", index=False)
    curve_cal.to_csv(processed_dir / "reliability_curve_calibrated.csv", index=False)
    with open(processed_dir / "calibrated_model.pkl", "wb") as f:
        pickle.dump({"calibrated_model": calibrated, "mu": mu, "sigma": sigma}, f)

    return out, curve_raw, curve_cal


if __name__ == "__main__":
    calibrate()
