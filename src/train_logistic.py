"""Multinomial logistic regression: P(home win / draw / away win) (PLAN §5.1, §10.8).

This is the model that feeds the Kalshi EV strategy directly. Fits both a
statsmodels MNLogit (for interpretable coefficients / odds ratios, §6.2) and
an L2-regularized sklearn model (§5.3) used downstream for calibration and
backtesting. Same chronological time-series split as the MLR (§5.4).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss

from db import PROJECT_ROOT, load_config
from features import MODEL_FEATURE_COLUMNS, load_model_dataset
from train_mlr import TEST_FRACTION, standardize, time_series_split

RESULT_CATEGORIES = ["away", "draw", "home"]  # fixed order: away=0, draw=1, home=2


def fit_logistic(config: dict | None = None, feature_columns: list[str] | None = None, seasons: list[str] | None = None, artifact_prefix: str = "logistic"):
    config = config or load_config()
    feature_columns = feature_columns or MODEL_FEATURE_COLUMNS
    df = load_model_dataset(config, feature_columns=feature_columns, seasons=seasons)
    train, test = time_series_split(df, TEST_FRACTION)
    print(f"train={len(train)} test={len(test)} (chronological split, test = most recent {TEST_FRACTION:.0%})")

    Xtr_z, Xte_z, mu, sigma = standardize(train, test, feature_columns)

    y_cat_train = pd.Categorical(train["result"], categories=RESULT_CATEGORIES)
    y_cat_test = pd.Categorical(test["result"], categories=RESULT_CATEGORIES)
    ytr_code, yte_code = y_cat_train.codes, y_cat_test.codes

    # --- statsmodels MNLogit: interpretable coefficients / odds ratios ---
    Xtr_const = sm.add_constant(Xtr_z)
    mnlogit = sm.MNLogit(ytr_code, Xtr_const).fit(disp=False, method="lbfgs", maxiter=200)
    print(mnlogit.summary())

    # --- sklearn, L2-regularized: used downstream for calibration/backtest ---
    sk_model = LogisticRegression(
        multi_class="multinomial", penalty="l2", C=1.0, max_iter=2000, random_state=0
    )
    sk_model.fit(Xtr_z, ytr_code)

    proba_train = sk_model.predict_proba(Xtr_z)  # columns ordered by sk_model.classes_ == [0,1,2]
    proba_test = sk_model.predict_proba(Xte_z)

    pred_test = proba_test.argmax(axis=1)
    cm = confusion_matrix(yte_code, pred_test, labels=[0, 1, 2])
    print("\nConfusion matrix (rows=actual, cols=predicted; order away/draw/home):")
    print(cm)

    # one-vs-rest AUC per outcome
    aucs = {}
    for i, cat in enumerate(RESULT_CATEGORIES):
        aucs[cat] = roc_auc_score((yte_code == i).astype(int), proba_test[:, i])
    print("\nOne-vs-rest AUC:", aucs)

    accuracy = float((pred_test == yte_code).mean())
    baseline_accuracy = float(pd.Series(yte_code).value_counts(normalize=True).max())  # "always predict plurality class"
    y_onehot = np.eye(3)[yte_code]
    brier = float(((proba_test - y_onehot) ** 2).sum(axis=1).mean())
    logloss = float(log_loss(yte_code, proba_test, labels=[0, 1, 2]))
    error_stats = {
        "accuracy": accuracy, "baseline_accuracy": baseline_accuracy,
        "log_loss": logloss, "brier_score": brier, "n_test": len(test),
        **{f"auc_{c}": v for c, v in aucs.items()},
    }
    print(f"Holdout accuracy: {accuracy:.3f}  (plurality-class baseline: {baseline_accuracy:.3f})")
    print(f"Log loss: {logloss:.3f}  Brier: {brier:.3f}  (n={len(test)})")

    proba_train_df = pd.DataFrame(proba_train, columns=[f"p_{c}" for c in RESULT_CATEGORIES], index=train.index)
    proba_test_df = pd.DataFrame(proba_test, columns=[f"p_{c}" for c in RESULT_CATEGORIES], index=test.index)
    train_out = pd.concat([train.reset_index(drop=True), proba_train_df.reset_index(drop=True)], axis=1)
    test_out = pd.concat([test.reset_index(drop=True), proba_test_df.reset_index(drop=True)], axis=1)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / f"{artifact_prefix}_model.pkl", "wb") as f:
        pickle.dump(
            {
                "sklearn_model": sk_model,
                "mnlogit_model": mnlogit,
                "mu": mu,
                "sigma": sigma,
                "features": feature_columns,
                "categories": RESULT_CATEGORIES,
            },
            f,
        )
    train_out.to_csv(processed_dir / f"{artifact_prefix}_train_predictions.csv", index=False)
    test_out.to_csv(processed_dir / f"{artifact_prefix}_test_predictions.csv", index=False)
    np.savetxt(processed_dir / f"{artifact_prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    pd.Series(error_stats).to_csv(processed_dir / f"{artifact_prefix}_error_stats.csv")

    return sk_model, mnlogit, train_out, test_out, cm, aucs, error_stats


if __name__ == "__main__":
    fit_logistic()
