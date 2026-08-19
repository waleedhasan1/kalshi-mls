"""Multiple linear regression: predicted goal differential (PLAN §5.2, §10.7).

Interpretable secondary signal. Time-series split — train on past matches,
test forward, never randomly shuffled across time (§5.4). Standardizes
features so coefficients are comparable in the coefficient plot (§6.1).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from db import PROJECT_ROOT, load_config
from features import MODEL_FEATURE_COLUMNS, load_model_dataset

TEST_FRACTION = 0.2


def time_series_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mu, sigma = train[cols].mean(), train[cols].std()
    train_z = (train[cols] - mu) / sigma
    test_z = (test[cols] - mu) / sigma
    return train_z, test_z, mu, sigma


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    vif = pd.DataFrame(
        {
            "feature": Xc.columns,
            "vif": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
        }
    )
    return vif[vif["feature"] != "const"].sort_values("vif", ascending=False)


def fit_mlr(config: dict | None = None, feature_columns: list[str] | None = None, seasons: list[str] | None = None, artifact_prefix: str = "mlr"):
    config = config or load_config()
    feature_columns = feature_columns or MODEL_FEATURE_COLUMNS
    df = load_model_dataset(config, feature_columns=feature_columns, seasons=seasons)
    train, test = time_series_split(df)
    print(f"train={len(train)} test={len(test)} (chronological split, test = most recent {TEST_FRACTION:.0%})")

    Xtr_z, Xte_z, mu, sigma = standardize(train, test, feature_columns)
    ytr, yte = train["goal_diff"].astype(float), test["goal_diff"].astype(float)

    Xtr_const = sm.add_constant(Xtr_z)
    model = sm.OLS(ytr, Xtr_const).fit()
    print(model.summary())

    vif = compute_vif(Xtr_z)
    print("\nVIF (train features):")
    print(vif.to_string(index=False))

    Xte_const = sm.add_constant(Xte_z, has_constant="add")
    pred_test = model.predict(Xte_const)
    pred_train = model.predict(Xtr_const)

    resid_test = yte - pred_test
    rmse = float(np.sqrt((resid_test**2).mean()))
    mae = float(resid_test.abs().mean())
    medae = float(resid_test.abs().median())
    ss_res = float((resid_test**2).sum())
    ss_tot = float(((yte - ytr.mean())**2).sum())
    holdout_r2 = 1 - ss_res / ss_tot
    error_stats = {"rmse": rmse, "mae": mae, "median_ae": medae, "holdout_r2": holdout_r2, "n_test": len(test)}
    print(f"\nHoldout RMSE={rmse:.3f}  MAE={mae:.3f}  MedAE={medae:.3f}  R2={holdout_r2:.3f}  (n={len(test)})")

    train_out = train.assign(pred_goal_diff=pred_train.values, residual=(ytr - pred_train).values)
    test_out = test.assign(pred_goal_diff=pred_test.values, residual=resid_test.values)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / f"{artifact_prefix}_model.pkl", "wb") as f:
        pickle.dump({"model": model, "mu": mu, "sigma": sigma, "features": feature_columns}, f)
    train_out.to_csv(processed_dir / f"{artifact_prefix}_train_predictions.csv", index=False)
    test_out.to_csv(processed_dir / f"{artifact_prefix}_test_predictions.csv", index=False)
    vif.to_csv(processed_dir / f"{artifact_prefix}_vif.csv", index=False)
    pd.Series(error_stats).to_csv(processed_dir / f"{artifact_prefix}_error_stats.csv")

    return model, train_out, test_out, vif, error_stats


if __name__ == "__main__":
    fit_mlr()
