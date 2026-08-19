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


def fit_mlr(config: dict | None = None):
    config = config or load_config()
    df = load_model_dataset(config)
    train, test = time_series_split(df)
    print(f"train={len(train)} test={len(test)} (chronological split, test = most recent {TEST_FRACTION:.0%})")

    Xtr_z, Xte_z, mu, sigma = standardize(train, test, MODEL_FEATURE_COLUMNS)
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
    print(f"\nHoldout RMSE={rmse:.3f}  MAE={mae:.3f}")

    train_out = train.assign(pred_goal_diff=pred_train.values, residual=(ytr - pred_train).values)
    test_out = test.assign(pred_goal_diff=pred_test.values, residual=resid_test.values)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "mlr_model.pkl", "wb") as f:
        pickle.dump({"model": model, "mu": mu, "sigma": sigma, "features": MODEL_FEATURE_COLUMNS}, f)
    train_out.to_csv(processed_dir / "mlr_train_predictions.csv", index=False)
    test_out.to_csv(processed_dir / "mlr_test_predictions.csv", index=False)
    vif.to_csv(processed_dir / "mlr_vif.csv", index=False)

    return model, train_out, test_out, vif


if __name__ == "__main__":
    fit_mlr()
