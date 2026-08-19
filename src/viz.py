"""All plotting helpers (PLAN §6). Every figure is saved to reports/figures/."""

from __future__ import annotations

import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc

from db import PROJECT_ROOT, load_config

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ============================================================ §6.1 MLR ====


def plot_mlr_diagnostics():
    with open(PROCESSED / "mlr_model.pkl", "rb") as f:
        art = pickle.load(f)
    model = art["model"]
    fitted = model.fittedvalues
    resid = model.resid
    influence = model.get_influence()
    student_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.4, s=14)
    ax.axhline(0, color="crimson", lw=1)
    ax.set_xlabel("Fitted goal differential")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Fitted")

    ax = axes[0, 1]
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q")

    ax = axes[1, 0]
    ax.scatter(fitted, np.sqrt(np.abs(student_resid)), alpha=0.4, s=14)
    ax.set_xlabel("Fitted goal differential")
    ax.set_ylabel(r"$\sqrt{|Standardized\ residual|}$")
    ax.set_title("Scale-Location")

    ax = axes[1, 1]
    ax.scatter(leverage, student_resid, s=np.clip(cooks_d * 800, 5, 300), alpha=0.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentized residual")
    ax.set_title("Residuals vs Leverage (point size = Cook's D)")

    fig.suptitle("MLR (goal differential) — regression diagnostics", y=1.01, fontsize=13)
    fig.tight_layout()
    _save(fig, "mlr_diagnostics_4panel")


def plot_mlr_coefficients():
    with open(PROCESSED / "mlr_model.pkl", "rb") as f:
        art = pickle.load(f)
    model = art["model"]
    params = model.params.drop("const")
    ci = model.conf_int().drop("const")
    order = params.abs().sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    ax.errorbar(
        params[order], y,
        xerr=[params[order] - ci.loc[order, 0], ci.loc[order, 1] - params[order]],
        fmt="o", color="steelblue", ecolor="steelblue", capsize=3,
    )
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Standardized coefficient (goal differential per 1 SD)")
    ax.set_title("MLR coefficients with 95% CI")
    fig.tight_layout()
    _save(fig, "mlr_coefficient_plot")


def plot_mlr_pred_vs_actual(prefix: str = "mlr", title_suffix: str = ""):
    test = pd.read_csv(PROCESSED / f"{prefix}_test_predictions.csv")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(test["goal_diff"], test["pred_goal_diff"], alpha=0.45, s=18)
    lims = [min(test["goal_diff"].min(), test["pred_goal_diff"].min()) - 0.5,
            max(test["goal_diff"].max(), test["pred_goal_diff"].max()) + 0.5]
    ax.plot(lims, lims, color="crimson", lw=1, label="perfect prediction")
    ax.set_xlabel(f"Actual goal differential (holdout, n={len(test)})")
    ax.set_ylabel("Predicted goal differential")
    ax.set_title(f"MLR: predicted vs actual (holdout){title_suffix}")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"{prefix}_pred_vs_actual")


def plot_mlr_corr_vif():
    with open(PROCESSED / "mlr_model.pkl", "rb") as f:
        art = pickle.load(f)
    features = art["features"]
    train = pd.read_csv(PROCESSED / "mlr_train_predictions.csv")
    vif = pd.read_csv(PROCESSED / "mlr_vif.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    corr = train[features].corr()
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=axes[0], square=True,
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Feature correlation heatmap")
    axes[0].tick_params(axis="x", rotation=90)

    vif_sorted = vif.sort_values("vif")
    axes[1].barh(vif_sorted["feature"], vif_sorted["vif"], color="steelblue")
    axes[1].axvline(5, color="crimson", lw=1, ls="--", label="common VIF=5 concern threshold")
    axes[1].set_xlabel("VIF")
    axes[1].set_title("Multicollinearity (VIF)")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "mlr_correlation_vif")


# ==================================================== §6.2 logistic ====


def plot_logistic_coefficients():
    with open(PROCESSED / "logistic_model.pkl", "rb") as f:
        art = pickle.load(f)
    mnlogit = art["mnlogit_model"]
    # params column 1 == the "y=2" (home outcome vs away baseline) equation;
    # statsmodels labels params columns 0-indexed but conf_int()'s MultiIndex
    # by the actual endog code (1=draw, 2=home) -- these are NOT the same axis.
    params = mnlogit.params[1].drop("const")
    ci = mnlogit.conf_int().loc["2"].drop("const")
    odds = np.exp(params)
    odds_ci_low, odds_ci_high = np.exp(ci["lower"]), np.exp(ci["upper"])
    order = params.sort_values().index

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    ax.errorbar(
        odds[order], y,
        xerr=[odds[order] - odds_ci_low[order], odds_ci_high[order] - odds[order]],
        fmt="o", color="darkorange", ecolor="darkorange", capsize=3,
    )
    ax.axvline(1, color="grey", lw=1, label="odds ratio = 1 (no effect)")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Odds ratio (home win vs away win), per 1 SD")
    ax.set_title("Logistic regression: odds ratios with 95% CI")
    ax.legend()
    fig.tight_layout()
    _save(fig, "logistic_odds_ratio_plot")


def plot_roc_curves():
    test = pd.read_csv(PROCESSED / "logistic_test_predictions.csv")
    categories = ["away", "draw", "home"]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for i, cat in enumerate(categories):
        y_true = (pd.Categorical(test["result"], categories=categories).codes == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true, test[f"p_{cat}"])
        ax.plot(fpr, tpr, label=f"{cat} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (holdout)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "logistic_roc_curves")


def plot_confusion_matrix(prefix: str = "logistic", title_suffix: str = ""):
    cm = np.loadtxt(PROCESSED / f"{prefix}_confusion_matrix.csv", delimiter=",")
    labels = ["away", "draw", "home"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (holdout, n={int(cm.sum())}){title_suffix}")
    fig.tight_layout()
    _save(fig, f"{prefix}_confusion_matrix")


def plot_long_history_comparison():
    path = PROCESSED / "long_history_comparison.csv"
    if not path.exists():
        print("skip long-history comparison: run train_long_history.py first")
        return
    comp = pd.read_csv(path, index_col=0)
    cols = comp.columns.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    vals = [comp.loc["logit_accuracy", c] for c in cols]
    base = [comp.loc["logit_baseline_accuracy", c] for c in cols]
    x = np.arange(len(cols))
    ax.bar(x - 0.18, vals, width=0.36, label="model accuracy", color="steelblue")
    ax.bar(x + 0.18, base, width=0.36, label="plurality-class baseline", color="lightgrey")
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Holdout accuracy")
    ax.set_title("Outcome accuracy vs. baseline")
    ax.legend(fontsize=8)

    ax = axes[1]
    rmse = [comp.loc["mlr_rmse", c] for c in cols]
    mae = [comp.loc["mlr_mae", c] for c in cols]
    ax.bar(x - 0.18, rmse, width=0.36, label="RMSE", color="darkorange")
    ax.bar(x + 0.18, mae, width=0.36, label="MAE", color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Goals")
    ax.set_title("Goal-differential error")
    ax.legend(fontsize=8)

    ax = axes[2]
    n = [comp.loc["n_test", c] for c in cols]
    ax.bar(x, n, color="slateblue")
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Holdout games")
    ax.set_title("Test-set size")

    fig.suptitle("2023-2026 full-feature model vs. 1996-2026 long-history model", y=1.03)
    fig.tight_layout()
    _save(fig, "long_history_comparison")


def plot_reliability_curve():
    raw = pd.read_csv(PROCESSED / "reliability_curve_raw.csv")
    cal = pd.read_csv(PROCESSED / "reliability_curve_calibrated.csv")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
    ax.plot(raw["mean_predicted"], raw["observed_freq"], "o-", label="raw model", color="steelblue")
    ax.plot(cal["mean_predicted"], cal["observed_freq"], "o-", label="calibrated (selected method)", color="darkorange")
    ax.set_xlabel("Mean predicted probability (bucket)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability / calibration curve (holdout, pooled across outcomes)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "reliability_curve")


# ================================================= §6.3 edge vs market ====


def _price_bucket_curve(df: pd.DataFrame, price_col: str, won_col: str, n_bins: int = 5) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    d = df.copy()
    d["bucket"] = pd.cut(d[price_col], bins, include_lowest=True)
    out = d.groupby("bucket", observed=True).agg(
        mean_price=(price_col, "mean"), win_rate=(won_col, "mean"), n=(won_col, "size")
    ).reset_index()
    return out


def plot_calibration_overlay():
    candidates = pd.read_csv(PROCESSED / "candidate_bets.csv")
    if candidates.empty:
        print("skip calibration overlay: no candidate bets with real Kalshi prices")
        return
    model_curve = _price_bucket_curve(candidates, "model_prob", "won", n_bins=5)
    kalshi_curve = _price_bucket_curve(candidates, "price", "won", n_bins=5)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
    ax.plot(model_curve["mean_price"], model_curve["win_rate"], "o-", label="model", color="steelblue")
    ax.plot(kalshi_curve["mean_price"], kalshi_curve["win_rate"], "o-", label="Kalshi price", color="darkorange")
    ax.set_xlabel("Mean predicted / implied probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title(f"Model vs Kalshi calibration overlay (n={len(candidates)} contract-outcomes)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "calibration_overlay_model_vs_kalshi")


def plot_favorite_longshot_bias():
    candidates = pd.read_csv(PROCESSED / "candidate_bets.csv")
    if candidates.empty:
        print("skip favorite-longshot bias chart: no candidate bets")
        return
    curve = _price_bucket_curve(candidates, "price", "won", n_bins=5)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="fair pricing")
    ax.plot(curve["mean_price"], curve["win_rate"], "o-", color="seagreen")
    for _, row in curve.iterrows():
        ax.annotate(f"n={row['n']:.0f}", (row["mean_price"], row["win_rate"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Kalshi contract price bucket (implied probability)")
    ax.set_ylabel("Realized win rate")
    ax.set_title(f"Favorite-longshot bias, our data (n={len(candidates)} contract-outcomes)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "favorite_longshot_bias")


def plot_ev_by_price_bucket():
    candidates = pd.read_csv(PROCESSED / "candidate_bets.csv")
    if candidates.empty:
        print("skip EV-by-bucket chart: no candidate bets")
        return
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    candidates["bucket"] = pd.cut(candidates["price"], bins, include_lowest=True)
    ev_by_bucket = candidates.groupby("bucket", observed=True)["ev"].mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["crimson" if v < 0 else "seagreen" for v in ev_by_bucket.values]
    ax.bar([str(b) for b in ev_by_bucket.index], ev_by_bucket.values, color=colors)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("Price bucket")
    ax.set_ylabel("Mean EV net of fees")
    ax.set_title("Expected value by price bucket")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, "ev_by_price_bucket")


# ==================================================== §6.4 backtest ====


def plot_bankroll_curve():
    path = PROCESSED / "bankroll_sim.csv"
    if not path.exists():
        print("skip bankroll curve: no bet_sim.py output")
        return
    sim = pd.read_csv(path)
    if sim.empty:
        print("skip bankroll curve: empty simulation")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(sim))
    ax.plot(x, sim["flat_bankroll"], "o-", label="flat staking", color="steelblue")
    ax.plot(x, sim["kelly_bankroll"], "o-", label="quarter-Kelly", color="darkorange")
    ax.axhline(sim["flat_bankroll"].iloc[0], color="grey", lw=1, ls="--", label="starting bankroll")
    ax.set_xlabel("Bet number (chronological)")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title(f"Cumulative bankroll — flat vs fractional Kelly (n={len(sim)} bets)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "bankroll_curve")


def plot_drawdown():
    path = PROCESSED / "bankroll_sim.csv"
    if not path.exists():
        return
    sim = pd.read_csv(path)
    if sim.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sim))
    ax.fill_between(x, sim["flat_drawdown"], 0, alpha=0.4, label="flat staking", color="steelblue")
    ax.fill_between(x, sim["kelly_drawdown"], 0, alpha=0.4, label="quarter-Kelly", color="darkorange")
    ax.set_xlabel("Bet number (chronological)")
    ax.set_ylabel("Drawdown ($, running peak to trough)")
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    _save(fig, "drawdown")


def plot_bet_distribution():
    candidates = pd.read_csv(PROCESSED / "candidate_bets.csv")
    if candidates.empty:
        print("skip bet distribution: no candidate bets")
        return
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    candidates["price_bucket"] = pd.cut(candidates["price"], bins, include_lowest=True)
    candidates["ev_sign"] = np.where(candidates["ev"] > 0, "positive EV", "negative EV")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts = candidates.groupby(["price_bucket", "ev_sign"], observed=True).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=ax, color=["crimson", "seagreen"])
    ax.set_xlabel("Price bucket")
    ax.set_ylabel("Number of candidate (game, side) pairs")
    ax.set_title("Bet distribution by price bucket and EV sign")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, "bet_distribution")


# ============================================================== main ====


def main():
    plot_mlr_diagnostics()
    plot_mlr_coefficients()
    plot_mlr_pred_vs_actual()
    plot_mlr_corr_vif()

    plot_logistic_coefficients()
    plot_roc_curves()
    plot_confusion_matrix()
    plot_reliability_curve()

    plot_calibration_overlay()
    plot_favorite_longshot_bias()
    plot_ev_by_price_bucket()

    plot_bankroll_curve()
    plot_drawdown()
    plot_bet_distribution()

    plot_mlr_pred_vs_actual(prefix="mlr_longhist", title_suffix=" — 1996-2026, long-history features")
    plot_confusion_matrix(prefix="logistic_longhist", title_suffix=" — 1996-2026, long-history features")
    plot_long_history_comparison()


if __name__ == "__main__":
    main()
