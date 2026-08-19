"""Same two models, refit on the extended 1996-2026 history (PLAN §5.4 taken
further): pulls in supplementary MLS results from footballcsv
(ingest_external_mls.py) to nearly double the dataset (5,762 -> 9,367 games)
and re-evaluates on a much larger chronological holdout.

The extra 1996-2012 games have no ASA advanced stats (no xG, no goals-added,
no weather), so this uses `features.LONG_HISTORY_FEATURE_COLUMNS` -- Elo,
rest-days, travel, and venue-advantage, all of which only need date/teams/
score and are available across the full history. This is a genuinely
different, larger, and more independent test of the same modeling approach,
not just a bigger number for its own sake -- see the comparison printed at
the end against the original 2023-2026, full-feature models.
"""

from __future__ import annotations

import pandas as pd

from db import PROJECT_ROOT, load_config
from features import LONG_HISTORY_FEATURE_COLUMNS
from train_logistic import fit_logistic
from train_mlr import fit_mlr


def run(config: dict | None = None):
    config = config or load_config()
    processed_dir = PROJECT_ROOT / "data" / "processed"

    print("=" * 70)
    print("MLR — goal differential, full 1996-2026 history, reduced feature set")
    print("=" * 70)
    _, _, _, _, mlr_error = fit_mlr(
        config, feature_columns=LONG_HISTORY_FEATURE_COLUMNS, seasons=None, artifact_prefix="mlr_longhist"
    )

    print()
    print("=" * 70)
    print("Logistic — home/draw/away, full 1996-2026 history, reduced feature set")
    print("=" * 70)
    _, _, _, _, _, _, logit_error = fit_logistic(
        config, feature_columns=LONG_HISTORY_FEATURE_COLUMNS, seasons=None, artifact_prefix="logistic_longhist"
    )

    # side-by-side vs. the original 2023-2026, full-feature models, if they've been run
    print()
    print("=" * 70)
    print("Comparison: 2023-2026 (full features) vs. 1996-2026 (long-history features)")
    print("=" * 70)
    try:
        mlr_orig = pd.read_csv(processed_dir / "mlr_error_stats.csv", index_col=0).squeeze("columns")
        logit_orig = pd.read_csv(processed_dir / "logistic_error_stats.csv", index_col=0).squeeze("columns")
        comparison = pd.DataFrame(
            {
                "2023-2026 (full features)": {
                    "mlr_rmse": mlr_orig["rmse"], "mlr_mae": mlr_orig["mae"], "mlr_holdout_r2": mlr_orig["holdout_r2"],
                    "logit_accuracy": logit_orig["accuracy"], "logit_baseline_accuracy": logit_orig["baseline_accuracy"],
                    "logit_log_loss": logit_orig["log_loss"], "logit_brier": logit_orig["brier_score"],
                    "n_test": mlr_orig["n_test"],
                },
                "1996-2026 (long-history features)": {
                    "mlr_rmse": mlr_error["rmse"], "mlr_mae": mlr_error["mae"], "mlr_holdout_r2": mlr_error["holdout_r2"],
                    "logit_accuracy": logit_error["accuracy"], "logit_baseline_accuracy": logit_error["baseline_accuracy"],
                    "logit_log_loss": logit_error["log_loss"], "logit_brier": logit_error["brier_score"],
                    "n_test": mlr_error["n_test"],
                },
            }
        )
        print(comparison.to_string())
        comparison.to_csv(processed_dir / "long_history_comparison.csv")
    except FileNotFoundError:
        print("(run train_mlr.py / train_logistic.py first for a side-by-side comparison)")

    return mlr_error, logit_error


if __name__ == "__main__":
    run()
