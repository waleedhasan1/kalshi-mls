"""Backtest vs Kalshi closing prices, EV net of fees (PLAN §7.2-§7.6).

Every candidate bet here comes from `calibrate.py`'s final holdout set (the
untouched second half of the chronological test split) joined to real Kalshi
closing prices captured pre-kickoff (see the leakage fix in features.py).
Only ~66 of our 5762 ASA games currently have a resolved Kalshi market at
all -- Kalshi's MLS per-game markets only have a few weeks of history at
build time -- so this is a small, honest backtest, not a large one. See the
§7.7 reality check: treat any edge here as a hypothesis to keep testing
forward, not a proven result.
"""

from __future__ import annotations

import pandas as pd

from db import PROJECT_ROOT, Bet, get_engine, get_session, load_config

SIDES = ["home", "away", "tie"]


def kalshi_fee(price: float, fee_rate: float) -> float:
    """Simplified approximation of Kalshi's published trading-fee formula
    (fee ~= fee_rate * price * (1 - price) per contract). Real fees vary by
    market/maker-taker side; this is a conservative, documented estimate,
    not Kalshi's exact schedule.
    """
    return fee_rate * price * (1 - price)


def build_candidate_bets(holdout: pd.DataFrame, config: dict) -> pd.DataFrame:
    fee_rate = config["betting"]["kalshi_fee_rate"]
    fav_min = config["betting"]["favorite_price_min_dollars"]

    rows = []
    for r in holdout.itertuples():
        for side in SIDES:
            price = getattr(r, f"kalshi_{side}_implied_prob", None)
            model_prob = getattr(r, f"p_cal_{'draw' if side == 'tie' else side}", None)
            if price is None or pd.isna(price) or model_prob is None or pd.isna(model_prob):
                continue
            effective_price = price + kalshi_fee(price, fee_rate)
            ev = model_prob * (1.0 / effective_price) - 1.0
            won = (r.result == "draw" if side == "tie" else r.result == side)
            rows.append(
                {
                    "game_id": r.game_id,
                    "date": r.date,
                    "side": side,
                    "price": price,
                    "effective_price": effective_price,
                    "model_prob": model_prob,
                    "ev": ev,
                    "is_favorite": price >= fav_min,
                    "won": bool(won),
                }
            )
    return pd.DataFrame(rows)


def backtest(config: dict | None = None):
    config = config or load_config()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    holdout = pd.read_csv(processed_dir / "calibrated_holdout_predictions.csv", parse_dates=["date"])

    candidates = build_candidate_bets(holdout, config)
    candidates.to_csv(processed_dir / "candidate_bets.csv", index=False)
    print(f"{len(candidates)} candidate (game, side) pairs with a real Kalshi closing price")

    favorites = candidates[candidates["is_favorite"]]
    print(f"{len(favorites)} are on the favorite side (price >= {config['betting']['favorite_price_min_dollars']})")

    model_bets = favorites[favorites["ev"] > 0].copy()
    print(f"{len(model_bets)} have positive model EV net of fees -> these are the bets placed")

    stake = 10.0
    model_bets["stake"] = stake
    model_bets["pnl"] = model_bets.apply(
        lambda r: stake * (1.0 / r["effective_price"] - 1.0) if r["won"] else -stake, axis=1
    )

    baseline = favorites.copy()  # §7.6: buy every favorite, no model filter
    baseline["stake"] = stake
    baseline["pnl"] = baseline.apply(
        lambda r: stake * (1.0 / r["effective_price"] - 1.0) if r["won"] else -stake, axis=1
    )

    def summarize(df: pd.DataFrame, label: str) -> dict:
        if df.empty:
            print(f"{label}: no bets")
            return {"label": label, "n_bets": 0}
        n, hit_rate = len(df), df["won"].mean()
        total_staked, total_pnl = df["stake"].sum(), df["pnl"].sum()
        roi = total_pnl / total_staked
        print(
            f"{label}: n={n} hit_rate={hit_rate:.1%} total_staked=${total_staked:.0f} "
            f"total_pnl=${total_pnl:+.2f} ROI={roi:+.1%}"
        )
        return {
            "label": label, "n_bets": n, "hit_rate": hit_rate,
            "total_staked": total_staked, "total_pnl": total_pnl, "roi": roi,
        }

    model_summary = summarize(model_bets, "Model (favorite + positive EV)")
    baseline_summary = summarize(baseline, "Baseline (every favorite, no model filter)")

    if not model_bets.empty:
        model_bets.to_csv(processed_dir / "model_bets.csv", index=False)
    if not baseline.empty:
        baseline.to_csv(processed_dir / "baseline_bets.csv", index=False)

    # persist to the `bets` table
    engine = get_engine(config)
    with get_session(engine) as session:
        for r in model_bets.itertuples():
            session.add(
                Bet(
                    game_id=r.game_id,
                    kalshi_ticker=None,
                    side=r.side,
                    stake=r.stake,
                    price_paid=r.price,
                    model_prob=r.model_prob,
                    ev=r.ev,
                    outcome="win" if r.won else "loss",
                    pnl=r.pnl,
                    is_backtest=True,
                    placed_at=r.date,
                )
            )
        session.commit()

    pd.DataFrame([model_summary, baseline_summary]).to_csv(processed_dir / "backtest_summary.csv", index=False)
    return candidates, model_bets, baseline, model_summary, baseline_summary


if __name__ == "__main__":
    backtest()
