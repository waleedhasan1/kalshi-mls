"""Bankroll simulation: flat staking vs fractional Kelly (PLAN §7.5, §6.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from db import PROJECT_ROOT, load_config

STARTING_BANKROLL = 1000.0
FLAT_STAKE_FRACTION = 0.01  # 1% of starting bankroll per bet
KELLY_FRACTION = 0.25  # quarter-Kelly, standard practice to tame variance
MAX_KELLY_STAKE_FRACTION = 0.10  # cap any single Kelly stake at 10% of bankroll


def kelly_stake_fraction(model_prob: float, effective_price: float) -> float:
    """f* = (b*p - q) / b, for a bet that pays b:1 net odds on win."""
    b = 1.0 / effective_price - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f = (b * model_prob - q) / b
    return max(0.0, f)


def simulate(bets: pd.DataFrame) -> pd.DataFrame:
    bets = bets.sort_values("date").reset_index(drop=True)

    flat_bankroll = STARTING_BANKROLL
    kelly_bankroll = STARTING_BANKROLL
    flat_stake_amt = STARTING_BANKROLL * FLAT_STAKE_FRACTION

    rows = []
    for r in bets.itertuples():
        # flat
        flat_pnl = flat_stake_amt * (1.0 / r.effective_price - 1.0) if r.won else -flat_stake_amt
        flat_bankroll += flat_pnl

        # fractional Kelly, sized off current bankroll
        f = min(KELLY_FRACTION * kelly_stake_fraction(r.model_prob, r.effective_price), MAX_KELLY_STAKE_FRACTION)
        kelly_stake_amt = kelly_bankroll * f
        kelly_pnl = kelly_stake_amt * (1.0 / r.effective_price - 1.0) if r.won else -kelly_stake_amt
        kelly_bankroll += kelly_pnl

        rows.append(
            {
                "date": r.date, "game_id": r.game_id, "side": r.side, "won": r.won,
                "flat_stake": flat_stake_amt, "flat_pnl": flat_pnl, "flat_bankroll": flat_bankroll,
                "kelly_stake": kelly_stake_amt, "kelly_pnl": kelly_pnl, "kelly_bankroll": kelly_bankroll,
            }
        )

    sim = pd.DataFrame(rows)
    sim["flat_drawdown"] = sim["flat_bankroll"] - sim["flat_bankroll"].cummax()
    sim["kelly_drawdown"] = sim["kelly_bankroll"] - sim["kelly_bankroll"].cummax()
    return sim


def run(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    bets_path = processed_dir / "model_bets.csv"
    if not bets_path.exists() or pd.read_csv(bets_path).empty:
        print("No model bets to simulate (run evaluate.py first, and/or wait for more Kalshi history).")
        return pd.DataFrame()

    bets = pd.read_csv(bets_path, parse_dates=["date"])
    sim = simulate(bets)

    print(f"Flat staking:  final bankroll=${sim['flat_bankroll'].iloc[-1]:.2f}  max drawdown=${sim['flat_drawdown'].min():.2f}")
    print(f"Fractional Kelly: final bankroll=${sim['kelly_bankroll'].iloc[-1]:.2f}  max drawdown=${sim['kelly_drawdown'].min():.2f}")

    sim.to_csv(processed_dir / "bankroll_sim.csv", index=False)
    return sim


if __name__ == "__main__":
    run()
