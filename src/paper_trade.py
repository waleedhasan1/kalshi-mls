"""Paper trading: run the fitted model against *live* Kalshi markets for games
that haven't been played yet, log what it would have bet, and settle those
paper bets once the real result is known (PLAN §7.7 -- "paper-trade forward
before risking real money").

No real money moves. `bets.is_backtest=False` marks these as live paper trades,
distinct from evaluate.py's historical backtest (`is_backtest=True`).

Usage:
    python3 paper_trade.py place    # find upcoming games, log candidate + placed bets
    python3 paper_trade.py settle   # check pending paper bets against real results
"""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db import PROJECT_ROOT, Bet, Game, KalshiMarket, KalshiPrice, get_engine, get_session, load_config
from evaluate import SIDES, kalshi_fee
from features import MODEL_FEATURE_COLUMNS
from live_features import build_live_feature_row
from resolve_game import parse_event_date
from train_logistic import RESULT_CATEGORIES

import pickle


def _load_model(config: dict):
    processed_dir = PROJECT_ROOT / "data" / "processed"
    with open(processed_dir / "logistic_model.pkl", "rb") as f:
        return pickle.load(f)


def _upcoming_events(engine) -> dict[str, list[KalshiMarket]]:
    with get_session(engine) as session:
        markets = session.execute(
            select(KalshiMarket).where(
                KalshiMarket.status == "active",
                KalshiMarket.resolved_game_id.is_(None),
                KalshiMarket.resolved_team_id.isnot(None) | (KalshiMarket.yes_outcome == "tie"),
            )
        ).scalars().all()
    by_event: dict[str, list[KalshiMarket]] = {}
    for m in markets:
        by_event.setdefault(m.event_ticker, []).append(m)
    return by_event


def _latest_price(engine, ticker: str) -> float | None:
    with get_session(engine) as session:
        p = session.execute(
            select(KalshiPrice).where(KalshiPrice.ticker == ticker).order_by(KalshiPrice.ts.desc()).limit(1)
        ).scalars().first()
    if not p:
        return None
    if p.price_close_dollars is not None:
        return p.price_close_dollars
    if p.yes_bid_close_dollars is not None and p.yes_ask_close_dollars is not None:
        return (p.yes_bid_close_dollars + p.yes_ask_close_dollars) / 2
    return None


def _placeholder_game_id(event_ticker: str) -> str:
    return f"sched_{event_ticker}"


def place(config: dict | None = None, stake: float = 10.0) -> pd.DataFrame:
    config = config or load_config()
    engine = get_engine(config)
    fee_rate = config["betting"]["kalshi_fee_rate"]
    fav_min = config["betting"]["favorite_price_min_dollars"]

    artifacts = _load_model(config)
    sk_model, mu, sigma = artifacts["sklearn_model"], artifacts["mu"], artifacts["sigma"]

    events = _upcoming_events(engine)
    print(f"{len(events)} upcoming MLS events with an active, unresolved Kalshi market")

    rows = []
    now = dt.datetime.utcnow()

    with get_session(engine) as session:
        for event_ticker, markets in events.items():
            home_m = next((m for m in markets if m.yes_outcome == "home"), None)
            away_m = next((m for m in markets if m.yes_outcome == "away"), None)
            if not home_m or not away_m or not home_m.resolved_team_id or not away_m.resolved_team_id:
                continue
            game_date = parse_event_date(event_ticker)
            if not game_date:
                continue
            kickoff = dt.datetime.combine(game_date, dt.time(0, 0))
            if kickoff < now - dt.timedelta(days=1):
                continue  # stale/past event still marked active; skip

            feat = build_live_feature_row(home_m.resolved_team_id, away_m.resolved_team_id, kickoff, config)
            if any(feat[c] is None for c in MODEL_FEATURE_COLUMNS):
                print(f"  {event_ticker}: skipped, missing feature data for a new/unseen team")
                continue

            x = pd.DataFrame([{c: feat[c] for c in MODEL_FEATURE_COLUMNS}])
            x_z = (x - mu) / sigma
            proba = sk_model.predict_proba(x_z)[0]  # order: away, draw, home
            probs = dict(zip(RESULT_CATEGORIES, proba))

            placeholder_id = _placeholder_game_id(event_ticker)
            stmt = sqlite_insert(Game).values(
                game_id=placeholder_id, date=kickoff, season=str(game_date.year),
                home_team_id=home_m.resolved_team_id, away_team_id=away_m.resolved_team_id,
                home_score=None, away_score=None, status="scheduled",
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=["game_id"])
            session.execute(stmt)

            for side in SIDES:
                m = home_m if side == "home" else away_m if side == "away" else next(
                    (mm for mm in markets if mm.yes_outcome == "tie"), None
                )
                if not m:
                    continue
                price = _latest_price(engine, m.ticker)
                if price is None:
                    continue
                model_prob = probs["draw"] if side == "tie" else probs[side]
                effective_price = price + kalshi_fee(price, fee_rate)
                ev = model_prob * (1.0 / effective_price) - 1.0
                is_favorite = price >= fav_min
                will_bet = is_favorite and ev > 0

                rows.append(
                    {
                        "event_ticker": event_ticker, "game_id": placeholder_id, "ticker": m.ticker,
                        "kickoff": kickoff, "side": side, "price": price, "model_prob": model_prob,
                        "ev": ev, "is_favorite": is_favorite, "placed": will_bet,
                    }
                )

                if will_bet:
                    already = session.execute(
                        select(Bet).where(
                            Bet.kalshi_ticker == m.ticker, Bet.is_backtest == False, Bet.outcome.is_(None)  # noqa: E712
                        )
                    ).scalars().first()
                    if already:
                        continue
                    session.add(
                        Bet(
                            game_id=placeholder_id, kalshi_ticker=m.ticker, side=side, stake=stake,
                            price_paid=price, model_prob=model_prob, ev=ev, outcome=None, pnl=None,
                            is_backtest=False, placed_at=now,
                        )
                    )
        session.commit()

    candidates = pd.DataFrame(rows)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "paper_trade_candidates.csv"
    if not candidates.empty:
        prior = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
        candidates["logged_at"] = now
        pd.concat([prior, candidates], ignore_index=True).to_csv(out_path, index=False)

    n_placed = int(candidates["placed"].sum()) if not candidates.empty else 0
    print(f"{len(candidates)} candidate (event, side) evaluations, {n_placed} placed as paper bets")
    if n_placed:
        print(candidates[candidates["placed"]][["event_ticker", "side", "price", "model_prob", "ev"]].to_string(index=False))
    return candidates


def settle(config: dict | None = None) -> pd.DataFrame:
    """Check pending paper bets (outcome IS NULL) against real ASA results."""
    config = config or load_config()
    engine = get_engine(config)
    settled = []

    with get_session(engine) as session:
        pending = session.execute(
            select(Bet).where(Bet.is_backtest == False, Bet.outcome.is_(None))  # noqa: E712
        ).scalars().all()
        print(f"{len(pending)} pending paper bets")

        for bet in pending:
            placeholder = session.get(Game, bet.game_id)
            if not placeholder:
                continue
            real = session.execute(
                select(Game).where(
                    Game.home_team_id == placeholder.home_team_id,
                    Game.away_team_id == placeholder.away_team_id,
                    Game.date >= placeholder.date - dt.timedelta(days=2),
                    Game.date <= placeholder.date + dt.timedelta(days=2),
                    Game.status == "FullTime",
                    Game.game_id != placeholder.game_id,
                )
            ).scalars().first()
            if not real:
                continue

            won = (
                real.home_score == real.away_score if bet.side == "tie"
                else real.home_score > real.away_score if bet.side == "home"
                else real.home_score < real.away_score
            )
            # recompute effective price the same way it was computed at placement time
            fee_rate = config["betting"]["kalshi_fee_rate"]
            effective_price = bet.price_paid + kalshi_fee(bet.price_paid, fee_rate)
            pnl = bet.stake * (1.0 / effective_price - 1.0) if won else -bet.stake

            bet.outcome = "win" if won else "loss"
            bet.pnl = pnl
            placeholder.home_score, placeholder.away_score, placeholder.status = real.home_score, real.away_score, "FullTime"

            settled.append(
                {"game_id": bet.game_id, "side": bet.side, "outcome": bet.outcome, "pnl": pnl,
                 "real_game_id": real.game_id, "final_score": f"{real.home_score}-{real.away_score}"}
            )
        session.commit()

    df = pd.DataFrame(settled)
    if not df.empty:
        print(df.to_string(index=False))
        print(f"\nSettled {len(df)} paper bets: {int((df['outcome']=='win').sum())} won, "
              f"{int((df['outcome']=='loss').sum())} lost, total P&L ${df['pnl'].sum():+.2f}")
    else:
        print("Nothing to settle yet.")
    return df


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "place"
    if action == "place":
        place()
    elif action == "settle":
        settle()
    else:
        print("usage: python3 paper_trade.py [place|settle]")
