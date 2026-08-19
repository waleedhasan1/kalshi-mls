"""Kalshi market -> ASA game_id -> player pool resolver (PLAN §3.2.3-3.2.4).

The SQL-relational core of the project: given a Kalshi MLS event, find the
matching ASA game, then the pool of players relevant to each side.
"""

from __future__ import annotations

import datetime as dt
import re

from sqlalchemy import select

from db import Game, KalshiMarket, PlayerMatchStat, get_engine, get_session, load_config

EVENT_TICKER_RE = re.compile(r"^KXMLSGAME-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_event_date(event_ticker: str) -> dt.date | None:
    """'KXMLSGAME-26AUG30STLDAL' -> date(2026, 8, 30)."""
    m = EVENT_TICKER_RE.match(event_ticker)
    if not m:
        return None
    year = 2000 + int(m.group("yy"))
    month = _MONTHS.get(m.group("mon"))
    if not month:
        return None
    return dt.date(year, month, int(m.group("dd")))


def resolve_all_markets(config: dict | None = None, date_tolerance_days: int = 2) -> int:
    """Fill in kalshi_markets.resolved_game_id for every unresolved event.

    Matches on (home_team_id, away_team_id, date within tolerance) since the
    Kalshi event date is the scheduled kickoff date and can drift a day from
    ASA's recorded UTC kickoff for late-night matches.
    """
    config = config or load_config()
    engine = get_engine(config)
    resolved_count = 0

    with get_session(engine) as session:
        markets = session.scalars(
            select(KalshiMarket).where(KalshiMarket.resolved_game_id.is_(None))
        ).all()

        by_event: dict[str, list[KalshiMarket]] = {}
        for m in markets:
            by_event.setdefault(m.event_ticker, []).append(m)

        for event_ticker, event_markets in by_event.items():
            event_date = parse_event_date(event_ticker)
            if not event_date:
                continue

            home_ids = {m.resolved_team_id for m in event_markets if m.yes_outcome == "home" and m.resolved_team_id}
            away_ids = {m.resolved_team_id for m in event_markets if m.yes_outcome == "away" and m.resolved_team_id}
            if not home_ids or not away_ids:
                continue
            home_id, away_id = next(iter(home_ids)), next(iter(away_ids))

            lo = dt.datetime.combine(event_date - dt.timedelta(days=date_tolerance_days), dt.time.min)
            hi = dt.datetime.combine(event_date + dt.timedelta(days=date_tolerance_days), dt.time.max)
            game = session.scalars(
                select(Game).where(
                    Game.home_team_id == home_id,
                    Game.away_team_id == away_id,
                    Game.date >= lo,
                    Game.date <= hi,
                )
            ).first()
            if not game:
                continue

            for m in event_markets:
                m.resolved_game_id = game.game_id
            resolved_count += len(event_markets)

        session.commit()

    return resolved_count


def get_player_pool(game_id: str, team_id: str, config: dict | None = None) -> list[dict]:
    """Players who actually appeared for `team_id` in `game_id` (backtest use)."""
    config = config or load_config()
    engine = get_engine(config)
    with get_session(engine) as session:
        rows = session.scalars(
            select(PlayerMatchStat).where(
                PlayerMatchStat.game_id == game_id, PlayerMatchStat.team_id == team_id
            )
        ).all()
        return [
            {
                "player_id": r.player_id,
                "minutes": r.minutes,
                "xgoals": r.xgoals,
                "xassists": r.xassists,
                "goals_added_above_avg": r.goals_added_above_avg,
            }
            for r in rows
        ]


if __name__ == "__main__":
    n = resolve_all_markets()
    print(f"Resolved {n} markets to ASA game_ids")
