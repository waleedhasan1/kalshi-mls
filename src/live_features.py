""""Current form" features for a team as of right now, for games that haven't
been played yet (paper_trade.py).

`features.py`'s pipeline is built entirely around completed ASA games -- every
rolling stat is a `shift(1)` over a team's own game history, which only makes
sense once a game has a result to shift past. For an upcoming fixture there is
no such row yet, so this module recomputes the same formulas (same Elo
constants, same rolling window, same haversine distance) evaluated at "now" --
i.e. including each team's most recent completed game, which is exactly the
rolling value the shifted pipeline *would* produce for their next fixture.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
from itscalledsoccer.client import AmericanSoccerAnalysis
from sqlalchemy import select

from db import Game, Team, get_engine, get_session, load_config
from features import ELO_HOME_ADV, ELO_K, ELO_START


def current_elo_ratings(engine) -> dict[str, float]:
    """Same update rule as features.compute_elo, but returns only the final
    (as-of-now) rating per team, after every completed game."""
    with get_session(engine) as session:
        games = session.execute(
            select(Game.date, Game.home_team_id, Game.away_team_id, Game.home_score, Game.away_score)
            .where(Game.home_score.isnot(None))
            .order_by(Game.date)
        ).all()

    ratings: dict[str, float] = {}
    for date, home_id, away_id, home_score, away_score in games:
        h = ratings.get(home_id, ELO_START)
        a = ratings.get(away_id, ELO_START)
        expected_home = 1.0 / (1.0 + 10 ** (-((h + ELO_HOME_ADV) - a) / 400))
        actual_home = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
        delta = ELO_K * (actual_home - expected_home)
        ratings[home_id] = h + delta
        ratings[away_id] = a - delta
    return ratings


def current_rolling_team_xg(engine, asa: AmericanSoccerAnalysis, team_ids: set[str], window: int, season: str) -> dict[str, dict]:
    """Last `window` completed games' xG for/against/over-performance, per team."""
    tx = asa.get_team_xgoals(leagues="mls", season_name=season, split_by_games=True)
    with get_session(engine) as session:
        game_dates = dict(session.execute(select(Game.game_id, Game.date)).all())
    tx["date"] = tx["game_id"].map(game_dates)
    tx = tx.dropna(subset=["date"]).sort_values("date")

    out = {}
    for team_id in team_ids:
        sub = tx[tx["team_id"] == team_id].tail(window)
        if sub.empty:
            out[team_id] = {"xg_roll": None, "xga_roll": None, "xg_overperf_roll": None}
            continue
        out[team_id] = {
            "xg_roll": float(sub["xgoals_for"].mean()),
            "xga_roll": float(sub["xgoals_against"].mean()),
            "xg_overperf_roll": float(sub["goal_difference_minus_xgoal_difference"].mean()),
        }
    return out


def current_rolling_goals_added(engine, team_ids: set[str], window: int) -> dict[str, float | None]:
    query = """
        SELECT pms.team_id, g.date, SUM(pms.goals_added_above_avg) AS team_ga
        FROM player_match_stats pms JOIN games g ON g.game_id = pms.game_id
        WHERE pms.team_id IN ({placeholders})
        GROUP BY pms.team_id, pms.game_id
        ORDER BY g.date
    """.format(placeholders=",".join(f"'{t}'" for t in team_ids))
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["date"])
    out = {}
    for team_id in team_ids:
        sub = df[df["team_id"] == team_id].tail(window)
        out[team_id] = float(sub["team_ga"].mean()) if not sub.empty else None
    return out


def current_rest_days(engine, team_ids: set[str], as_of: dt.datetime) -> dict[str, float | None]:
    with get_session(engine) as session:
        games = session.execute(
            select(Game.date, Game.home_team_id, Game.away_team_id).where(Game.home_score.isnot(None))
        ).all()
    last_played: dict[str, dt.datetime] = {}
    for date, home_id, away_id in games:
        for t in (home_id, away_id):
            if t not in last_played or date > last_played[t]:
                last_played[t] = date
    return {
        t: (as_of - last_played[t]).total_seconds() / 86400 if t in last_played else None
        for t in team_ids
    }


def current_venue_advantage(engine, team_ids: set[str]) -> dict[str, float | None]:
    with get_session(engine) as session:
        games = session.execute(
            select(Game.home_team_id, Game.home_score, Game.away_score).where(Game.home_score.isnot(None))
        ).all()
    out = {}
    for team_id in team_ids:
        home_games = [(h, a) for hid, h, a in games if hid == team_id]
        out[team_id] = (sum(1 for h, a in home_games if h > a) / len(home_games)) if len(home_games) >= 3 else None
    return out


def current_matches_since_manager_change(engine, team_ids: set[str]) -> dict[str, int | None]:
    """Approximate: last known counter value from that team's most recent
    completed game, +1 (one more match has been played under that manager
    since). Manager-change data isn't available for an unplayed fixture."""
    query = """
        SELECT g.home_team_id AS team_id, g.date,
               f.home_matches_since_manager_change AS counter
        FROM features f JOIN games g ON g.game_id = f.game_id
        WHERE g.home_team_id IN ({placeholders})
        UNION ALL
        SELECT g.away_team_id AS team_id, g.date,
               f.away_matches_since_manager_change AS counter
        FROM features f JOIN games g ON g.game_id = f.game_id
        WHERE g.away_team_id IN ({placeholders})
    """.format(placeholders=",".join(f"'{t}'" for t in team_ids))
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["date"])
    out = {}
    for team_id in team_ids:
        sub = df[df["team_id"] == team_id].dropna(subset=["counter"]).sort_values("date")
        out[team_id] = int(sub["counter"].iloc[-1]) + 1 if not sub.empty else None
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return None
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_live_feature_row(
    home_team_id: str, away_team_id: str, kickoff: dt.datetime, config: dict | None = None
) -> dict:
    """One feature row, computed as-of-now, matching features.MODEL_FEATURE_COLUMNS."""
    config = config or load_config()
    engine = get_engine(config)
    asa = AmericanSoccerAnalysis()
    window = config["features"]["rolling_window_matches"]
    season = config["asa"]["player_stats_seasons"][-1]
    team_ids = {home_team_id, away_team_id}

    elo = current_elo_ratings(engine)
    xg = current_rolling_team_xg(engine, asa, team_ids, window, season)
    ga = current_rolling_goals_added(engine, team_ids, window)
    rest = current_rest_days(engine, team_ids, kickoff)
    venue = current_venue_advantage(engine, team_ids)
    manager = current_matches_since_manager_change(engine, team_ids)

    with get_session(engine) as session:
        teams = {t.team_id: t for t in session.execute(select(Team).where(Team.team_id.in_(team_ids))).scalars()}
    home_t, away_t = teams.get(home_team_id), teams.get(away_team_id)
    travel_km = _haversine_km(
        away_t.stadium_lat if away_t else None, away_t.stadium_lon if away_t else None,
        home_t.stadium_lat if home_t else None, home_t.stadium_lon if home_t else None,
    )
    tz_hours = (
        (home_t.stadium_lon - away_t.stadium_lon) / 15.0
        if home_t and away_t and home_t.stadium_lon is not None and away_t.stadium_lon is not None
        else None
    )

    home_rest, away_rest = rest.get(home_team_id), rest.get(away_team_id)

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "kickoff": kickoff,
        "home_elo": elo.get(home_team_id, ELO_START),
        "away_elo": elo.get(away_team_id, ELO_START),
        "home_xg_roll": xg[home_team_id]["xg_roll"],
        "away_xg_roll": xg[away_team_id]["xg_roll"],
        "home_xga_roll": xg[home_team_id]["xga_roll"],
        "away_xga_roll": xg[away_team_id]["xga_roll"],
        "home_xg_overperf_roll": xg[home_team_id]["xg_overperf_roll"],
        "away_xg_overperf_roll": xg[away_team_id]["xg_overperf_roll"],
        "home_goals_added_roll": ga.get(home_team_id),
        "away_goals_added_roll": ga.get(away_team_id),
        "rest_days_diff": (home_rest - away_rest) if home_rest is not None and away_rest is not None else None,
        "away_travel_km": travel_km,
        "away_timezone_change_hours": tz_hours,
        "home_venue_advantage": venue.get(home_team_id),
        "home_matches_since_manager_change": manager.get(home_team_id),
        "away_matches_since_manager_change": manager.get(away_team_id),
    }
