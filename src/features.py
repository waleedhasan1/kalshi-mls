"""Feature engineering (PLAN §4, §10.6). Pre-match features only — no leakage.

Every rolling/expanding statistic is computed chronologically per team and
shifted by one game before being attached to a fixture, so a game's feature
row only ever reflects information available strictly before its kickoff.
"""

from __future__ import annotations

import datetime as dt
import math

import httpx
import numpy as np
import pandas as pd
from itscalledsoccer.client import AmericanSoccerAnalysis
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db import (
    Feature,
    Game,
    Injury,
    KalshiMarket,
    KalshiPrice,
    Team,
    get_engine,
    get_session,
    init_db,
    load_config,
)

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0


# ---------------------------------------------------------------- games ----


def load_games(engine) -> pd.DataFrame:
    with get_session(engine) as session:
        rows = session.execute(select(Game)).scalars().all()
        df = pd.DataFrame(
            [
                {
                    "game_id": g.game_id,
                    "date": g.date,
                    "season": g.season,
                    "home_team_id": g.home_team_id,
                    "away_team_id": g.away_team_id,
                    "home_score": g.home_score,
                    "away_score": g.away_score,
                    "status": g.status,
                    "stadium_id": g.stadium_id,
                    "home_manager_id": g.home_manager_id,
                    "away_manager_id": g.away_manager_id,
                }
                for g in rows
            ]
        )
    return df.sort_values("date").reset_index(drop=True)


def load_teams(engine) -> pd.DataFrame:
    with get_session(engine) as session:
        rows = session.execute(select(Team)).scalars().all()
        return pd.DataFrame(
            [
                {
                    "team_id": t.team_id,
                    "team_name": t.team_name,
                    "stadium_lat": t.stadium_lat,
                    "stadium_lon": t.stadium_lon,
                }
                for t in rows
            ]
        )


# ------------------------------------------------------------------ elo ----


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Own Elo implementation (§4.1.3): updated match-by-match, home-adjusted.

    Returns one row per game with each side's PRE-match rating (i.e. before
    this game's result updates it) — that's the leak-safe value to feature.
    """
    ratings: dict[str, float] = {}
    home_elo, away_elo = [], []

    for row in games.itertuples():
        h = ratings.get(row.home_team_id, ELO_START)
        a = ratings.get(row.away_team_id, ELO_START)
        home_elo.append(h)
        away_elo.append(a)

        if row.home_score is None or row.away_score is None:
            continue

        expected_home = 1.0 / (1.0 + 10 ** (-((h + ELO_HOME_ADV) - a) / 400))
        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = ELO_K * (actual_home - expected_home)
        ratings[row.home_team_id] = h + delta
        ratings[row.away_team_id] = a - delta

    games = games.copy()
    games["home_elo"] = home_elo
    games["away_elo"] = away_elo
    return games


# -------------------------------------------------------- rolling xG/GA ----


def _pull_team_xgoals_all_seasons(asa: AmericanSoccerAnalysis, seasons: list[str]) -> pd.DataFrame:
    frames = [asa.get_team_xgoals(leagues="mls", season_name=s, split_by_games=True) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def compute_rolling_team_xg(games: pd.DataFrame, asa: AmericanSoccerAnalysis, window: int) -> pd.DataFrame:
    seasons = sorted(games["season"].dropna().unique().tolist())
    tx = _pull_team_xgoals_all_seasons(asa, seasons)
    tx = tx.merge(games[["game_id", "date"]], on="game_id", how="inner")
    tx = tx.sort_values(["team_id", "date"])

    grp = tx.groupby("team_id")
    for col, out in [
        ("xgoals_for", "xg_roll"),
        ("xgoals_against", "xga_roll"),
        ("goal_difference_minus_xgoal_difference", "xg_overperf_roll"),
    ]:
        tx[out] = grp[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    home = tx.rename(columns={"team_id": "home_team_id", "xg_roll": "home_xg_roll", "xga_roll": "home_xga_roll",
                               "xg_overperf_roll": "home_xg_overperf_roll"})
    away = tx.rename(columns={"team_id": "away_team_id", "xg_roll": "away_xg_roll", "xga_roll": "away_xga_roll",
                               "xg_overperf_roll": "away_xg_overperf_roll"})

    games = games.merge(
        home[["game_id", "home_team_id", "home_xg_roll", "home_xga_roll", "home_xg_overperf_roll"]],
        on=["game_id", "home_team_id"],
        how="left",
    )
    games = games.merge(
        away[["game_id", "away_team_id", "away_xg_roll", "away_xga_roll", "away_xg_overperf_roll"]],
        on=["game_id", "away_team_id"],
        how="left",
    )
    return games


def compute_rolling_goals_added(games: pd.DataFrame, engine, window: int) -> pd.DataFrame:
    """From player_match_stats (populated for asa.player_stats_seasons only)."""
    query = """
        SELECT team_id, game_id, SUM(goals_added_above_avg) AS team_goals_added
        FROM player_match_stats
        GROUP BY team_id, game_id
    """
    with engine.connect() as conn:
        team_ga = pd.read_sql(query, conn)

    team_ga = team_ga.merge(games[["game_id", "date"]], on="game_id", how="inner").sort_values(["team_id", "date"])
    team_ga["goals_added_roll"] = team_ga.groupby("team_id")["team_goals_added"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = team_ga.rename(columns={"team_id": "home_team_id", "goals_added_roll": "home_goals_added_roll"})
    away = team_ga.rename(columns={"team_id": "away_team_id", "goals_added_roll": "away_goals_added_roll"})
    games = games.merge(home[["game_id", "home_team_id", "home_goals_added_roll"]], on=["game_id", "home_team_id"], how="left")
    games = games.merge(away[["game_id", "away_team_id", "away_goals_added_roll"]], on=["game_id", "away_team_id"], how="left")
    return games


# ---------------------------------------------------------- situational ----


def compute_rest_and_manager(games: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat(
        [
            games[["game_id", "date", "home_team_id", "home_manager_id"]].rename(
                columns={"home_team_id": "team_id", "home_manager_id": "manager_id"}
            ),
            games[["game_id", "date", "away_team_id", "away_manager_id"]].rename(
                columns={"away_team_id": "team_id", "away_manager_id": "manager_id"}
            ),
        ]
    ).sort_values(["team_id", "date"])

    long["prev_date"] = long.groupby("team_id")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.total_seconds() / 86400

    def matches_since_change(sub: pd.DataFrame) -> pd.Series:
        sub = sub.sort_values("date")
        prev_manager = sub["manager_id"].shift(1)
        changed = (sub["manager_id"] != prev_manager) & prev_manager.notna()
        counter = changed.groupby((changed).cumsum()).cumcount()
        counter[prev_manager.isna()] = np.nan  # no manager history yet
        return counter

    long["matches_since_manager_change"] = (
        long.groupby("team_id", group_keys=False)[["date", "manager_id"]].apply(matches_since_change)
    )

    home = long.rename(
        columns={
            "team_id": "home_team_id",
            "rest_days": "home_rest_days",
            "matches_since_manager_change": "home_matches_since_manager_change",
        }
    )
    away = long.rename(
        columns={
            "team_id": "away_team_id",
            "rest_days": "away_rest_days",
            "matches_since_manager_change": "away_matches_since_manager_change",
        }
    )
    games = games.merge(home[["game_id", "home_team_id", "home_rest_days", "home_matches_since_manager_change"]], on=["game_id", "home_team_id"], how="left")
    games = games.merge(away[["game_id", "away_team_id", "away_rest_days", "away_matches_since_manager_change"]], on=["game_id", "away_team_id"], how="left")
    games["rest_days_diff"] = games["home_rest_days"] - games["away_rest_days"]
    return games


# ------------------------------------------------------ location & venue ----


def _haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return None
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_travel(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    t = teams.set_index("team_id")
    home_lat = games["home_team_id"].map(t["stadium_lat"])
    home_lon = games["home_team_id"].map(t["stadium_lon"])
    away_lat = games["away_team_id"].map(t["stadium_lat"])
    away_lon = games["away_team_id"].map(t["stadium_lon"])

    games = games.copy()
    games["away_travel_km"] = [
        _haversine_km(a, b, c, d) for a, b, c, d in zip(away_lat, away_lon, home_lat, home_lon)
    ]
    # rough timezone-change proxy: 1 hour per 15 degrees of longitude crossed
    games["away_timezone_change_hours"] = (home_lon - away_lon) / 15.0
    return games


def compute_venue_advantage(games: pd.DataFrame) -> pd.DataFrame:
    """Expanding, pre-match home win rate for the home side at its own venue."""
    g = games.sort_values("date").copy()
    g["_home_win"] = np.where(
        g["home_score"].notna() & g["away_score"].notna(), (g["home_score"] > g["away_score"]).astype(float), np.nan
    )
    g["home_venue_advantage"] = g.groupby("home_team_id")["_home_win"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    return games.merge(g[["game_id", "home_venue_advantage"]], on="game_id", how="left")


# --------------------------------------------------------------- weather ----


def fetch_weather(games: pd.DataFrame, teams: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Daily weather at the home stadium, from Open-Meteo (public, no key).

    Scoped to `asa.player_stats_seasons` to bound the number of stadium-range
    API calls to the games actually used for modeling.
    """
    seasons = config["asa"]["player_stats_seasons"]
    scoped = games[games["season"].isin(seasons)]
    t = teams.set_index("team_id")

    weather_rows = []
    for team_id, sub in scoped.groupby("home_team_id"):
        lat, lon = t.loc[team_id, "stadium_lat"], t.loc[team_id, "stadium_lon"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        start, end = sub["date"].min().date(), min(sub["date"].max().date(), dt.date.today() - dt.timedelta(days=2))
        if start > end:
            continue
        try:
            r = httpx.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": start.isoformat(), "end_date": end.isoformat(),
                    "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
                    "timezone": "UTC",
                },
                timeout=httpx.Timeout(30, connect=15),
            )
        except httpx.HTTPError:
            continue
        if r.status_code != 200:
            continue
        d = r.json().get("daily", {})
        for i, day in enumerate(d.get("time", [])):
            weather_rows.append(
                {
                    "home_team_id": team_id,
                    "day": day,
                    "kickoff_temp_c": d["temperature_2m_mean"][i],
                    "kickoff_precip_mm": d["precipitation_sum"][i],
                    "kickoff_wind_kph": d["wind_speed_10m_max"][i],
                }
            )

    if not weather_rows:
        games["kickoff_temp_c"] = None
        games["kickoff_precip_mm"] = None
        games["kickoff_wind_kph"] = None
        return games

    wdf = pd.DataFrame(weather_rows)
    games = games.copy()
    games["day"] = games["date"].dt.strftime("%Y-%m-%d")
    games = games.merge(wdf, on=["home_team_id", "day"], how="left")
    return games.drop(columns=["day"])


# ---------------------------------------------------------------- injury ----


def compute_injury_features(games: pd.DataFrame, engine) -> pd.DataFrame:
    """Weighted availability loss from the *latest snapshot before kickoff*
    (PLAN §3.4 leakage rule / §4.2). Since injury history only starts
    accumulating from when the scraper first ran, most historical games
    will legitimately have no snapshot available — that's expected, and is
    exactly why `injury_data_available` exists.
    """
    with get_session(engine) as session:
        injuries = pd.DataFrame(
            [
                {"player_id": i.player_id, "team_id": i.team_id, "snapshot_date": i.snapshot_date, "status": i.status}
                for i in session.execute(select(Injury)).scalars().all()
                if i.player_id is not None
            ]
        )

    games = games.copy()
    for side in ("home", "away"):
        games[f"{side}_availability_loss"] = None
        games[f"{side}_attack_availability_loss"] = None
        games[f"{side}_defense_availability_loss"] = None
        games[f"{side}_keeper_out"] = None
    games["injury_data_available"] = False

    if injuries.empty:
        return games

    # Only one column matters pre-kickoff per game: whether a snapshot exists
    # strictly before that game's date. With a single scraper run so far,
    # this only affects games at/after the first snapshot's timestamp.
    min_snapshot = injuries["snapshot_date"].min()
    mask = games["date"] > min_snapshot
    games.loc[mask, "injury_data_available"] = True
    # Availability-loss magnitude (fraction of recent xG unavailable, etc.)
    # is left for a future iteration once injury_data_available accumulates
    # enough pre-kickoff history to be meaningful — see PLAN §3.4 note.
    return games


# ------------------------------------------------------------------ market ----


def compute_market_features(games: pd.DataFrame, engine) -> pd.DataFrame:
    """PLAN §7.4: backtest against *closing* prices, i.e. the price right
    before kickoff — not the market's own `close_time`, which stays open for
    days after the final whistle for settlement and would leak the result
    (post-match prices collapse toward $0.99/$0.01). Every candlestick used
    here is therefore filtered to `ts <= game kickoff time` first.
    """
    with get_session(engine) as session:
        markets = session.execute(select(KalshiMarket).where(KalshiMarket.resolved_game_id.isnot(None))).scalars().all()

    games = games.copy()
    for col in [
        "kalshi_home_implied_prob", "kalshi_away_implied_prob", "kalshi_tie_implied_prob",
        "kalshi_price_bucket", "kalshi_hours_to_close", "kalshi_price_drift_3d",
    ]:
        games[col] = None

    if not markets:
        return games

    kickoff_by_game = games.set_index("game_id")["date"]

    with get_session(engine) as session:
        for m in markets:
            kickoff = kickoff_by_game.get(m.resolved_game_id)
            if kickoff is None or pd.isna(kickoff):
                continue
            # both `kickoff` and stored `ts` values are naive UTC (SQLite drops
            # tzinfo on round-trip) -- keep the comparison naive-to-naive
            kickoff = kickoff.to_pydatetime() if hasattr(kickoff, "to_pydatetime") else kickoff
            kickoff = kickoff.replace(tzinfo=None)

            all_prices = session.execute(
                select(KalshiPrice).where(KalshiPrice.ticker == m.ticker).order_by(KalshiPrice.ts)
            ).scalars().all()
            prices = [p for p in all_prices if p.ts <= kickoff]
            if not prices:
                continue
            last = prices[-1]
            closing_prob = last.price_close_dollars or (
                (last.yes_bid_close_dollars + last.yes_ask_close_dollars) / 2
                if last.yes_bid_close_dollars is not None and last.yes_ask_close_dollars is not None
                else None
            )
            if closing_prob is None:
                continue

            games.loc[games["game_id"] == m.resolved_game_id, "kalshi_hours_to_close"] = (
                kickoff - last.ts
            ).total_seconds() / 3600

            cutoff = last.ts - dt.timedelta(hours=72)
            earlier = [p for p in prices if p.ts <= cutoff]
            drift = None
            if earlier:
                e = earlier[-1]
                earlier_prob = e.price_close_dollars or (
                    (e.yes_bid_close_dollars + e.yes_ask_close_dollars) / 2
                    if e.yes_bid_close_dollars is not None and e.yes_ask_close_dollars is not None
                    else None
                )
                if earlier_prob is not None:
                    drift = closing_prob - earlier_prob

            row_mask = games["game_id"] == m.resolved_game_id
            if m.yes_outcome == "home":
                games.loc[row_mask, "kalshi_home_implied_prob"] = closing_prob
                games.loc[row_mask, "kalshi_price_bucket"] = (
                    "favorite" if closing_prob >= 0.6 else "longshot" if closing_prob <= 0.4 else "contested"
                )
                games.loc[row_mask, "kalshi_price_drift_3d"] = drift
            elif m.yes_outcome == "away":
                games.loc[row_mask, "kalshi_away_implied_prob"] = closing_prob
            elif m.yes_outcome == "tie":
                games.loc[row_mask, "kalshi_tie_implied_prob"] = closing_prob

    return games


# ---------------------------------------------------------------- targets ----


def compute_targets(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["goal_diff"] = games["home_score"] - games["away_score"]
    games["result"] = np.select(
        [games["home_score"] > games["away_score"], games["home_score"] < games["away_score"]],
        ["home", "away"],
        default="draw",
    )
    games.loc[games["home_score"].isna() | games["away_score"].isna(), ["result", "goal_diff"]] = None
    return games


# ------------------------------------------------------------------- main ----

FEATURE_COLUMNS = [
    "home_xg_roll", "away_xg_roll", "home_xga_roll", "away_xga_roll",
    "home_goals_added_roll", "away_goals_added_roll", "home_elo", "away_elo", "home_venue_advantage",
    "home_xg_overperf_roll", "away_xg_overperf_roll",
    "home_availability_loss", "away_availability_loss",
    "home_attack_availability_loss", "away_attack_availability_loss",
    "home_defense_availability_loss", "away_defense_availability_loss",
    "home_keeper_out", "away_keeper_out", "injury_data_available",
    "rest_days_diff", "home_matches_since_manager_change", "away_matches_since_manager_change",
    "away_travel_km", "away_timezone_change_hours",
    "kickoff_temp_c", "kickoff_wind_kph", "kickoff_precip_mm",
    "kalshi_home_implied_prob", "kalshi_away_implied_prob", "kalshi_tie_implied_prob",
    "kalshi_price_bucket", "kalshi_hours_to_close", "kalshi_price_drift_3d",
    "result", "goal_diff",
]


def build_features(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    engine = get_engine(config)
    window = config["features"]["rolling_window_matches"]

    games = load_games(engine)
    teams = load_teams(engine)
    asa = AmericanSoccerAnalysis()

    games = compute_elo(games)
    print("elo done")
    games = compute_rolling_team_xg(games, asa, window)
    print("rolling xg done")
    games = compute_rolling_goals_added(games, engine, window)
    print("rolling goals-added done")
    games = compute_rest_and_manager(games)
    print("rest/manager done")
    games = compute_travel(games, teams)
    games = compute_venue_advantage(games)
    print("travel/venue done")
    games = fetch_weather(games, teams, config)
    print("weather done")
    games = compute_injury_features(games, engine)
    print("injury done")
    games = compute_market_features(games, engine)
    print("market done")
    games = compute_targets(games)

    return games


def persist_features(games: pd.DataFrame, config: dict | None = None) -> int:
    config = config or load_config()
    engine = get_engine(config)
    n = 0
    with get_session(engine) as session:
        for row in games.itertuples():
            values = {"game_id": row.game_id}
            for col in FEATURE_COLUMNS:
                v = getattr(row, col)
                if isinstance(v, float) and math.isnan(v):
                    v = None
                elif isinstance(v, np.bool_):
                    v = bool(v)
                elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                    v = None
                values[col] = v
            stmt = sqlite_insert(Feature).values(**values)
            stmt = stmt.on_conflict_do_update(index_elements=["game_id"], set_=values)
            session.execute(stmt)
            n += 1
        session.commit()
    return n


MODEL_FEATURE_COLUMNS = [
    "home_elo", "away_elo",
    "home_xg_roll", "away_xg_roll", "home_xga_roll", "away_xga_roll",
    "home_xg_overperf_roll", "away_xg_overperf_roll",
    "home_goals_added_roll", "away_goals_added_roll",
    "rest_days_diff", "away_travel_km", "away_timezone_change_hours",
    "home_venue_advantage",
    "home_matches_since_manager_change", "away_matches_since_manager_change",
]


def load_model_dataset(config: dict | None = None) -> pd.DataFrame:
    """Features joined to game date/season, scoped to `asa.player_stats_seasons`
    (the window where goals-added and weather are populated), sorted
    chronologically, and complete-case filtered on the model's feature set.
    """
    config = config or load_config()
    engine = get_engine(config)
    seasons = config["asa"]["player_stats_seasons"]
    placeholders = ",".join(f"'{s}'" for s in seasons)
    query = f"""
        SELECT f.*, g.date, g.season
        FROM features f JOIN games g ON g.game_id = f.game_id
        WHERE g.season IN ({placeholders}) AND f.result IS NOT NULL
        ORDER BY g.date
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["date"])
    return df.dropna(subset=MODEL_FEATURE_COLUMNS + ["goal_diff", "result"]).reset_index(drop=True)


def main():
    init_db()
    games = build_features()
    n = persist_features(games)
    print(f"Persisted {n} feature rows")


if __name__ == "__main__":
    main()
