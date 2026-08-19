"""American Soccer Analysis ingest (PLAN §3.2.5, §10.3).

Populates teams / players / games / player_match_stats from the
`itscalledsoccer` client. Games are pulled for all available seasons
(cheap, and useful for Elo history); per-player match stats are pulled only
for `asa.player_stats_seasons` in config.yaml, since squads several years
old add little value to a live model and pulling them is the expensive part.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter

import time

import pandas as pd
from itscalledsoccer.client import AmericanSoccerAnalysis
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db import Game, Player, PlayerMatchStat, Team, get_engine, get_session, init_db, load_config


def _retry(fn, *args, attempts: int = 4, **kwargs):
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _nan_to_none(v):
    return None if pd.isna(v) else v


def ingest_teams_and_stadia(asa: AmericanSoccerAnalysis, config: dict) -> None:
    league = config["asa"]["league"]
    teams = asa.get_teams(leagues=league)
    stadia = asa.get_stadia(leagues=league).set_index("stadium_id")
    games = asa.get_games(leagues=league)

    # most common home stadium per team, to attach coords to a team row
    home_stadium = (
        games.dropna(subset=["stadium_id"])
        .groupby("home_team_id")["stadium_id"]
        .agg(lambda s: Counter(s).most_common(1)[0][0])
    )

    engine = get_engine(config)
    with get_session(engine) as session:
        for _, t in teams.iterrows():
            stadium_id = home_stadium.get(t["team_id"])
            stadium = stadia.loc[stadium_id] if stadium_id in stadia.index else None
            row = {
                "team_id": t["team_id"],
                "team_name": t["team_name"],
                "team_abbreviation": _nan_to_none(t.get("team_abbreviation")),
                "home_city": _nan_to_none(stadium["city"]) if stadium is not None else None,
                "stadium_name": _nan_to_none(stadium["stadium_name"]) if stadium is not None else None,
                "stadium_lat": _nan_to_none(stadium["latitude"]) if stadium is not None else None,
                "stadium_lon": _nan_to_none(stadium["longitude"]) if stadium is not None else None,
            }
            stmt = sqlite_insert(Team).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=["team_id"], set_=row)
            session.execute(stmt)
        session.commit()
    print(f"Upserted {len(teams)} teams")


def ingest_games(asa: AmericanSoccerAnalysis, config: dict) -> pd.DataFrame:
    league = config["asa"]["league"]
    games = asa.get_games(leagues=league)
    engine = get_engine(config)
    with get_session(engine) as session:
        for _, g in games.iterrows():
            row = {
                "game_id": g["game_id"],
                "date": dt.datetime.fromisoformat(g["date_time_utc"].replace(" UTC", "+00:00")),
                "season": _nan_to_none(g.get("season_name")),
                "home_team_id": g["home_team_id"],
                "away_team_id": g["away_team_id"],
                "home_score": int(g["home_score"]) if not pd.isna(g["home_score"]) else None,
                "away_score": int(g["away_score"]) if not pd.isna(g["away_score"]) else None,
                "status": _nan_to_none(g.get("status")),
            }
            stmt = sqlite_insert(Game).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=["game_id"], set_=row)
            session.execute(stmt)
        session.commit()
    print(f"Upserted {len(games)} games")
    return games


def ingest_players(asa: AmericanSoccerAnalysis, config: dict) -> None:
    league = config["asa"]["league"]
    players = asa.get_players(leagues=league)
    engine = get_engine(config)
    with get_session(engine) as session:
        for _, p in players.iterrows():
            height_cm = None
            if not pd.isna(p.get("height_ft")):
                height_cm = ((p["height_ft"] or 0) * 12 + (p.get("height_in") or 0)) * 2.54
            weight_kg = _nan_to_none(p.get("weight_lb"))
            weight_kg = weight_kg * 0.453592 if weight_kg is not None else None
            position = _nan_to_none(p.get("primary_general_position")) or _nan_to_none(
                p.get("primary_broad_position")
            )
            row = {
                "player_id": p["player_id"],
                "player_name": p["player_name"],
                "position": position,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "nationality": _nan_to_none(p.get("nationality")),
            }
            stmt = sqlite_insert(Player).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=["player_id"], set_=row)
            session.execute(stmt)
        session.commit()
    print(f"Upserted {len(players)} players")


def ingest_player_match_stats(asa: AmericanSoccerAnalysis, config: dict) -> None:
    league = config["asa"]["league"]
    seasons = config["asa"]["player_stats_seasons"]

    def pull_per_season(method, **kwargs):
        frames = []
        for season in seasons:
            frames.append(_retry(method, leagues=league, season_name=season, split_by_games=True, **kwargs))
        return pd.concat(frames, ignore_index=True)

    xg = pull_per_season(asa.get_player_xgoals)
    xp = pull_per_season(asa.get_player_xpass)
    ga = pull_per_season(asa.get_player_goals_added)
    gk = pull_per_season(asa.get_goalkeeper_xgoals)

    # collapse goals_added's nested action-type list into one total per player-game
    ga = ga.copy()
    ga["goals_added_above_avg"] = ga["data"].apply(
        lambda actions: sum(a["goals_added_above_avg"] for a in actions) if isinstance(actions, list) else None
    )

    key = ["player_id", "game_id", "team_id"]
    merged = xg.merge(xp[key + ["pass_completion_percentage", "xpass_completion_percentage"]], on=key, how="left")
    merged = merged.merge(ga[key + ["goals_added_above_avg"]], on=key, how="left")
    merged = merged.merge(
        gk[["player_id", "game_id", "team_id", "xgoals_gk_faced"]].rename(
            columns={"xgoals_gk_faced": "xgoals_against"}
        ),
        on=key,
        how="left",
    )

    engine = get_engine(config)
    with get_session(engine) as session:
        for _, r in merged.iterrows():
            row = {
                "player_id": r["player_id"],
                "game_id": r["game_id"],
                "team_id": r["team_id"],
                "minutes": _nan_to_none(r.get("minutes_played")),
                "goals": _nan_to_none(r.get("goals")),
                "xgoals": _nan_to_none(r.get("xgoals")),
                "xassists": _nan_to_none(r.get("xassists")),
                "xgoals_against": _nan_to_none(r.get("xgoals_against")),
                "pass_completion_pct": _nan_to_none(r.get("pass_completion_percentage")),
                "xpass_completion_pct": _nan_to_none(r.get("xpass_completion_percentage")),
                "goals_added_above_avg": _nan_to_none(r.get("goals_added_above_avg")),
            }
            stmt = sqlite_insert(PlayerMatchStat).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=["player_id", "game_id"], set_=row)
            session.execute(stmt)
        session.commit()
    print(f"Upserted {len(merged)} player_match_stats rows (seasons {seasons})")


def main():
    init_db()
    config = load_config()
    asa = AmericanSoccerAnalysis()

    print("Ingesting teams + stadia...")
    ingest_teams_and_stadia(asa, config)

    print("Ingesting games (all seasons)...")
    ingest_games(asa, config)

    print("Ingesting players...")
    ingest_players(asa, config)

    print("Ingesting player match stats...")
    ingest_player_match_stats(asa, config)

    print("Done.")


if __name__ == "__main__":
    main()
