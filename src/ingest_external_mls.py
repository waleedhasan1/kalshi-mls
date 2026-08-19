"""Supplementary historical MLS results, pre-dating ASA's own coverage.

American Soccer Analysis's `get_games` only goes back to 2013. footballcsv's
`major-league-soccer` repo (public domain / CC0, no key) has clean season-by-
season results back to the league's 1996 debut. This extends `games` with
1996-2012 (the years ASA doesn't have) so Elo, rest-days, travel and venue
features -- which only need date/teams/score, not ASA's xG data -- get a much
longer, more independent history to warm up on and to backtest against.

These extra games carry no play-level ASA stats (no player_match_stats, no
team-level xG), so `home_xg_roll` etc. stay NULL for them -- exactly the kind
of gap `train_long_history.py`'s reduced feature set is built to work around.

Source: https://github.com/footballcsv/major-league-soccer (CC0)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re

import httpx
import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db import Game, Team, get_engine, get_session, init_db, load_config

RAW_URL = "https://raw.githubusercontent.com/footballcsv/major-league-soccer/master/{year}/1-mls.csv"
YEARS = range(1996, 2013)  # ASA covers 2013 onward already

DATE_RE = re.compile(r"\((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\)\s*(\d{1,2} \w{3} \d{4})")
SCORE_RE = re.compile(r"^(\d+)-(\d+)")
TEAM_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")

# footballcsv team name -> ASA team_id, hand-verified against `teams` (§ same
# discipline as team_crosswalk.py: small fixed set, exact map beats fuzzy match).
NAME_TO_TEAM_ID: dict[str, str] = {
    "CD Chivas USA": "4wM42l4qjB",
    "Chicago Fire": "X0Oq66zq6D",
    "Colorado Rapids": "pzeQZ6xQKw",
    "Columbus Crew SC": "mvzqoLZQap",
    "D.C. United": "EKXMeX3Q64",
    "FC Dallas": "mKAqBBmqbg",
    "Houston Dynamo": "YgOMngl5wN",
    "LA Galaxy": "kaDQ0wRqEv",
    "Montreal Impact": "APk5LGOMOW",  # renamed CF Montreal in 2021
    "New England Revolution": "19vQ2095K6",
    "New York Red Bulls": "a2lqRX2Mr0",
    "Philadelphia Union": "9z5k7Yg5A3",
    "Portland Timbers": "WBLMvYAQxe",
    "Real Salt Lake": "a2lqR4JMr0",
    "San Jose Earthquakes": "0KPqjA456v",
    "Seattle Sounders FC": "jYQJ19EqGR",
    "Sporting Kansas City": "Z2vQ1xlqrA",
    "Toronto FC": "kRQabn8MKZ",
    "Vancouver Whitecaps FC": "lgpMOvnQzy",
    # defunct, folded 2001, no successor franchise in ASA's team list
    "Miami Fusion": "ext_miami_fusion",
    "Tampa Bay Mutiny": "ext_tampa_bay_mutiny",
}

EXTRA_TEAMS = [
    {"team_id": "ext_miami_fusion", "team_name": "Miami Fusion"},
    {"team_id": "ext_tampa_bay_mutiny", "team_name": "Tampa Bay Mutiny"},
]


def fetch_season(year: int) -> pd.DataFrame:
    r = httpx.get(RAW_URL.format(year=year), timeout=20)
    r.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(r.text))


def parse_date(raw: str) -> dt.datetime | None:
    m = DATE_RE.search(raw)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%d %b %Y")


def clean_team(raw: str) -> str:
    return TEAM_SUFFIX_RE.sub("", raw).strip()


def make_game_id(date: dt.datetime, home_id: str, away_id: str) -> str:
    key = f"{date.date().isoformat()}|{home_id}|{away_id}"
    return "ext_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def parse_season(df: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    unmapped = set()
    for r in df.to_dict("records"):
        date = parse_date(str(r.get("Date", "")))
        home_name, away_name = clean_team(str(r.get("Team 1", ""))), clean_team(str(r.get("Team 2", "")))
        ft = str(r.get("FT", ""))
        m = SCORE_RE.match(ft)
        if not date or not m:
            continue
        home_id = NAME_TO_TEAM_ID.get(home_name)
        away_id = NAME_TO_TEAM_ID.get(away_name)
        if not home_id or not away_id:
            unmapped.update({home_name, away_name} - set(NAME_TO_TEAM_ID))
            continue
        rows.append(
            {
                "game_id": make_game_id(date, home_id, away_id),
                "date": date,
                "season": str(year),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_score": int(m.group(1)),
                "away_score": int(m.group(2)),
                "status": "FullTime",
            }
        )
    if unmapped:
        print(f"  {year}: unmapped team names (skipped their games): {unmapped}")
    return rows


def ingest(config: dict | None = None) -> int:
    config = config or load_config()
    init_db()
    engine = get_engine(config)

    with get_session(engine) as session:
        for t in EXTRA_TEAMS:
            stmt = sqlite_insert(Team).values(**t)
            stmt = stmt.on_conflict_do_nothing(index_elements=["team_id"])
            session.execute(stmt)
        session.commit()

    all_rows = []
    for year in YEARS:
        try:
            df = fetch_season(year)
        except httpx.HTTPError as e:
            print(f"  {year}: fetch failed ({e}), skipping")
            continue
        rows = parse_season(df, year)
        print(f"  {year}: {len(rows)} games")
        all_rows.extend(rows)

    with get_session(engine) as session:
        for row in all_rows:
            stmt = sqlite_insert(Game).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=["game_id"], set_=row)
            session.execute(stmt)
        session.commit()

    print(f"Upserted {len(all_rows)} external games ({YEARS.start}-{YEARS.stop - 1})")
    return len(all_rows)


if __name__ == "__main__":
    ingest()
