"""MLS injury/availability scraper (PLAN §3.4, §10.5).

Source: Rotowire's soccer injury table (`/soccer/tables/injury-report.php`),
the JSON feed backing https://www.rotowire.com/soccer/injury-report.php?league=MLS.
robots.txt allows generic user agents on this path (only /account/, /forum/,
login and a handful of legacy game paths are disallowed).

Every run is a timestamped snapshot — see the leakage warning in PLAN §3.4:
an injury feature is only valid if a backtest joins the *snapshot dated
before that match's kickoff*, never today's table applied to a past game.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
from rapidfuzz import fuzz, process
from sqlalchemy import select

from db import Injury, Player, PlayerMatchStat, get_engine, get_session, init_db, load_config, PROJECT_ROOT
from team_crosswalk import rotowire_abbr_to_asa_team_id

STATUS_MAP = {
    "OUT": "out",
    "SUS": "suspended",
    "GTD": "questionable",
    "DTD": "doubtful",
    "IR": "out",
}

FUZZY_MATCH_THRESHOLD = 88


def fetch_raw(config: dict) -> tuple[dict, Path]:
    icfg = config["injuries"]
    url = "https://www.rotowire.com/soccer/tables/injury-report.php"
    headers = {
        "User-Agent": icfg["user_agent"],
        "Referer": "https://www.rotowire.com/soccer/injury-report.php?league=MLS",
    }
    r = httpx.get(url, params={"league": "MLS"}, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    raw_dir = PROJECT_ROOT / icfg["raw_html_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = raw_dir / f"rotowire_mls_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.json"
    snapshot_path.write_text(r.text)
    return data, snapshot_path


def _asa_player_pool_by_team(session) -> dict[str, list[tuple[str, str]]]:
    """team_id -> [(player_id, player_name), ...] from recent appearances."""
    rows = session.execute(
        select(PlayerMatchStat.team_id, Player.player_id, Player.player_name)
        .join(Player, Player.player_id == PlayerMatchStat.player_id)
        .distinct()
    ).all()
    pool: dict[str, list[tuple[str, str]]] = {}
    for team_id, player_id, player_name in rows:
        pool.setdefault(team_id, []).append((player_id, player_name))
    return pool


def match_player(name: str, team_id: str | None, pool: dict[str, list[tuple[str, str]]]) -> str | None:
    if not team_id or team_id not in pool:
        return None
    candidates = pool[team_id]
    choices = {pid: pname for pid, pname in candidates}
    result = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=FUZZY_MATCH_THRESHOLD)
    if result:
        _, _score, player_id = result
        return player_id
    return None


def ingest_injury_snapshot(config: dict | None = None) -> dict:
    config = config or load_config()
    init_db()
    data, snapshot_path = fetch_raw(config)

    now = dt.datetime.now(dt.timezone.utc)
    engine = get_engine(config)
    unmatched: list[str] = []

    with get_session(engine) as session:
        pool = _asa_player_pool_by_team(session)

        for row in data:
            team_id = rotowire_abbr_to_asa_team_id(row.get("team", ""))
            player_id = match_player(row.get("player", ""), team_id, pool)
            if player_id is None:
                unmatched.append(f"{row.get('player')} ({row.get('team')})")

            status_raw = row.get("status", "")
            status = STATUS_MAP.get(status_raw, "doubtful")

            injury = Injury(
                player_id=player_id,
                player_name_raw=row.get("player", ""),
                team_id=team_id,
                snapshot_date=now,
                status=status,
                status_raw=f"{status_raw}: {row.get('injury', '')}",
                source="rotowire",
                scraped_at=now,
            )
            session.add(injury)
        session.commit()

    if unmatched:
        log_path = snapshot_path.with_suffix(".unmatched.log")
        log_path.write_text("\n".join(unmatched))
        print(f"{len(unmatched)} unmatched player names logged to {log_path}")

    return {"total": len(data), "unmatched": len(unmatched), "snapshot": str(snapshot_path)}


if __name__ == "__main__":
    result = ingest_injury_snapshot()
    print(f"Ingested {result['total']} injury rows ({result['unmatched']} unmatched) from {result['snapshot']}")
