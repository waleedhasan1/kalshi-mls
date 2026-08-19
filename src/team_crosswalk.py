"""Kalshi <-> ASA team crosswalk (PLAN §3.2.2).

Kalshi's MLS event sub_titles use short abbreviations ("STL vs DAL (Aug 30)")
that mostly, but not always, match American Soccer Analysis's own
`team_abbreviation`. Built once by hand against the live `KXMLSGAME` series
and ASA's `get_teams(leagues="mls")` output — most robust for a fixed ~30
team league; avoids fuzzy-match errors on short, easily-confused codes
(e.g. "SD" / "SJ", "NE" / "NYC").

Mismatches confirmed by inspection (2026-08-19):
  - FC Dallas:      Kalshi "DAL"  -> ASA abbreviation "FCD"
  - NE Revolution:  Kalshi "NE"   -> ASA abbreviation "NER"
  - San Jose:       Kalshi "SJ"   -> ASA abbreviation "SJE"
  - CF Montreal:    Kalshi uses both "MTL" and "MON" -> ASA abbreviation "MTL"

Chivas USA is defunct (no longer fields Kalshi markets) and is intentionally
left out of this crosswalk.
"""

from __future__ import annotations

# Kalshi abbreviation (as seen in event sub_title / market ticker) -> ASA team_id
KALSHI_ABBR_TO_ASA_TEAM_ID: dict[str, str] = {
    "ATL": "KAqBN0Vqbg",   # Atlanta United FC
    "ATX": "gpMOLwl5zy",   # Austin FC
    "MTL": "APk5LGOMOW",   # CF Montreal
    "MON": "APk5LGOMOW",   # CF Montreal (alt abbreviation seen on Kalshi)
    "CLT": "NPqxKXZ59d",   # Charlotte FC
    "CHI": "X0Oq66zq6D",   # Chicago Fire FC
    "COL": "pzeQZ6xQKw",   # Colorado Rapids
    "CLB": "mvzqoLZQap",   # Columbus Crew
    "DCU": "EKXMeX3Q64",   # D.C. United
    "CIN": "NWMWlBK5lz",   # FC Cincinnati
    "DAL": "mKAqBBmqbg",   # FC Dallas (ASA abbr "FCD")
    "HOU": "YgOMngl5wN",   # Houston Dynamo FC
    "MIA": "zeQZkL1MKw",   # Inter Miami CF
    "LAG": "kaDQ0wRqEv",   # LA Galaxy
    "LAFC": "eVq3ya6MWO",  # Los Angeles FC
    "MIN": "kRQand1MKZ",   # Minnesota United FC
    "NSH": "vzqoOgNqap",   # Nashville SC
    "NE": "19vQ2095K6",    # New England Revolution (ASA abbr "NER")
    "NYC": "Vj58weDM8n",   # New York City FC
    "NYRB": "a2lqRX2Mr0",  # New York Red Bulls
    "ORL": "jYQJ8EW5GR",   # Orlando City SC
    "PHI": "9z5k7Yg5A3",   # Philadelphia Union
    "POR": "WBLMvYAQxe",   # Portland Timbers FC
    "RSL": "a2lqR4JMr0",   # Real Salt Lake
    "SD": "zeQZBOzQKw",    # San Diego FC
    "SJ": "0KPqjA456v",    # San Jose Earthquakes (ASA abbr "SJE")
    "SEA": "jYQJ19EqGR",   # Seattle Sounders FC
    "SKC": "Z2vQ1xlqrA",   # Sporting Kansas City
    "STL": "wvq9B9wQWn",   # St. Louis City SC
    "TOR": "kRQabn8MKZ",   # Toronto FC
    "VAN": "lgpMOvnQzy",   # Vancouver Whitecaps FC
}


def kalshi_abbr_to_asa_team_id(abbr: str) -> str | None:
    """Look up an ASA team_id for a Kalshi team abbreviation. None if unknown."""
    return KALSHI_ABBR_TO_ASA_TEAM_ID.get(abbr.strip().upper())


# Rotowire's soccer injury-report JSON (`/soccer/tables/injury-report.php?league=MLS`)
# uses a third, inconsistent abbreviation set (sometimes city code, sometimes nickname
# code) that matches neither Kalshi's nor ASA's. Built by hand (2026-08-19) by cross-
# referencing each Rotowire abbreviation's sample player against ASA's own team
# rosters — see the ingest_asa-populated `players`/`player_match_stats` tables.
ROTOWIRE_ABBR_TO_ASA_TEAM_ID: dict[str, str] = {
    "ATL": "KAqBN0Vqbg",   # Atlanta United FC
    "ATX": "gpMOLwl5zy",   # Austin FC
    "CLB": "mvzqoLZQap",   # Columbus Crew
    "CLT": "NPqxKXZ59d",   # Charlotte FC
    "DAL": "mKAqBBmqbg",   # FC Dallas
    "DCU": "EKXMeX3Q64",   # D.C. United
    "FIR": "X0Oq66zq6D",   # Chicago Fire FC
    "GAL": "kaDQ0wRqEv",   # LA Galaxy
    "HOU": "YgOMngl5wN",   # Houston Dynamo FC
    "LAF": "eVq3ya6MWO",   # Los Angeles FC
    "MIA": "zeQZkL1MKw",   # Inter Miami CF
    "MNU": "kRQand1MKZ",   # Minnesota United FC
    "MTL": "APk5LGOMOW",   # CF Montreal
    "NER": "19vQ2095K6",   # New England Revolution
    "NSH": "vzqoOgNqap",   # Nashville SC
    "NYC": "Vj58weDM8n",   # New York City FC
    "NYR": "a2lqRX2Mr0",   # New York Red Bulls
    "ORL": "jYQJ8EW5GR",   # Orlando City SC
    "PHI": "9z5k7Yg5A3",   # Philadelphia Union
    "POR": "WBLMvYAQxe",   # Portland Timbers FC
    "RAP": "pzeQZ6xQKw",   # Colorado Rapids
    "RSL": "a2lqR4JMr0",   # Real Salt Lake
    "SD": "zeQZBOzQKw",    # San Diego FC
    "SJE": "0KPqjA456v",   # San Jose Earthquakes
    "SKC": "Z2vQ1xlqrA",   # Sporting Kansas City
    "SOU": "jYQJ19EqGR",   # Seattle Sounders FC
    "STL": "wvq9B9wQWn",   # St. Louis City SC
    "TOR": "kRQabn8MKZ",   # Toronto FC
    "WHI": "lgpMOvnQzy",   # Vancouver Whitecaps FC
    "CIN": "NWMWlBK5lz",   # FC Cincinnati (no injuries at build time, added for completeness)
}


def rotowire_abbr_to_asa_team_id(abbr: str) -> str | None:
    """Look up an ASA team_id for a Rotowire team abbreviation. None if unknown."""
    return ROTOWIRE_ABBR_TO_ASA_TEAM_ID.get(abbr.strip().upper())
