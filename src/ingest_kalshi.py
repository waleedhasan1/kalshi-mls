"""Kalshi public market data ingest for MLS (PLAN §3.3, §10.2).

Read-only, unauthenticated access to Kalshi's public market data. Pulls
events/markets under the KXMLSGAME series into `kalshi_markets`, and
historical candlesticks into `kalshi_prices`.
"""

from __future__ import annotations

import datetime as dt
import re
import time

import httpx
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db import KalshiMarket, KalshiPrice, get_engine, get_session, init_db, load_config
from team_crosswalk import kalshi_abbr_to_asa_team_id

SUBTITLE_RE = re.compile(r"^(?P<home>[\w]+) vs (?P<away>[\w]+) \((?P<date>[^)]+)\)$")


class KalshiClient:
    def __init__(self, config: dict | None = None):
        config = config or load_config()
        self.base_url = config["kalshi"]["base_url"]
        self.min_interval = 1.0 / config["kalshi"].get("rate_limit_per_sec", 15)
        self._client = httpx.Client(base_url=self.base_url, timeout=20)
        self._last_call = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            r = self._client.get(path, params=params)
            self._last_call = time.monotonic()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    def get_series(self, limit: int = 200) -> list[dict]:
        return self._get("/series", {"limit": limit}).get("series", [])

    def iter_events(self, series_ticker: str, with_nested_markets: bool = False):
        cursor = None
        while True:
            params = {"series_ticker": series_ticker, "limit": 200}
            if with_nested_markets:
                params["with_nested_markets"] = "true"
            if cursor:
                params["cursor"] = cursor
            data = self._get("/events", params)
            yield from data.get("events", [])
            cursor = data.get("cursor")
            if not cursor:
                break

    def get_markets_for_event(self, event_ticker: str) -> list[dict]:
        return self._get("/markets", {"event_ticker": event_ticker}).get("markets", [])

    def get_candlesticks(
        self, series_ticker: str, market_ticker: str, start_ts: int, end_ts: int, period_interval: int = 60
    ) -> list[dict]:
        data = self._get(
            f"/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        return data.get("candlesticks", [])

    def close(self):
        self._client.close()


def parse_event_subtitle(sub_title: str) -> tuple[str, str, str] | None:
    """'STL vs DAL (Aug 30)' -> ('STL', 'DAL', 'Aug 30')."""
    m = SUBTITLE_RE.match(sub_title.strip())
    if not m:
        return None
    return m.group("home"), m.group("away"), m.group("date")


def determine_yes_outcome(market: dict, home_abbr: str, away_abbr: str) -> tuple[str, str | None]:
    """Return ("home"|"away"|"tie", kalshi_team_name_or_None) for a market's Yes side."""
    yes_sub = (market.get("yes_sub_title") or "").strip()
    title = (market.get("title") or "")
    if "tie" in title.lower() or yes_sub.lower() == "tie":
        return "tie", None
    team = market.get("custom_strike", {}).get("soccer_team")
    ticker_suffix = market["ticker"].split("-")[-1]
    if ticker_suffix.upper() == home_abbr.upper():
        return "home", yes_sub
    if ticker_suffix.upper() == away_abbr.upper():
        return "away", yes_sub
    return "tie", yes_sub


def ingest_markets(client: KalshiClient, config: dict) -> list[dict]:
    """Pull all KXMLSGAME events/markets and upsert into kalshi_markets.

    Returns the list of market dicts ingested (used by ingest_prices to
    avoid a second API round-trip).
    """
    series_ticker = config["kalshi"]["mls_series_ticker"]
    engine = get_engine(config)
    all_markets: list[dict] = []

    with get_session(engine) as session:
        for event in client.iter_events(series_ticker):
            parsed = parse_event_subtitle(event.get("sub_title", ""))
            if not parsed:
                continue
            home_abbr, away_abbr, _ = parsed
            home_team_id = kalshi_abbr_to_asa_team_id(home_abbr)
            away_team_id = kalshi_abbr_to_asa_team_id(away_abbr)

            markets = client.get_markets_for_event(event["event_ticker"])
            for m in markets:
                outcome, kalshi_name = determine_yes_outcome(m, home_abbr, away_abbr)
                resolved_team_id = (
                    home_team_id if outcome == "home" else away_team_id if outcome == "away" else None
                )
                row = {
                    "ticker": m["ticker"],
                    "event_ticker": event["event_ticker"],
                    "series_ticker": series_ticker,
                    "title": m.get("title"),
                    "sub_title": event.get("sub_title"),
                    "yes_outcome": outcome,
                    "kalshi_team_name": kalshi_name,
                    "resolved_team_id": resolved_team_id,
                    "resolved_game_id": None,  # filled by resolve_game.py
                    "open_time": _parse_ts(m.get("open_time")),
                    "close_time": _parse_ts(m.get("close_time")),
                    "status": m.get("status"),
                    "result": m.get("result") or None,
                }
                stmt = sqlite_insert(KalshiMarket).values(**row)
                stmt = stmt.on_conflict_do_update(index_elements=["ticker"], set_=row)
                session.execute(stmt)
                all_markets.append(m)
        session.commit()
    return all_markets


def _parse_ts(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def ingest_prices(client: KalshiClient, config: dict, markets: list[dict], period_interval: int = 60) -> None:
    """Pull candlesticks for each market ticker and upsert into kalshi_prices."""
    series_ticker = config["kalshi"]["mls_series_ticker"]
    engine = get_engine(config)

    with get_session(engine) as session:
        for m in markets:
            ticker = m["ticker"]
            open_ts = _parse_ts(m.get("open_time"))
            close_ts = _parse_ts(m.get("close_time"))
            if not open_ts or not close_ts:
                continue
            start = int(open_ts.timestamp())
            end = int(close_ts.timestamp())
            if end <= start:
                continue
            try:
                candles = client.get_candlesticks(series_ticker, ticker, start, end, period_interval)
            except httpx.HTTPStatusError:
                continue
            for c in candles:
                yes_bid = c.get("yes_bid", {})
                yes_ask = c.get("yes_ask", {})
                price = c.get("price", {})
                row = {
                    "ticker": ticker,
                    "ts": dt.datetime.fromtimestamp(c["end_period_ts"], tz=dt.timezone.utc),
                    "period_interval_minutes": period_interval,
                    "yes_bid_close_dollars": _to_float(yes_bid.get("close_dollars")),
                    "yes_ask_close_dollars": _to_float(yes_ask.get("close_dollars")),
                    "price_close_dollars": _to_float(price.get("close_dollars")),
                    "volume": _to_float(c.get("volume_fp")),
                    "open_interest": _to_float(c.get("open_interest_fp")),
                }
                stmt = sqlite_insert(KalshiPrice).values(**row)
                stmt = stmt.on_conflict_do_update(index_elements=["ticker", "ts"], set_=row)
                session.execute(stmt)
            session.commit()


def _to_float(v) -> float | None:
    if v in (None, ""):
        return None
    return float(v)


def main():
    init_db()
    config = load_config()
    client = KalshiClient(config)
    try:
        print("Confirming MLS series exists...")
        series = client.get_series()
        mls = [s for s in series if s["ticker"] == config["kalshi"]["mls_series_ticker"]]
        assert mls, "KXMLSGAME series not found — Kalshi may have delisted per-game MLS markets"
        print(f"Found series: {mls[0]['ticker']} - {mls[0]['title']}")

        print("Ingesting markets...")
        markets = ingest_markets(client, config)
        print(f"Ingested {len(markets)} markets")

        print("Ingesting candlesticks (this can take a while)...")
        ingest_prices(client, config, markets)
        print("Done.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
