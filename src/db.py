"""SQLAlchemy engine + schema for the Kalshi MLS project (see PLAN §3.5)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[str] = mapped_column(String, primary_key=True)  # ASA team_id
    team_name: Mapped[str] = mapped_column(String, nullable=False)
    team_abbreviation: Mapped[str | None] = mapped_column(String)
    home_city: Mapped[str | None] = mapped_column(String)
    stadium_name: Mapped[str | None] = mapped_column(String)
    stadium_lat: Mapped[float | None] = mapped_column(Float)
    stadium_lon: Mapped[float | None] = mapped_column(Float)


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[str] = mapped_column(String, primary_key=True)  # ASA player_id
    player_name: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[str | None] = mapped_column(String)
    height_cm: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    nationality: Mapped[str | None] = mapped_column(String)


class Game(Base):
    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String, primary_key=True)  # ASA game_id
    date: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    season: Mapped[str | None] = mapped_column(String)
    home_team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.team_id"), nullable=False)
    away_team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.team_id"), nullable=False)
    home_score: Mapped[int | None] = mapped_column()
    away_score: Mapped[int | None] = mapped_column()
    status: Mapped[str | None] = mapped_column(String)  # scheduled / final / postponed
    stadium_id: Mapped[str | None] = mapped_column(String)
    home_manager_id: Mapped[str | None] = mapped_column(String)
    away_manager_id: Mapped[str | None] = mapped_column(String)

    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])


Index("ix_games_date", Game.date)
Index("ix_games_teams", Game.home_team_id, Game.away_team_id)


class PlayerMatchStat(Base):
    __tablename__ = "player_match_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(String, ForeignKey("players.player_id"), nullable=False)
    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.game_id"), nullable=False)
    team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.team_id"), nullable=False)
    minutes: Mapped[float | None] = mapped_column(Float)
    goals: Mapped[float | None] = mapped_column(Float)
    xgoals: Mapped[float | None] = mapped_column(Float)
    xassists: Mapped[float | None] = mapped_column(Float)
    xgoals_against: Mapped[float | None] = mapped_column(Float)  # keepers
    pass_completion_pct: Mapped[float | None] = mapped_column(Float)
    xpass_completion_pct: Mapped[float | None] = mapped_column(Float)
    goals_added_above_avg: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("player_id", "game_id", name="uq_player_game"),
        Index("ix_pms_game", "game_id"),
        Index("ix_pms_player", "player_id"),
    )


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[str | None] = mapped_column(String, ForeignKey("players.player_id"))
    player_name_raw: Mapped[str] = mapped_column(String, nullable=False)  # as scraped, pre-match
    team_id: Mapped[str | None] = mapped_column(String, ForeignKey("teams.team_id"))
    snapshot_date: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)  # normalized enum, see §3.4.5
    status_raw: Mapped[str | None] = mapped_column(String)
    source: Mapped[str] = mapped_column(String, nullable=False)
    scraped_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_injuries_player_date", "player_id", "snapshot_date"),
    )


class KalshiMarket(Base):
    __tablename__ = "kalshi_markets"

    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    event_ticker: Mapped[str] = mapped_column(String, nullable=False)
    series_ticker: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(String)
    sub_title: Mapped[str | None] = mapped_column(String)
    yes_outcome: Mapped[str | None] = mapped_column(String)  # "home" / "away" / "tie"
    kalshi_team_name: Mapped[str | None] = mapped_column(String)  # raw name for the "yes" side
    resolved_team_id: Mapped[str | None] = mapped_column(String, ForeignKey("teams.team_id"))
    resolved_game_id: Mapped[str | None] = mapped_column(String, ForeignKey("games.game_id"))
    open_time: Mapped[dt.datetime | None] = mapped_column(DateTime)
    close_time: Mapped[dt.datetime | None] = mapped_column(DateTime)
    status: Mapped[str | None] = mapped_column(String)
    result: Mapped[str | None] = mapped_column(String)  # "yes" / "no" once settled

    __table_args__ = (Index("ix_markets_event", "event_ticker"),)


class KalshiPrice(Base):
    __tablename__ = "kalshi_prices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String, ForeignKey("kalshi_markets.ticker"), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)  # candle end_period_ts
    period_interval_minutes: Mapped[int | None] = mapped_column()
    yes_bid_close_dollars: Mapped[float | None] = mapped_column(Float)
    yes_ask_close_dollars: Mapped[float | None] = mapped_column(Float)
    price_close_dollars: Mapped[float | None] = mapped_column(Float)  # last traded price, if any
    volume: Mapped[float | None] = mapped_column(Float)
    open_interest: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "ts", name="uq_price_ticker_ts"),
        Index("ix_prices_ticker_ts", "ticker", "ts"),
    )


class Feature(Base):
    """One model-ready row per game (post feature-engineering). See PLAN §4."""

    __tablename__ = "features"

    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.game_id"), primary_key=True)

    # 4.1 core form & strength
    home_xg_roll: Mapped[float | None] = mapped_column(Float)
    away_xg_roll: Mapped[float | None] = mapped_column(Float)
    home_xga_roll: Mapped[float | None] = mapped_column(Float)
    away_xga_roll: Mapped[float | None] = mapped_column(Float)
    home_goals_added_roll: Mapped[float | None] = mapped_column(Float)
    away_goals_added_roll: Mapped[float | None] = mapped_column(Float)
    home_elo: Mapped[float | None] = mapped_column(Float)
    away_elo: Mapped[float | None] = mapped_column(Float)
    home_venue_advantage: Mapped[float | None] = mapped_column(Float)
    home_xg_overperf_roll: Mapped[float | None] = mapped_column(Float)
    away_xg_overperf_roll: Mapped[float | None] = mapped_column(Float)

    # 4.2 injury / availability
    home_availability_loss: Mapped[float | None] = mapped_column(Float)
    away_availability_loss: Mapped[float | None] = mapped_column(Float)
    home_attack_availability_loss: Mapped[float | None] = mapped_column(Float)
    away_attack_availability_loss: Mapped[float | None] = mapped_column(Float)
    home_defense_availability_loss: Mapped[float | None] = mapped_column(Float)
    away_defense_availability_loss: Mapped[float | None] = mapped_column(Float)
    home_keeper_out: Mapped[bool | None] = mapped_column(Boolean)
    away_keeper_out: Mapped[bool | None] = mapped_column(Boolean)
    injury_data_available: Mapped[bool] = mapped_column(Boolean, default=False)

    # 4.3 situational
    rest_days_diff: Mapped[float | None] = mapped_column(Float)
    home_congestion_flag: Mapped[bool | None] = mapped_column(Boolean)
    away_congestion_flag: Mapped[bool | None] = mapped_column(Boolean)
    home_matches_since_manager_change: Mapped[int | None] = mapped_column()
    away_matches_since_manager_change: Mapped[int | None] = mapped_column()

    # 4.4 location & environment
    away_travel_km: Mapped[float | None] = mapped_column(Float)
    away_timezone_change_hours: Mapped[float | None] = mapped_column(Float)
    kickoff_temp_c: Mapped[float | None] = mapped_column(Float)
    kickoff_wind_kph: Mapped[float | None] = mapped_column(Float)
    kickoff_precip_mm: Mapped[float | None] = mapped_column(Float)

    # 4.5 market-derived
    kalshi_home_implied_prob: Mapped[float | None] = mapped_column(Float)
    kalshi_away_implied_prob: Mapped[float | None] = mapped_column(Float)
    kalshi_tie_implied_prob: Mapped[float | None] = mapped_column(Float)
    kalshi_price_bucket: Mapped[str | None] = mapped_column(String)
    kalshi_hours_to_close: Mapped[float | None] = mapped_column(Float)
    kalshi_price_drift_3d: Mapped[float | None] = mapped_column(Float)

    # targets
    result: Mapped[str | None] = mapped_column(String)  # "home" / "draw" / "away"
    goal_diff: Mapped[int | None] = mapped_column()


class Bet(Base):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.game_id"), nullable=False)
    kalshi_ticker: Mapped[str | None] = mapped_column(String, ForeignKey("kalshi_markets.ticker"))
    side: Mapped[str] = mapped_column(String, nullable=False)  # home / draw / away
    stake: Mapped[float] = mapped_column(Float, nullable=False)
    price_paid: Mapped[float] = mapped_column(Float, nullable=False)
    model_prob: Mapped[float] = mapped_column(Float, nullable=False)
    ev: Mapped[float] = mapped_column(Float, nullable=False)
    outcome: Mapped[str | None] = mapped_column(String)  # win / loss, filled post-settlement
    pnl: Mapped[float | None] = mapped_column(Float)
    is_backtest: Mapped[bool] = mapped_column(Boolean, default=True)
    placed_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


def get_engine(config: dict | None = None):
    config = config or load_config()
    db_path = PROJECT_ROOT / config["db"]["path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def init_db(engine=None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def get_session(engine=None) -> Session:
    engine = engine or get_engine()
    return Session(engine)


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {get_engine().url}")
