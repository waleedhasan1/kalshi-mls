# Kalshi MLS Betting Model

A data-science pipeline that trains models on historical MLS matches to estimate
match-outcome probabilities, then compares them against Kalshi contract prices to
look for positive-EV bets in the regions where Kalshi is known to be mispriced.
Built around the favorite–longshot bias documented in Bürgi, Deng & Whelan,
*"Makers and Takers: The Economics of the Kalshi Prediction Market"* (2025) — see
`docs/makers-and-takers-kalshi.pdf`. Full design in `PLAN (2).md`.

**Educational project.** The paper this is built on found that Kalshi
participants net-lose to fees, and any published edge tends to decay as others
exploit it. Nothing here guarantees profit — see `reports/summary.ipynb` for the
honest read on current results, and paper-trade forward before risking real money.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API keys needed — Kalshi's market data and American Soccer Analysis are both
public/unauthenticated. Rotowire's injury feed is scraped politely (see
`config.yaml`'s `injuries.user_agent` / `scrape_delay_seconds`).

## Running the pipeline

Run from `src/`, in order (each step reads the last one's output from
`data/mls.db` or `data/processed/`):

```bash
cd src
python3 db.py               # create the SQLite schema
python3 ingest_kalshi.py    # Kalshi markets + hourly candlesticks
python3 ingest_asa.py       # teams/players/games/player-match-stats
python3 resolve_game.py     # match Kalshi events -> ASA game_ids
python3 scrape_injuries.py  # one timestamped injury snapshot (re-run periodically)
python3 features.py         # build the features table (PLAN §4)
python3 train_mlr.py        # goal-differential regression
python3 train_logistic.py   # win/draw/away multinomial logistic
python3 calibrate.py        # isotonic/Platt calibration, picks whichever wins
python3 evaluate.py         # EV backtest vs Kalshi closing prices
python3 bet_sim.py          # flat vs fractional-Kelly bankroll simulation
python3 viz.py              # all 14 figures -> reports/figures/
```

Then open `reports/summary.ipynb` for the narrated walkthrough (or re-execute it:
`jupyter nbconvert --to notebook --execute --inplace reports/summary.ipynb`).

## Current state / known limits

- **Kalshi market history is short.** `KXMLSGAME` only has a few weeks of
  history at build time, so only ~66 of the 5,762 ASA games have a real resolved
  Kalshi price. `evaluate.py`'s backtest (n=3 model-selected bets) is not yet a
  meaningful sample — re-run it periodically as more games settle.
- **Injury availability-loss features are deferred.** Only one scraper snapshot
  exists so far; `injury_data_available` is live, but the weighted-loss magnitude
  features need snapshot history to accumulate first (see the leakage rule in
  `PLAN (2).md` §3.4 — a snapshot only counts if it predates that match's kickoff).
- **Congestion flag (§4.3.2) is unimplemented** — ASA is MLS-only and doesn't
  carry Concacaf Champions Cup / Leagues Cup fixtures needed to detect it.
- Re-running `scrape_injuries.py` on a schedule (e.g. daily) is what makes the
  injury and price-drift features useful going forward — this is meant to be an
  ongoing pipeline, not a one-shot build.

## Repo layout

See `PLAN (2).md` §8. `src/` holds one module per pipeline stage; `data/mls.db`
(SQLite, gitignored) is the relational core; `data/processed/` holds model
artifacts and backtest CSVs (gitignored); `reports/figures/` holds all plots.
