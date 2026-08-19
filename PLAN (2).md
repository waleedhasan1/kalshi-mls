# Kalshi MLS Betting Model — Project Plan

A data-science project that trains models on historical MLS matches to estimate
match outcome probabilities, then compares those probabilities against Kalshi
contract prices to identify positive expected-value (EV) bets. Built around the
documented favorite–longshot bias in Kalshi markets.

---

## 1. Objective

Two models, two targets:

1. **Multinomial logistic regression** → `P(home win / draw / away win)`.
   MLS has draws, so this is a three-outcome problem, not binary. This model
   feeds the Kalshi EV strategy directly.
2. **Multiple linear regression (MLR)** → predicted **goal differential**
   (`home_goals − away_goals`) as a continuous target. Interpretable secondary
   signal; can be mapped back to win/draw/loss probabilities via the residual
   distribution to cross-check the logistic model.

The edge is not "predict soccer better than everyone." It is: **be better
calibrated than Kalshi's prices, net of fees, in the regions where Kalshi is
known to be mispriced.**

---

## 2. The Kalshi Edge

Based on Bürgi, Deng & Whelan, *"Makers and Takers: The Economics of the Kalshi
Prediction Market"* (2025), which analyzed 300,000+ contracts:

- **Low-priced (longshot) contracts are overpriced** — they win less often than
  their price implies. Fade or avoid.
- **High-priced (favorite) contracts are underpriced** — they win more often and
  yield small positive returns. This is where the edge lives.
- The bias exists for both Makers and Takers but is **more pronounced on the
  Taker side**.
- Kalshi earns on commissions and participants net-lose, so any edge must clear
  **both the bias and the fees**.

**Strategic implication:** the model does not need to beat the whole market. It
needs to find high-probability favorites where Kalshi's price still leaves room
after fees, and systematically avoid the overpriced longshot side.

---

## 3. Data Intake

### 3.1 Sources

| Data | Source | Access |
|---|---|---|
| Games, players, teams, xG, xPass, goals-added, stadia | American Soccer Analysis (`itscalledsoccer` Python pkg) | Free, no key |
| Contract prices, order book, historical candlesticks | Kalshi API (public, no key) | Free |
| Injuries / suspensions / availability | Scraped (MLS report, Transfermarkt, Rotowire) | Free, fragile |

### 3.2 Game-resolution pipeline

Given a Kalshi market, resolve it to a set of players. This is fundamentally a
name-matching problem — the fiddly part of the project.

- **3.2.1** Parse the Kalshi market → extract both team names and match
  date/time from the market title/ticker.
- **3.2.2** Map Kalshi team names → ASA `team_id`. Strings won't match exactly
  ("Sporting KC" vs "Sporting Kansas City"). Build a **manual crosswalk
  dictionary** of all ~30 MLS teams once — most robust for a small fixed set,
  avoids fuzzy-match errors.
- **3.2.3** Resolve the fixture via `get_games` filtered by `team_id` + date →
  ASA `game_id`.
- **3.2.4** Get the player pool.
  - *Limitation:* ASA gives players who **appeared** in a completed game and
    season-level tables — it does **not** publish a confirmed starting XI before
    kickoff.
  - For **upcoming** games: pull each team's season squad + rolling per-player
    stats, weighted by recent minutes played.
  - For **backtesting** past games: use actual appearances from `get_games`.
- **3.2.5** Fetch per-player stats: `get_player_xgoals`, `get_player_xpass`,
  `get_player_goals_added` (split by season), joined to `get_players` for
  names/positions.
- **3.2.6** Aggregate to team-match features.

### 3.3 Kalshi API — verified (as of build)

- **Docs:** `docs.kalshi.com`. Current REST version is **v2**.
- **Base URL (production):** `https://api.elections.kalshi.com/trade-api/v2`.
  (Older material cites `trading-api.kalshi.com`; confirm against the live docs
  on build day.)
- **Market data is PUBLIC — no API key required** for prices, order books,
  market details, series, and events. Auth is only needed for trading /
  portfolio. Since this project is read-only analysis, **skip auth entirely.**
- **Endpoints that matter:**
  - `GET /series` — discover what's listed; **grep for MLS/soccer series first.**
  - `GET /events`, `GET /markets` — the tradeable contracts.
  - `GET /markets/{ticker}/orderbook` — current book.
  - **Candlesticks** (`GET /series/{series}/markets/{ticker}/candlesticks`) —
    **historical price over time per contract**; this is how you capture the
    **closing price** for honest backtesting.
- **Pricing format (changed Mar 2026):** prices are fixed-point **dollar
  strings** (e.g. `"0.6500"`); legacy integer-cent fields were removed. Use the
  `_dollars` fields. Some markets tick to $0.001.
- **Rate limits:** tiered token buckets; free Basic tier (~20 reads/sec) is
  ample for pulling MLS markets.
- **SDK:** official `pip install kalshi-python`, or just `httpx` / `requests`
  for public data.

> ⚠️ **Open question to resolve first:** confirm Kalshi currently lists
> **per-match MLS markets** (not only futures like title/playoff winners). Hit
> `GET /series` and search for soccer/MLS tickers before committing to the
> per-game strategy. If single-game markets aren't offered, fall back to
> whatever binary MLS markets exist — the favorite–longshot bias still applies.

### 3.4 Injury / availability scraping

The highest-value soccer signal ASA does **not** provide. Injury news moves the
market slowly, so it's exactly the kind of inefficiency the Kalshi strategy
wants — but it's also the hardest and most leak-prone data in the project.
Handle with discipline.

- **3.4.1 Pick a source.** Candidates, in rough order of structure:
  - **MLS official availability report** — published weekly; most authoritative.
  - **Transfermarkt** MLS injury table — structured, widely used.
  - **Rotowire** MLS injuries — good for status granularity (out / questionable).
  Start with one, get it working end-to-end, add others only if needed.
- **3.4.2 Build a polite scraper.** `requests` + `BeautifulSoup` (or `httpx`).
  Set a real user-agent, throttle (1 request / few seconds), cache raw HTML to
  `data/raw/injuries/` so re-parsing never re-hits the site. **Check the site's
  robots.txt and terms of service before scraping.**
- **3.4.3 Timestamp every snapshot.** Store *when* each injury record was
  captured. This single discipline is what makes the feature valid — see the
  leakage warning below.
- **3.4.4 Match scraped player names → ASA `player_id`.** Same crosswalk problem
  as teams, but harder (hundreds of players, accents, nicknames). Use a fuzzy
  match (`rapidfuzz`) with a manual override table for the ones it gets wrong.
  Log unmatched names for review rather than silently dropping them.
- **3.4.5 Normalize to a status enum.** Map every source's wording onto a fixed
  set: `out`, `doubtful`, `questionable`, `suspended`, `available`. Keep the
  raw string too, for auditing.
- **3.4.6 Write to an `injuries` table** keyed by `player_id`, `snapshot_date`,
  `status`, `source`.

> ⚠️ **LEAKAGE — the make-or-break rule.** An injury feature is only valid if it
> reflects what was known **before kickoff**. For each game, join injuries using
> only the **latest snapshot dated before that match's kickoff** — never today's
> table applied to a past game. Getting this wrong makes the backtest look
> brilliant and the live results collapse. Because you're building the injury
> history forward from now, **early backtests will have sparse injury data**;
> that's expected. Treat missing injury data as "no known injury," and add a
> boolean `injury_data_available` flag so the model (and you) can tell the
> difference between "nobody hurt" and "we don't know."

### 3.5 Storage & SQL layer

SQL is a core part of this project, not just a dump format. The
game-resolution step (Kalshi market → team → game → player pool → per-player
stats) is inherently relational, and SQL joins express it far more cleanly than
chained pandas merges. It also lets you re-query historical price snapshots by
`game_id` + timestamp without re-pulling from the API.

- **Engine:** **SQLite** (`mls.db`) to start — zero-config, ships with Python
  via `sqlite3`. Postgres is a clean drop-in upgrade later.
- **Interface:** **SQLAlchemy** (avoid hardcoded SQL strings scattered around).
- **Caching:** cache ASA pulls and raw scraped HTML locally. `raw/` →
  `processed/` separation on disk alongside the DB.

**Schema (normalized):**

| Table | Purpose |
|---|---|
| `teams` | ASA team_id, name, home city, stadium coords |
| `players` | ASA player_id, name, position, height/weight |
| `games` | ASA game_id, date, home/away team_id, final score |
| `player_match_stats` | per-player, per-game xG/xPass/goals-added, minutes |
| `injuries` | player_id, snapshot_date, status, source (see 3.4) |
| `kalshi_markets` | ticker, series, resolved game_id, team crosswalk |
| `kalshi_prices` | **time-series** of price per contract (for closing price) |
| `features` | one model-ready row per game (post feature-engineering) |
| `bets` | backtest log: stake, price, model_prob, EV, outcome, P&L |

The join from `kalshi_markets` → `games` → `player_match_stats` → `injuries`
(date-filtered) is the resolver's SQL core; the `features` table is what the
models read from.

---

## 4. Feature Engineering

Pre-match features only — no leakage. Grouped by theme.

### 4.1 Core form & strength
- **4.1.1** Rolling xG / xGA (last N matches), goals-added for/against.
- **4.1.2** Squad-weighted attacking and defensive xG from the resolved player
  pool (weighted by recent minutes).
- **4.1.3** Elo rating (build your own, updated per match).
- **4.1.4** xG over/under-performance — the gap between actual goals and xG;
  often more predictive of *future* results than actual goals, since it flags
  teams due to regress. High-signal and free from ASA.

### 4.2 Injury / availability (from §3.4, date-filtered)
- **4.2.1** Weighted availability loss — **fraction of the team's recent minutes
  (or recent xG / goals-added) that is unavailable.** A team missing 40% of its
  attacking output is a very different bet than one missing a backup fullback.
- **4.2.2** Split by position group — attack / midfield / defense / goalkeeper,
  so the model can weight an absent striker differently from an absent defender.
- **4.2.3** First-choice keeper out flag — goalkeeper availability is arguably
  the highest per-player swing; track it explicitly.
- **4.2.4** `injury_data_available` boolean — distinguishes "no injuries" from
  "no data" (critical while injury history is still sparse).

### 4.3 Situational
- **4.3.1** Rest-days asymmetry — the *difference* in rest between the two sides,
  not just each team's own rest.
- **4.3.2** Congestion / continental-competition flag — teams rotating for
  Concacaf Champions Cup or Leagues Cup; MLS congestion effects are real.
- **4.3.3** Matches-since-managerial-change counter — recently-changed sides
  behave unpredictably for a few matches.

### 4.4 Location & environment (you already have stadium coords)
- **4.4.1** Travel distance & time-zone change for the away side. MLS is uniquely
  cross-continental — a genuinely strong, differentiating feature.
- **4.4.2** Per-stadium home advantage — let the model learn venue-specific home
  effects (Colorado altitude, Texas heat, coast-to-coast travel burden) rather
  than one league-wide constant.
- **4.4.3** Kickoff weather — match-time temperature, wind, precipitation from
  the stadium coords. Extreme heat and heavy rain suppress scoring.

### 4.5 Market-derived
- **4.5.1** Kalshi implied probability, price bucket (favorite/longshot),
  time-to-close.
- **4.5.2** Pre-kickoff price drift — how much the Kalshi price moved in the days
  before the match (from candlestick data). Sharp late moves often reflect news
  like injuries, so this **partially proxies the injury signal even without the
  scrape** — a useful, leak-safe fallback.

---

## 5. Modeling

- **5.1** Multinomial logistic regression (`statsmodels` / `sklearn`) for
  outcome.
- **5.2** MLR (`statsmodels`, for coefficients / p-values / VIF) for goal
  differential.
- **5.3** Regularization (L1/L2), multicollinearity checks (VIF).
- **5.4** Time-series split — train on past seasons, test forward. Never
  random-shuffle across time.

---

## 6. Visualization

Visualization is a first-class deliverable here, not an afterthought — both for
model diagnostics and for reading the betting edge. All plots via
`matplotlib` / `seaborn`; each saved to `reports/figures/` and surfaced in a
summary notebook.

### 6.1 Regression diagnostics (MLR — goal differential)

The classic four-panel regression check, because MLR's validity rests on its
assumptions:

- **Residuals vs. fitted** — detect non-linearity and heteroscedasticity.
- **Q–Q plot of residuals** — check normality of errors.
- **Scale–location plot** — check constant variance.
- **Residuals vs. leverage** (with Cook's distance) — spot influential games.

Plus:

- **Coefficient plot** — each predictor's estimate with 95% confidence-interval
  whiskers, sorted by magnitude. The single most useful chart for
  *communicating* which features move goal differential.
- **Predicted vs. actual goal differential** scatter with a 45° reference line.
- **Correlation heatmap / VIF bar chart** — expose multicollinearity among
  features before trusting coefficients.

### 6.2 Logistic-regression diagnostics

- **Coefficient / odds-ratio plot** — exponentiated coefficients with CIs, so
  each feature reads as "multiplies the odds of a home win by X."
- **ROC curve + AUC** (one-vs-rest per outcome) and **confusion matrix** heatmap.
- **Reliability (calibration) curve** — predicted probability bucket vs.
  observed frequency, with the diagonal = perfect calibration. **The most
  important plot in the whole project**, because the strategy is a bet on
  calibration.

### 6.3 The edge visualizations (model vs. Kalshi)

- **Calibration overlay** — your model's reliability curve plotted *against*
  Kalshi's price-vs-outcome curve on the same axes. The vertical gap in the
  high-price (favorite) region is your theoretical edge, made visible.
- **Favorite–longshot bias chart** — contract price bucket (x) vs. realized win
  rate (y), replicating the paper's finding on your own data and showing where
  model and market diverge.
- **EV-by-price-bucket bar chart** — expected value net of fees across price
  buckets; visually confirms bets cluster on the favorite side.

### 6.4 Backtest / bankroll visualizations

- **Cumulative ROI / bankroll curve** over the held-out season — flat staking
  vs. fractional Kelly on the same axes.
- **Drawdown chart** — running peak-to-trough, so the risk is legible.
- **Bet distribution** — count and stake size by price bucket and EV threshold.

---

## 7. Calibration → Betting Evaluation → Reality Check

- **7.1** Calibrate probabilities (isotonic / Platt); verify with the
  reliability curve.
- **7.2** Compute EV net of Kalshi fees: `EV = model_prob × payout − 1`.
- **7.3** Filter to the favorite side; longshots are no-bet by default.
- **7.4** Backtest against **closing prices** (accuracy improves toward close, so
  closing price is the honest benchmark).
- **7.5** Bankroll sim: flat vs. fractional Kelly. Report ROI, max drawdown, hit
  rate by bucket.
- **7.6** Baseline to beat: simply buying every high-priced favorite. If the
  model can't beat that, the features aren't adding value beyond the known bias.
- **7.7** Reality check: the paper's own conclusion is that participants net-lose
  to fees; a published edge erodes as others exploit it. Paper-trade forward
  before risking real money.

---

## 8. Proposed Repository Layout

```
kalshi-mls/
├── data/
│   ├── raw/
│   │   └── injuries/          # cached scraped HTML
│   ├── processed/
│   └── mls.db                 # SQLite
├── reports/
│   ├── figures/               # all saved plots
│   └── summary.ipynb          # narrative + visuals
├── src/
│   ├── db.py                  # SQLAlchemy engine + schema (§3.5)
│   ├── ingest_asa.py          # ASA client wrapper, caching
│   ├── ingest_kalshi.py       # Kalshi public market data + candlesticks
│   ├── scrape_injuries.py     # injury scraper + name match + snapshots (§3.4)
│   ├── team_crosswalk.py      # Kalshi ↔ ASA name map
│   ├── resolve_game.py        # Kalshi market → ASA game_id → player pool
│   ├── features.py            # all features (§4), incl. injuries & travel
│   ├── train_logistic.py      # multinomial win/draw/away
│   ├── train_mlr.py           # goal differential
│   ├── calibrate.py
│   ├── viz.py                 # all plotting helpers (§6)
│   ├── evaluate.py            # backtest vs Kalshi closing, EV net fees
│   └── bet_sim.py
├── config.yaml
└── requirements.txt
```

---

## 9. Stack

`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `itscalledsoccer`,
`kalshi-python` (or `httpx`), `beautifulsoup4`, `rapidfuzz`, `sqlalchemy`,
`sqlite3`, `matplotlib`, `seaborn`, `PyYAML`, `jupyter`.

---

## 10. Suggested Build Order

Each phase should be working and committed before the next.

1. **Database** — `db.py`: define the SQLAlchemy schema (§3.5) first, so
   everything downstream writes into a real store.
2. **Kalshi ingest** — `ingest_kalshi.py`: **first task: `GET /series` and
   confirm MLS markets exist.** Then pull markets + candlesticks into
   `kalshi_markets` / `kalshi_prices`.
3. **ASA ingest** — `ingest_asa.py` + `team_crosswalk.py`: prove ASA access,
   populate `teams` / `players` / `games` / `player_match_stats`.
4. **Resolver** — `resolve_game.py`: the Kalshi-market → player-pool resolver
   (the SQL joins).
5. **Injury scraper** — `scrape_injuries.py`: stand up one source end-to-end
   with timestamped snapshots and name matching (§3.4). Start it early so injury
   history accumulates while you build the rest.
6. **Features** — `features.py`: core + situational + location first; plug in
   injury features (§4.2) once the scraper has run a while.
7. **MLR** — `train_mlr.py` + regression diagnostic plots (§6.1).
8. **Logistic** — `train_logistic.py` + calibration plots (§6.2).
9. **Evaluation** — `evaluate.py` + `bet_sim.py` + edge/backtest visuals
   (§6.3–6.4).

---

> **Note.** This is an educational project. The paper it builds on shows Kalshi
> participants net-lose to fees, and a published edge tends to decay as others
> exploit it. Nothing here guarantees profit; treat any real-money use as risk
> capital only.
