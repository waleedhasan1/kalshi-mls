# Pricing MLS Matches: An End-to-End Data Science Case Study

**Building a full statistical modeling pipeline — from raw sports and market data to a calibrated probability model and an honest backtest — to test whether a documented pricing bias in Kalshi's sports prediction markets is real and exploitable.**

*Python · SQL · scikit-learn · statsmodels · SQLAlchemy · pandas*

---

## Why this project

I wanted a project that forced me to own every stage a production data science pipeline actually has — data acquisition and entity resolution, leak-safe feature engineering, statistical modeling with real diagnostics, calibration, and a backtest that reports its own limitations instead of hiding them. I picked a domain where I could validate results against a published academic finding rather than just eyeballing a metric: Bürgi, Deng & Whelan's 2025 paper *"Makers and Takers: The Economics of the Kalshi Prediction Market,"* which found that Kalshi's prediction-market contracts show a classic **favorite–longshot bias** — longshot contracts are systematically overpriced, favorites are systematically underpriced.

The question I set out to answer: **build an independent win-probability model for MLS matches, then check — in my own data — whether that same bias shows up, and whether a model-driven strategy can clear it net of fees.** The short answer is that the bias replicates cleanly; whether a model can trade it profitably is still an open question, honestly reported below.

This ended up touching retail-style customer/product analytics patterns (entity resolution across messy identifiers, relational data modeling), sports-betting market pricing and EV analysis, and soccer performance analytics (xG, player-level data, MLS specifically) — which is why I think it's a relevant work sample across a few different kinds of data science roles, not just one.

---

## What I built

A 12-stage pipeline, each stage reading the last one's output from a relational store, fully re-runnable end to end with no API keys required:

```
ingest (Kalshi + American Soccer Analysis + injury scraper)
  → entity resolution (market → team → game → player pool)
  → leak-safe feature engineering
  → statistical modeling (regression + classification)
  → probability calibration
  → EV backtest vs. real market prices
  → bankroll simulation
  → visualization (14 diagnostic figures)
```

**Data:** 31 MLS teams, 5,762 completed games (2013–2026), 3,556 players, and ~38.7k per-player-per-game stat rows from American Soccer Analysis; 336 Kalshi prediction-market contracts and 59,070 hourly price candles; one timestamped injury-availability snapshot, scraped and fuzzy-matched to player IDs at an 88% hit rate.

**Modeling:** a multiple linear regression on goal differential (interpretable coefficients, VIF-checked for multicollinearity) and a multinomial logistic regression on match outcome (home/draw/away) — the model that actually drives the betting decision. Both trained on a strictly chronological split; nothing is ever shuffled across time, since a sports model that peeks at the future is worthless in production.

**Calibration and backtesting:** isotonic and Platt scaling tested against raw model probabilities on a held-out calibration slice; an expected-value backtest against real Kalshi closing prices, net of an approximated trading fee; a flat-vs-fractional-Kelly bankroll simulation.

---

## Data engineering: the part that isn't glamorous but is where most of the actual work lives

Three independent data sources — Kalshi, American Soccer Analysis, and Rotowire's injury feed — each identify the same 30 MLS teams with **three different abbreviation schemes** (Kalshi: `DAL`; ASA: `FCD`; Rotowire: `DAL` again but for a different team elsewhere in the list). None of them share a common key. I hand-built and verified a crosswalk table for each pairing by cross-referencing real players against known team rosters, rather than trusting a fuzzy match to get a 30-team, low-cardinality problem right — fuzzy matching is the right tool for the hundreds of player names in the injury feed (I used `rapidfuzz` there, with unmatched names logged for review rather than silently dropped), but the wrong tool for a small fixed set where a wrong match is a silent, hard-to-detect data corruption.

The whole thing sits on a normalized SQLite schema (nine tables — teams, players, games, player-match-stats, injuries, markets, prices, engineered features, and bets) accessed through SQLAlchemy, so the "Kalshi market → resolved team → resolved game → player pool → per-player stats" join — the actual analytical core of the project — is expressed as a real relational join instead of a chain of fragile pandas merges.

### A data-integrity bug I caught, not just one I avoided

Midway through, I found that my "closing price" for each market was actually being pulled from *after* Kalshi's markets closed for settlement — days after the match ended — rather than the price right before kickoff. Because settled markets collapse toward $0.99 or $0.01 once the result is known, this would have quietly made the backtest look far more accurate than it actually was: a textbook data leakage bug, and exactly the kind that makes small-sample backtests untrustworthy if nobody checks for it. I caught it by noticing implausible price values in an exploratory query, traced it to the root cause, fixed the query to filter every price candle to strictly before kickoff, and re-ran the downstream pipeline to confirm the fix. I'd rather find this kind of thing myself than have a stakeholder find it after a model ships.

---

## Statistical modeling and diagnostics

The regression work follows classical practice, not a black-box `.fit()` call:

- **Multiple linear regression** on goal differential, with the full four-panel residual diagnostic (residuals vs. fitted, Q–Q, scale-location, Cook's distance/leverage), a standardized coefficient plot with 95% confidence intervals, and a VIF check (max VIF ≈ 2.05 — no meaningful multicollinearity).
- **Multinomial logistic regression** on match outcome, reported two ways: a `statsmodels` MNLogit for interpretable odds ratios, and an L2-regularized `scikit-learn` model for downstream calibration and backtesting. Evaluated with one-vs-rest ROC/AUC, a confusion matrix, and — the chart I'd call the most important one in the project — a reliability (calibration) curve, because the entire betting strategy is a bet on calibration, not raw accuracy.
- **Calibration, tested rather than assumed.** I fit both isotonic regression and Platt scaling on a held-out calibration slice and compared each against the raw, uncalibrated probabilities on a final, untouched holdout. At the sample size available (~178 rows), **neither calibration method actually improved the Brier score** — isotonic regression overfit the small calibration set. Rather than reporting the calibrated version because "calibration is best practice," the pipeline checks which approach actually wins and falls back to raw probabilities when neither does. That's a small thing, but it's the difference between running a method and understanding why you're running it.

Feature engineering (Elo ratings implemented from scratch, rolling xG/xG-against, travel distance and time-zone shift between stadiums, rest-day asymmetry, manager tenure, kickoff weather via a public weather API, market-derived price drift) is entirely leak-safe: every rolling or expanding statistic is shifted by one match before being attached to a game, so a feature can never see the result of the game it's predicting.

---

## The domain result: the bias replicates

Bucketing 180 real (game, contract) pairs by Kalshi's implied probability against what actually happened:

| Price bucket | Mean price | Realized win rate | n |
|---|---|---|---|
| Longshot | $0.17 | **5.6%** | 18 |
| Contested (low) | $0.28 | 27.5% | 109 |
| Contested (high) | $0.48 | 50.0% | 44 |
| Favorite | $0.66 | **77.8%** | 9 |

This is the paper's shape, in independent data I collected and processed myself: longshots lose more than their price implies, favorites win more than their price implies. Filtering to the favorite side and positive model expected value net of fees, a naive "buy every favorite, no model" baseline already shows **+3.3% ROI at a 70% hit rate** on the 10 favorite-priced contracts available. The model-driven subset (n=3) is too small to evaluate — and I say so directly in the write-up rather than reporting a flattering number from three data points. Kalshi's MLS market only has a few weeks of price history at the time of this build; the pipeline is built to be re-run as that history accumulates, not as a one-shot result.

---

## Skills this project demonstrates

| Area | Where it shows up |
|---|---|
| Production-style ML pipeline design | 12 modular, re-runnable stages; SQL-backed state instead of notebook-only state |
| Statistical modeling & inference | OLS + multinomial logistic regression, VIF, confidence intervals, odds ratios, calibration |
| Python / scientific stack | pandas, NumPy, scikit-learn, statsmodels, SciPy |
| SQL & relational data modeling | 9-table normalized schema, SQLAlchemy, entity-resolution joins across 3 data sources |
| Data integrity & auditing | Caught and fixed a real leakage bug; logs unmatched entity-resolution cases instead of silently dropping them |
| Communicating to non-technical stakeholders | 14 annotated diagnostic figures, a narrated executed notebook, this write-up |
| Statistical maturity / avoiding overclaiming | Explicit small-sample caveats, baseline comparisons, honest negative calibration result |
| Sports betting / market pricing domain knowledge | EV-net-of-fees modeling, favorite–longshot bias analysis, Kelly-criterion bankroll simulation |
| Soccer analytics domain knowledge | xG/xG-against, goals-added, MLS-specific team and player data, direct MLS market experience |

**Stack:** Python, pandas, NumPy, scikit-learn, statsmodels, SQLAlchemy, SQLite, matplotlib, seaborn, rapidfuzz, httpx.

---

*Full write-up with all 14 figures: [interactive report](https://claude.ai/code/artifact/59e8f1fc-2809-4bac-aadf-2645ba4ed12a). Source: private GitHub repository, available on request.*
