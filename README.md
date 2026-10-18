# IPOPilot AI 🚀

**AI-powered IPO application & capital-allocation planner for the Indian stock market.**

Given a limited budget, several simultaneous IPOs, a number of eligible applicants (PANs) and uncertain
allotment, *how should an investor allocate applications to optimise their objective?*

IPOPilot combines **deep learning** (probabilistic listing-return forecasting, FinBERT news sentiment),
**probability modelling** (SEBI allotment lottery mathematics, demand forecasting) and **mathematical
optimization** (integer programming with OR-Tools CP-SAT + Monte-Carlo risk simulation) behind a FastAPI
backend and a Next.js web app.

> ⚠️ Educational research and decision-support tool — **not investment advice** and not a SEBI-registered
> investment adviser. GMP is unofficial; model outputs are probabilistic estimates that can be wrong.

---

## Contents
1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Architecture](#architecture)
4. [Data sources](#data-sources)
5. [The models](#the-models)
6. [The optimizer](#the-optimizer)
7. [Results](#results)
8. [API reference](#api-reference)
9. [Project layout](#project-layout)
10. [Configuration & deployment](#configuration--deployment)
11. [Limitations & honest caveats](#limitations--honest-caveats)
12. [Changelog](#changelog)

A plain-language walkthrough of every component is in **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**.

---

## What it does

| Feature | Description |
|---|---|
| **Live IPO intelligence** | Every open / upcoming mainboard & SME IPO: price band, lot, min investment, category-wise subscription (QIB / sNII / bNII / Retail), GMP from 2 sources with timestamps, auto-refresh every 30 min in market hours. |
| **Listing-return distribution** | A deep ensemble of Mixture-Density Networks predicts the full distribution of listing-day returns (mean, quantiles, P(loss)), not a single number. |
| **Allotment probability** | Explicit SEBI-rule lottery maths for Retail, sNII, bNII and the post-July-2025 SME 2-lot rule, with uncertainty; exact multi-applicant odds `E[1-(1-p)^n]`. |
| **Demand forecasting** | Forecasts *final* category subscription from intraday snapshots while an issue is open (self-refitting as data accumulates). |
| **Winner's-curse aware EV** | Demand and returns are simulated jointly — you are allotted mostly when demand (and the listing pop) is weakest. |
| **Planner (objectives A–E)** | A: max expected profit · B: max P(≥1 allotment) · C: max return on blocked capital · D: risk-adjusted (CVaR penalty) · E: compare all four. |
| **Realistic constraints** | One application per PAN per IPO, duplicate-PAN detection, day-by-day blocked-capital limits with T+1 unblocking (capital reuse), board filters, risk limits. |
| **Explanations** | Why each IPO was selected or excluded, break-even sensitivity, occlusion-based feature attribution for every prediction. |
| **What-if simulator** | Change GMP / subscription and watch the model, allotment odds and expected profit update. |
| **Research module** | Walk-forward evaluation vs. baselines + strategy backtest answering the project's research question. |

## Quick start

**Requirements:** Python 3.10+ (3.11 tested), Node 20+ (24 tested). An NVIDIA GPU is used automatically if
available (tested on RTX 4050) but is not required.

```powershell
# 1) Backend dependencies (install the CUDA build of PyTorch first if you have a GPU)
cd backend
pip install -r requirements.txt

# 2) Load data (≈ 30–60 min the first time: 3,300 IPOs, subscription + GMP history, 20y market data)
python -m ipopilot.cli bootstrap          # add --quick to skip the historical backfills
python -m ipopilot.cli news --history     # headlines + FinBERT sentiment for listed IPOs (optional)

# 3) Train & evaluate the models, then run the strategy backtest
python -m ipopilot.cli train
python -m ipopilot.cli backtest

# 4) Run the API (http://127.0.0.1:8000/docs)
python -m ipopilot.cli serve

# 5) Run the web app (http://localhost:3000)
cd ../frontend
npm install
npm run dev
```

Or on Windows simply: `.\scripts\setup.ps1` once, then `.\scripts\start.ps1`.

## Architecture

```
                 ┌──────────────── Data sources ────────────────┐
                 │ NSE API · ipos.markets (NSE/BSE/SEBI data)   │
                 │ IPOWatch & MarketLiveIPOGMP (GMP, unofficial)│
                 │ Yahoo Finance (NIFTY/VIX) · Google News RSS  │
                 └──────────────────────┬───────────────────────┘
                                        │  polite, cached, retried HTTP
                              Ingestion pipeline + validation
                    (cross-source matching by symbol → fuzzy name)
                                        │  append-only, time-stamped
                          SQLite / PostgreSQL (SQLAlchemy 2.0)
                                        │
                     Point-in-time feature engineering (no leakage)
              ┌─────────────────────────┼─────────────────────────┐
     Model 1: Deep MDN ensemble   Model 3: FinBERT        Demand forecaster
     (listing-return distribution) (news sentiment)       (final subscription)
              └─────────────────────────┼─────────────────────────┘
                     Model 2: SEBI allotment lottery mathematics
                                        │
                Joint scenario generator (Gaussian copula, winner's curse)
                                        │
                  Optimizer: OR-Tools CP-SAT  (objectives A / B / C / D / E)
                                        │
                   Monte-Carlo plan evaluation · explanations · what-if
                                        │
                FastAPI REST API  ──►  Next.js 15 web app (Recharts, Tailwind)
                       ▲
                APScheduler: live refresh (30 min), news (hourly), daily retrain inputs
```

## Data sources

| Source | Type | Used for |
|---|---|---|
| [NSE India](https://www.nseindia.com/market-data/all-upcoming-issues-ipo) — `ipo-current-issue`, `all-upcoming-issues`, `ipo-detail`, `public-past-issues` | official | open/upcoming issues, **bid lot**, price band, registrar, category-wise subscription incl. sNII/bNII |
| [BSE India](https://www.bseindia.com/publicissue.html) (via ipos.markets) | official | SME issues and BSE codes |
| [SEBI](https://www.sebi.gov.in/) (ICDR Regulations & 2025 SME amendment) | official | allotment rules encoded in Model 2 |
| [ipos.markets API](https://ipos.markets/api-docs) | aggregator of official data, free, no key | 3,298-issue history since 1989, **listing gains (labels)**, final category subscription for ~2,000 issues |
| [IPOWatch](https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/) | unofficial GMP | live GMP table + **day-wise GMP history** for ~1,000 past IPOs |
| [MarketLiveIPOGMP](https://marketliveipogmp.com/) | unofficial GMP | GMP, price, lot size (second source) |
| [Yahoo Finance](https://finance.yahoo.com/quote/%5ENSEI) | market data | NIFTY 50, India VIX, NIFTY Smallcap 250 (20 years daily) |
| [Google News RSS](https://news.google.com/) | news | headlines for FinBERT sentiment |

All observations (subscription snapshots, GMP, news) are **append-only and time-stamped**; GMP history is never
overwritten (intraday changes are kept in a trail). Scrapers are rate-limited per host (≥1 s between requests)
and cache immutable historical pages on disk. The "Niva Market Watch" source mentioned in the original brief
could not be located as a public dataset, so IPOWatch was used as the second GMP source.

## The models

### Model 1 — Deep probabilistic listing-return model (PyTorch)
* **Target:** `y = log(1 + listing gain)` for 1,950 listed IPOs (2012–2026, mainboard + SME).
* **Features (34, point-in-time at the close of bidding):** price, issue size, anchor share, final QIB/NII/Retail/total
  subscription, GMP level/max/trend/count, NIFTY 20/60-day return and volatility, India VIX, smallcap momentum, IPO-market
  "heat" (mean listing gain of the previous issues that *listed before* the decision), FinBERT sentiment, SME-rule flag,
  plus explicit missing-value indicators; board & sector embeddings.
* **Architecture:** *wide & deep* — residual MLP (3 blocks, LayerNorm, GELU, dropout) plus a linear "wide" path →
  **Mixture Density Network** head with 4 Gaussian components. **Deep ensemble** of 5 seeds; predictive distribution =
  mixture of mixtures. Variance **temperature scaling** for calibrated intervals. Trained with AdamW + cosine schedule,
  recency-weighted NLL (half-life 4 years), early stopping; hyper-parameters chosen on the validation period only.
* **Evaluation:** walk-forward — every model is retrained each quarter on all issues that closed before it.
* **Explainability:** occlusion attribution per information group (GMP, QIB demand, market regime, …).
* **Baselines:** board climatology, GMP-only, ridge regression, LightGBM quantile regression.

### Model 2 — Allotment probability (mathematical, by design)
For an oversubscribed category with subscription `T` and average application size `L` (in minimum-application units):
`P(allot) = min(1, L / T)`; when `L/T ≥ 1` the remainder is allotted proportionately. `L` gets a lognormal prior per
category (SME individual applications are fixed-size ⇒ `L = 1` exactly); if the real number of applications is known
the probability is exact. Multiple PANs: `P(≥1) = E_p[1-(1-p)^n]`, which is *lower* than the naive formula because
applications share the unknown `p`. A neural network is deliberately **not** used here: the rule is known, and public
application-level labels do not exist.

### Model 3 — FinBERT news sentiment
`ProsusAI/finbert` scores each headline (positive/negative/neutral) on GPU; headlines are also topic-tagged (earnings,
regulatory, governance, IPO demand, industry). Point-in-time aggregates feed Model 1.

### Demand forecaster
`log(final/current) ~ N(a_cat·(1-f), s0 + s1·(1-f))` where `f` = fraction of the bidding window elapsed. Priors encode
that QIB/NII books fill on the last day; parameters are re-estimated automatically from stored intraday snapshots.

## The optimizer

Binary variables `x[a,i,o]` = applicant `a` submits option `o` (category + lots) to IPO `i`, solved with
**OR-Tools CP-SAT**:

* `Σ_o x[a,i,o] ≤ 1` — one application per PAN per IPO;
* for every calendar day `d`: Σ amount of applications whose funds are blocked on `d` ≤ capital (funds are blocked
  from the bid on the closing day until allotment, T+1) — so capital is *reused* across IPOs that close on different days;
* optional per-applicant capital, max number of IPOs, board filters, risk filter on `P(loss)`.

| Objective | Formulation |
|---|---|
| A · Max expected profit | `max Σ E[profit]` |
| B · Max P(≥1 allotment) | `max Σ −log(1−p)` (EV ≥ 0 by default) |
| C · Max return on blocked capital | ratio objective solved exactly with **Dinkelbach** iterations |
| D · Risk-adjusted | `max Σ (E[profit] − λ·CVaR10 downside)`, λ from the risk preference |
| E · Compare | runs A–D and tabulates expected profit, P(≥1), P(loss), CVaR 5%, peak capital |

Each plan is then simulated jointly (4,000 scenarios): listing returns correlated across IPOs through a Gaussian copula,
allotment probabilities coupled to the demand that also drives returns, independent lottery draws per applicant.

## Results

Results are written to `data/models/metrics.json` and `data/models/backtest.json` and shown on the **Model & research**
page. See [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md#9-results) for the full tables and their interpretation.

<!-- RESULTS:START -->
**Data (as of 2026-10-18):** 3,284 IPOs in the universe · 1,950 listed issues used for training (617 mainboard,
1,333 SME, 2012–2026) · ~1,000 final category-wise subscription records · 7,875 GMP observations for 946 IPOs ·
8,967 FinBERT-scored headlines · 20 years of NIFTY / VIX / Smallcap bars.

**Listing-return forecasting — 574 out-of-sample IPOs (Apr 2025 – Oct 2026), walk-forward quarterly retraining**

| Model | MAE | RMSE | CRPS ↓ | Directional acc. | 80% coverage | AUC P(loss) ↑ | Brier ↓ |
|---|---|---|---|---|---|---|---|
| Climatology | 21.0% | 32.5% | 17.7 | 63.6% | 75.6% | 0.474 | 0.256 |
| GMP only | **16.5%** | 26.3% | 14.2 | 66.0% | 81.7% | 0.694 | 0.213 |
| Ridge | 18.4% | 24.6% | 14.4 | **67.8%** | 83.3% | 0.730 | 0.201 |
| LightGBM quantile | 17.4% | 25.2% | 13.9 | 66.6% | 65.5% | 0.721 | 0.220 |
| **Deep MDN ensemble** | 17.2% | **24.1%** | **13.8** | 67.1% | 83.4% | **0.750** | **0.192** |

The deep model is the best *probabilistic* forecaster (CRPS, RMSE, P(loss) ranking and calibration) — exactly what the
optimizer consumes — while GMP alone remains the best *median* predictor. Frozen (never-retrained) models degrade under
the 2025–26 regime shift (deep MDN CRPS 15.4), which is why the system retrains periodically.

**Strategy backtest — 74 weekly windows, 404 issues, ₹2L capital, 3 PANs, lottery simulated with actual outcomes**

| Strategy | Total E[profit] | Per window | Return on capital | Loss windows | Worst window | Sharpe-like |
|---|---|---|---|---|---|---|
| Apply to everything | ₹68,480 | ₹925 | 0.90% | 32% | −₹16,542 | 0.15 |
| Rank by GMP | ₹55,434 | ₹749 | 0.88% | 30% | −₹20,064 | 0.15 |
| Rule: subscription ≥ 10× | ₹1,18,545 | ₹1,602 | **2.35%** | **4%** | **−₹309** | 0.51 |
| Optimizer + LightGBM | ₹99,078 | ₹1,339 | 1.55% | 16% | −₹19,273 | 0.26 |
| **Optimizer + Deep MDN** | **₹1,29,833** | **₹1,755** | 2.11% | 8% | −₹7,184 | **0.55** |
| Oracle (perfect foresight) | ₹1,84,811 | ₹2,497 | 3.39% | 0% | ₹0 | 0.64 |

**Answer to the research question:** yes — the deep-distribution optimizer earned the most and had the best
risk-adjusted score: 2.3× the GMP-ranking strategy, +31% over the same optimizer fed with LightGBM, and ~70% of the
perfect-foresight oracle. The simple "subscription ≥ 10×" rule is a strong, low-risk competitor (fewer losing weeks,
higher return per rupee deployed) — an honest finding worth discussing.
<!-- RESULTS:END -->

## API reference

Interactive docs at `http://127.0.0.1:8000/docs`.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | status, model version, last ingestion |
| `GET /api/ipos?include_demo=` | active IPOs with full analytics |
| `GET /api/ipos/{slug}` | detail: distribution, explanation, GMP & subscription history, news, options |
| `POST /api/ipos/{slug}/whatif` | re-predict with `{gmp_pct, sub_total, sub_qib, sub_nii, sub_rii, final_subscription}` |
| `POST /api/plan` | build a plan: `{capital, applicants[{name,pan,capital}], objective, risk, boards, holding, …}` |
| `POST /api/allotment/calc` | stand-alone allotment calculator |
| `GET /api/history`, `/api/history/stats` | listed IPOs & yearly listing-gain statistics |
| `GET /api/model` | model card: metrics + backtest |
| `GET /api/sources` | data sources, row counts, recent ingestion runs |
| `POST /api/admin/refresh` | trigger live ingestion (`X-Admin-Key` if configured) |

## Project layout

```
backend/
  ipopilot/
    config.py               settings (env / .env)
    db.py                   SQLAlchemy schema (append-only observations)
    cli.py                  bootstrap / sync / backfill / news / train / backtest / serve
    scheduler.py            APScheduler jobs
    ingestion/              http (polite client), nse, ipos_markets, gmp, market, news, pipeline, util
    ml/                     features (point-in-time), listing_model (MDN ensemble), train, sentiment (FinBERT)
    engine/                 allotment, subscription (forecaster), optimizer (CP-SAT), service, backtest
    api/app.py              FastAPI app
  tests/test_engine.py      unit tests (allotment maths, SEBI options, optimizer constraints, parsers)
frontend/                   Next.js 15 + Tailwind v4 + Recharts
  app/                      / (live IPOs), /ipo/[slug], /planner, /calculator, /history, /model, /sources
  components/               ui primitives, charts, nav
  lib/api.ts                typed API client + formatters
data/                       SQLite DB, raw cache, trained models (git-ignored)
docs/PROJECT_GUIDE.md       full explanation of the project
scripts/                    setup.ps1, start.ps1, start.sh
docker-compose.yml          PostgreSQL + API + web
```

## Configuration & deployment

Copy `.env.example` to `.env`. Key settings: `DATABASE_URL` (PostgreSQL in production), `IPOPILOT_REFRESH_MINUTES`,
`IPOPILOT_SELL_COST_PCT`, `IPOPILOT_ADMIN_KEY`, `IPOPILOT_CORS_ORIGINS`. The frontend proxies `/api/*` to
`BACKEND_URL` (default `http://127.0.0.1:8000`), so the browser talks to a single origin.

**Docker:** `docker compose up -d --build`, then `docker compose exec api python -m ipopilot.cli bootstrap` and
`… train`. **Split hosting:** the frontend deploys to Vercel (set `BACKEND_URL`); the API to any Python host with a
persistent volume (Render, Railway, Fly.io, a VM).

**Tests:** `cd backend && python -m pytest -q`.

## Limitations & honest caveats

* **Historical lot sizes** are not in the public dataset; the backtest approximates the minimum application by
  SEBI ticket-size guidance (≈₹15k mainboard, ≈₹1.2L per SME lot-unit). Live plans use the real NSE bid lot.
* **Application counts** are not published before the basis of allotment, so `L` is a prior with uncertainty.
* **GMP** is unofficial and only available for ~half the post-2019 issues; the model learns its reliability from data.
* **Decisions vs. final subscription:** the model is trained on final subscription; live predictions integrate over
  the *forecast* of final subscription, and the backtest adds closing-day forecast noise.
* **Listing day only:** long-term holding is flagged but not modelled.
* **Regulation:** personalised advice to the public in India requires SEBI registration; a disclaimer does not change that.

## Changelog

* **v1.0.0 (2026-10-18)** — initial release: ingestion from 8 sources with backfills, deep MDN ensemble + baselines,
  FinBERT sentiment, SEBI allotment model, demand forecaster, CP-SAT optimizer with objectives A–E, Monte-Carlo
  evaluation, strategy backtest, FastAPI API with scheduler, Next.js web app (7 pages, dark mode), Docker stack, tests.
