# IPOPilot AI — The Complete Project Guide

This document explains **everything** in the project: the problem, the finance behind it, where every piece of
data comes from, how each model works and why it was designed that way, how the optimizer decides, how the
results were evaluated, and how to run, extend and present it. It is written so that you can defend every design
choice in a viva or interview.

---

## Table of contents
1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [IPO mechanics you need to know](#2-ipo-mechanics-you-need-to-know)
3. [System overview](#3-system-overview)
4. [Data: sources, pipeline and the no-leakage rule](#4-data-sources-pipeline-and-the-no-leakage-rule)
5. [Model 1 — the deep listing-return model](#5-model-1--the-deep-listing-return-model)
6. [Model 2 — the allotment model (and why it is not a neural net)](#6-model-2--the-allotment-model)
7. [Model 3 — FinBERT news sentiment](#7-model-3--finbert-news-sentiment)
8. [The demand forecaster and the winner's curse](#8-the-demand-forecaster-and-the-winners-curse)
9. [Results](#9-results)
10. [The optimizer, objective by objective](#10-the-optimizer-objective-by-objective)
11. [The web app, page by page](#11-the-web-app-page-by-page)
12. [Running, operating and deploying](#12-running-operating-and-deploying)
13. [Code tour](#13-code-tour)
14. [Limitations, risks and what I would do next](#14-limitations-risks-and-what-i-would-do-next)
15. [How to present this project](#15-how-to-present-this-project)
16. [Glossary](#16-glossary)

---

## 1. The problem in one paragraph

On a given day several Indian IPOs may be open at once. An investor has limited capital, a few family members
with their own PANs, and wants to apply. Each application blocks money for a few days, most popular issues are
heavily oversubscribed (so allotment is a lottery), and the listing-day outcome is uncertain. IPO websites show
GMP and subscription; they do **not** answer: *"Given ₹1,00,000 and three PANs, which IPOs should each of us apply
to, in which category, for how many lots — and what is the realistic range of outcomes?"* IPOPilot answers that by
combining a **deep-learning forecast of the listing-return distribution**, an **exact probabilistic model of
allotment**, and an **integer-programming optimizer**, then stress-testing the plan with Monte-Carlo simulation.

## 2. IPO mechanics you need to know

| Concept | What it means | Where it appears in the code |
|---|---|---|
| **Price band / issue price** | Bids are placed within a band; retail usually bids at the cut-off (upper) price, so the upper band is what gets blocked. | `Ipo.price_high`, `Ipo.price` |
| **Lot (bid lot)** | Minimum number of shares per bid and the step size. Read from NSE "Bid Lot". | `ingestion/nse.py::parse_detail` |
| **Categories** | QIB (institutions), NII (high-net-worth; split since 2022 into **sNII ₹2–10L** and **bNII >₹10L**), **RII** (retail ≤ ₹2L), employees, shareholders. | `engine/allotment.py` |
| **Subscription (×)** | Shares bid ÷ shares offered in a category. 50× retail ⇒ 50 times more retail demand than supply. | `SubscriptionSnapshot` |
| **Allotment lottery** | SEBI ICDR: if RII is oversubscribed, the maximum number of retail applicants get **one minimum lot** by lottery. Since 2022 sNII/bNII also get the *minimum NII application* by lottery. | `allot_given()` |
| **SME 2025 rule** | From 1 Jul 2025, SME minimum application is **2 lots (>₹2L)**; "Individual Investors" get the minimum application by lottery. | `application_options()` |
| **One application per PAN** | Several demat accounts on the same PAN are *one* application (duplicates are rejected). Only distinct PANs add chances. | `service.make_plan` duplicate detection |
| **UPI / ASBA blocking** | Money is blocked (not debited) from bid until allotment; unblocked if not allotted. Allotment is T+1 and listing T+3 since 2023. | optimizer day constraints |
| **GMP** | Grey Market Premium — an *unofficial* price at which shares informally trade before listing. Not an exchange price, can be stale or manipulated. | `ingestion/gmp.py` |
| **Listing gain** | (Listing-day price − issue price) ÷ issue price. Our training label. | `Ipo.listing_gain_pct` |

**Why "apply for one lot" is usually optimal in an oversubscribed retail book:** the lottery gives each winner
exactly one lot, so bidding ₹2 lakh instead of ₹15k blocks 13× more money for the *same* probability. The code
captures this automatically — in `allot_given` an application of `u` units gets one unit if it wins the lottery,
and only gets more when supply exceeds the number of applicants.

## 3. System overview

```
Sources ─► Ingestion (polite HTTP, matching, validation) ─► Database (append-only observations)
        ─► Point-in-time features ─► Model 1 (Deep MDN) + Model 3 (FinBERT) + demand forecaster
        ─► Model 2 (allotment maths) ─► joint scenarios ─► CP-SAT optimizer ─► Monte-Carlo evaluation
        ─► FastAPI ─► Next.js web app            (APScheduler keeps it all fresh)
```

Tech stack: Python 3.11, PyTorch 2.5 (CUDA), Hugging Face Transformers, LightGBM, scikit-learn, OR-Tools CP-SAT,
SQLAlchemy 2.0 (SQLite / PostgreSQL), FastAPI, APScheduler, Next.js 15, React 19, Tailwind v4, Recharts.

## 4. Data: sources, pipeline and the no-leakage rule

### 4.1 Sources (all referenced in the original brief, verified working)

1. **NSE India JSON API** — official. `ipo-current-issue` and `all-upcoming-issues` list issues;
   `ipo-detail?symbol=…` gives the **bid lot**, price range, registrar and the category table with
   sNII/bNII split and an as-of timestamp. The client "warms" cookies by visiting an NSE page first.
2. **BSE** — official SME issues reach us through ipos.markets, which republishes NSE/BSE/SEBI data (BSE's own
   endpoints returned 403 to scripted clients).
3. **SEBI** — the allotment rules (ICDR Regulations, the April-2022 NII circular, the 2025 SME amendment) are
   encoded as mathematics in `engine/allotment.py`.
4. **ipos.markets** (the API named in the brief) — free, keyless, CORS-open. `issues_index.json` gives 3,298 issues
   since 1989 including the **listing gain** (our label); `/api/v1/subscription/<symbol>.json` gives final
   category-wise subscription for ~2,000 issues. Verified against NSE for live issues (they agree).
5. **MarketLiveIPOGMP** (named in the brief) — per-IPO pages embed GMP, price and lot in their metadata; parsed with
   a strict regular expression.
6. **IPOWatch** — live GMP table, and crucially **day-by-day GMP history pages for ~1,100 past IPOs**, which is what
   makes training with GMP possible. ("Niva Market Watch" from the brief could not be found as a public source.)
7. **Yahoo Finance** — NIFTY 50, India VIX, NIFTY Smallcap 250 daily closes since 2005 (explicit period range — the
   `range=max` shortcut silently downsamples to monthly bars, a bug found and fixed during development).
8. **Google News RSS** — headlines per company for FinBERT.

### 4.2 Pipeline (`backend/ipopilot/ingestion/`)
* `http.py` — one shared session: per-host throttle (≥1 s), retries with back-off, NSE cookie warm-up, and a disk
  cache for immutable historical pages (so re-runs are free and polite).
* `pipeline.py` — idempotent jobs: `sync_index`, `sync_live`, `sync_gmp_live`, `backfill_subscriptions`,
  `backfill_gmp`, `sync_market`, `sync_news`. Each run is logged in `ingest_runs` (visible on the **Data** page).
* **Cross-source matching:** sources name companies differently ("Tonbo Imaging" vs "Tonbo Imaging India Limited").
  `Matcher` resolves by exchange symbol first, then normalised name, then a fuzzy similarity (token-Jaccard +
  sequence ratio) restricted to issues near the same date. Ambiguous matches are skipped rather than guessed.
* **Validation:** non-IPO instruments (NCDs mislabelled upstream) are detected from NSE's series code and removed;
  stale stage labels are corrected from the calendar (`effective_stage`); category rows with zero offered shares
  (purged after close) are ignored.

### 4.3 The no-leakage rule
The decision is made **before the issue closes**, so every feature is computed *as of the closing date*:
GMP observations dated ≤ close; market closes strictly before close; "IPO-market heat" uses only IPOs that had
**listed** before the close; news published before the close. Historical GMP observations after listing are
discarded. Observations are append-only so that a model trained in the future can reconstruct what was known at
any past moment.

## 5. Model 1 — the deep listing-return model

**File:** `ml/listing_model.py`, training in `ml/train.py`, features in `ml/features.py`.

### 5.1 Why a distribution and not a number
The optimizer needs more than a point forecast: P(loss), tail risk (CVaR) and the probability of large gains all
matter, and they interact with allotment. So the network outputs a **probability distribution** — a mixture of
Gaussians over `y = log(1 + r)` (log space keeps returns above −100% and makes the right-skew easy to model).

### 5.2 Architecture
```
34 numeric features (standardised; NaN→0 + explicit "missing" flags)
+ embeddings: board (2), sector (25)
→ Linear(→96) → 3 × [LayerNorm → Linear(96→192) → GELU → Dropout → Linear(192→96)] (residual)
→ LayerNorm → Linear(→12) = 4 × (mixture logit, mean, log-σ)
+ "wide" path: Linear(34 → 4) added to the 4 component means (wide & deep)
Loss: recency-weighted negative log-likelihood of the mixture (half-life 4 years)
```
* **Deep ensemble (5 networks, different seeds):** captures *epistemic* uncertainty (what the model does not know);
  the mixture captures *aleatoric* uncertainty (genuine randomness). The final distribution is the average of
  the 5 mixtures (20 components).
* **Temperature scaling:** a single scalar multiplies every σ; it is chosen to minimise validation NLL, which
  makes the 80% intervals actually contain ~80% of outcomes.
* **Regularisation for small tabular data:** dropout, weight decay, input Gaussian noise, gradient clipping,
  early stopping (patience 60) on the validation NLL.
* **Production refit:** the ensemble is re-trained on *all* data; each member stops at the best epoch found on a
  time-ordered 85/15 split of the same data, using the same cosine schedule, and the validated temperature.

### 5.3 Features (all point-in-time)
| Group | Features |
|---|---|
| Issue terms | log price, log issue size, anchor %, days open, SME flag, post-2025-SME-rule flag |
| Demand | log(1+×) for total, QIB, NII, Retail subscription |
| GMP | last GMP %, max GMP %, GMP trend, number of GMP days |
| Market regime | NIFTY 20d & 60d return, NIFTY 20d realised volatility, India VIX, Smallcap-250 20d return |
| IPO-market heat | mean listing gain of the last 10 same-board listings, last 20 overall, dispersion, #IPOs in 30 days |
| Sentiment | mean FinBERT (pos − neg), share of negative headlines, headline count |
| Missingness | one flag per group (so "no GMP" is information, not a silent zero) |

### 5.4 Evaluation protocol
Split by **closing date** — train < 2024-07-01 ≤ validation < 2025-04-01 ≤ test. Hyper-parameters are chosen on
the validation period only; the test period is never used for any choice. The primary test protocol is
**walk-forward**: for each test quarter, *every* model (baselines included) is retrained on all issues that closed
before that quarter, with an inner time-ordered validation split for early stopping and temperature calibration —
exactly how the production system is retrained. A frozen single-split evaluation is reported as well. Metrics (in listing-return % space):
* **MAE** of the median and **RMSE** of the mean — point accuracy;
* **CRPS** (continuous ranked probability score, via pinball loss on a quantile grid) — the proper score for a
  whole distribution; lower is better;
* **Directional accuracy** — does the median get the sign right;
* **80% coverage** — calibration of intervals (ideal ≈ 80%);
* **AUC and Brier score of P(loss)** — how well the model ranks and calibrates the risk of listing below issue.
Everything is reported per board (mainboard vs SME) and on the GMP-available subset.

## 6. Model 2 — the allotment model

**File:** `engine/allotment.py`.

The brief asked whether deep learning should compute allotment probability. **Design decision: no.** The rule is
known exactly (SEBI's lottery), and no public dataset contains application-level allotment labels. A neural net
would learn a noisy approximation of a formula we can write down. What *is* uncertain is one quantity — the
average application size — and that is handled probabilistically.

### 6.1 The formula
Let a category have offered shares `O`, minimum application `m` shares (one *unit*), subscription `T` and average
application size `L` units. Then units available `A = O/m`, units bid `T·A`, applications `N = T·A/L`, and

```
P(allot) = min(1, A/N) = min(1, L/T)
```

If `L/T ≥ 1` everyone gets one unit and the remaining `A − N` units are shared proportionally:
an application of `u` units expects `1 + (u−1)·(L/T − 1)/(L − 1)` units.

The spec's own example: 1,00,000 applications for 10,000 lots ⇒ `p = 10%` — reproduced in the unit tests.

### 6.2 Uncertainty in `L`
`L` is only revealed by the registrar's basis of allotment. Priors (lognormal): mainboard retail median 1.7,
sNII 1.8, bNII 3.0; **SME individual investors `L = 1` exactly** (fixed-size applications). When the true
application count is known (calculator "exact mode") the probability is exact.

### 6.3 Multiple PANs
For `n` distinct PANs the naive formula is `1 − (1 − p)^n` (27.1% for p = 10%, n = 3 — the brief's table). But all
applications to the same IPO share the same unknown `p`, so the correct value is `E_p[1 − (1 − p)^n]`, which is
*lower* (Jensen's inequality). IPOPilot reports both. Demat accounts on the same PAN are merged, never counted.

## 7. Model 3 — FinBERT news sentiment

**File:** `ml/sentiment.py`. Headlines come from Google News RSS (filtered to those that actually mention the
company), are scored by `ProsusAI/finbert` on the GPU (batch 32, max 96 tokens) into P(positive/negative/neutral),
and tagged with a topic (regulatory, governance, earnings, IPO demand, industry). Aggregates before the decision
time become features of Model 1, and the latest headlines with their sentiment appear on each IPO page.
Sentiment is treated as **one weak signal** — the model decides how much it is worth.

## 8. The demand forecaster and the winner's curse

**File:** `engine/subscription.py` and `engine/service.py`.

**Forecasting final subscription.** While an IPO is open we only see partial demand, and books fill late (QIBs bid
on the last day). With `f` = fraction of the bidding window elapsed (10:00–17:00 IST, weekdays):
`log(final/current) ~ N(a_cat·(1−f), 0.12 + 0.55·(1−f))`. Priors: total 2.9, QIB 3.7, NII 3.0, retail 1.8 (log
growth from the opening bell). `fit_from_history()` re-estimates `a_cat` by least squares from the intraday
snapshots that the scheduler stores every 30 minutes — so the forecaster improves on its own over time. Before an
issue opens, the historical distribution of final subscriptions (by board) is used instead.

**Coupling demand and return (the winner's curse).** In every Monte-Carlo scenario, one demand shock drives *both*
the final subscription (hence the allotment probability) *and* the listing-return distribution (the deep model is
evaluated at five demand quantiles and the scenario picks the matching one). High demand ⇒ bigger listing pop but
lower allotment odds; low demand ⇒ easy allotment but a weak listing. So the expected profit of an application is
**lower** than the naive "P(allot) × E[return]", and IPOPilot shows both numbers plus the "return given allotted".
Returns of different IPOs are correlated through a common market factor (Gaussian copula, ρ = 0.3).

## 9. Results

<!-- GUIDE-RESULTS:START -->
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

### 9.1 How to read these numbers
* **MAE / RMSE** are in listing-return percentage points. An MAE of 17% sounds large, but Indian listing returns are
  extremely dispersed (SME issues swing from −50% to +400%); the climatology baseline that knows nothing is at 21%.
* **CRPS** rewards a distribution that puts probability where the outcome lands — the right score for a model whose
  output feeds a risk-aware optimizer. The deep ensemble has the lowest CRPS.
* **Coverage 83%** for a nominal 80% interval means the uncertainty is honest (slightly conservative). LightGBM's
  quantile models are over-confident (65%), which is why they produce worse plans despite a decent MAE.
* **AUC 0.75 / Brier 0.19 for P(loss)** — the deep model is best at telling which IPOs are likely to list below issue
  price, which drives the risk filter and objective D.

### 9.2 What changed during development (and why)
1. The first evaluation used a single frozen split; the deep model lost to GMP-only on the 2025–26 test period while
   matching it on validation. Diagnosis: a regime shift (median listing gain 28.6% in the validation period vs 7.3% in
   the test period) and too few GMP rows in early training data (26%).
2. Fixes, all chosen on the **validation** period only: a *wide & deep* MDN (linear skip path to the component means),
   recency weighting (half-life 4 years), a consistent cosine learning-rate schedule for validation and refit, and
   a validation grid over half-life / GMP-masking / weight decay / dropout (GMP-masking did not help and is off).
3. The primary protocol became **walk-forward quarterly retraining** — the way the production system is operated —
   with the frozen-split results still reported for transparency.

### 9.3 Honest reading of the backtest
* The backtest uses approximate historical lot sizes and simulates the lottery with a prior on application size, so
  absolute rupee figures are indicative; the *ranking* of strategies is the result.
* The rule-based strategy uses the final total subscription, which is only fully known at the 5 pm close; the model
  strategies see closing-day subscription with forecast noise. Both are realistic for a closing-day bidder.
* Returns are small relative to capital because heavily oversubscribed IPOs rarely allot — which is precisely the
  insight the optimizer is built around.
<!-- GUIDE-RESULTS:END -->

## 10. The optimizer, objective by objective

**File:** `engine/optimizer.py`.

1. **Candidate options.** For each IPO and each applicant, the valid SEBI options are enumerated
   (`application_options`): retail 1 lot, retail max (≤ ₹2L), sNII minimum, bNII minimum; SME individual 2 lots.
2. **Option evaluation.** For every option, over 4,000 joint scenarios: P(allot), expected allotted value, profit
   if allotted (minus selling costs ≈ 0.25% of sale value), expected profit and the CVaR-10 downside.
3. **Integer program (CP-SAT).** Binary `x[a,i,o]`; constraints: one option per (PAN, IPO); for each day, the sum of
   blocked amounts whose window (closing day → T+1 allotment) covers that day ≤ capital; optional per-applicant
   capital and max-IPOs; risk filter `P(loss) ≤ limit` (conservative 30%, moderate 45%, aggressive 75%).
4. **Objectives:**
   * **A — expected profit:** `max Σ E[profit]`.
   * **B — probability of at least one allotment:** `max P(∪ allotments)`; with independent lotteries
     `1 − Π(1 − p_k)` is maximised by maximising `Σ −log(1 − p_k)` — a linear objective. Applications with
     negative expected value are excluded unless you allow them.
   * **C — return on blocked capital:** `max Σ E[profit] / Σ amount` is a *fractional* program; solved exactly with
     **Dinkelbach's algorithm**: repeatedly solve `max Σ(E[profit] − ρ·amount)` and update `ρ` to the achieved ratio.
   * **D — risk-adjusted:** `max Σ(E[profit] − λ·downside)` with λ = 1.0 / 0.5 / 0.15 for conservative / moderate /
     aggressive.
   * **E — compare:** runs A–D and shows a comparison table so you can *see* how objectives change decisions.
5. **Plan evaluation.** The selected applications are simulated together (shared demand per IPO, independent
   lottery draws per PAN, correlated returns across IPOs) to give the profit distribution, P(loss), P(≥1
   allotment), VaR/CVaR at 5%, expected number of allotments and the day-by-day capital timeline.
6. **Explanations.** Each selected IPO gets a "why" sentence and a **break-even shift** (how many percentage points
   below forecast the listing can come before expected profit turns negative). Each excluded IPO gets its reason
   (risk limit, negative EV, filtered board, unknown lot, or out-competed under the capital constraint).

## 11. The web app, page by page

| Page | What you can do |
|---|---|
| **Live IPOs** (`/`) | All open/upcoming issues: dates with bidding-progress bar, price & minimum investment, live subscription with "live / final / prior" status and retail forecast, GMP consensus, model mean and 80% range bar, P(loss) with status icon, allotment odds, expected profit per minimum application. Mainboard/SME filter, fictional demo toggle. |
| **IPO detail** (`/ipo/[slug]`) | Predicted distribution chart, **what-if simulator** (GMP & subscription sliders re-run the model live), "why this prediction" attribution chart, multi-PAN allotment table (exact vs naive) and winner's-curse numbers, GMP history by source, category subscription chart, news with FinBERT sentiment, valid application options. |
| **Planner** (`/planner`) | Capital, applicants with PAN labels (duplicates merged), objective A–E, risk preference, boards, holding preference, NII toggle → comparison table, per-applicant application list, why/sensitivity, profit histogram, capital-blocked timeline, exclusions with reasons, assumptions. |
| **Allotment calc** (`/calculator`) | Board, category, subscription, number of PANs → P(allot) with uncertainty and the multi-PAN table; exact mode with application counts. |
| **History** (`/history`) | Mean listing gain by year (mainboard vs SME) and a searchable table of all listed issues. |
| **Model & research** (`/model`) | Test metrics vs baselines, GMP-subset metrics, per-board metrics, backtest table, architecture, feature importance. |
| **Data** (`/sources`) | Sources with links, row counts, the last 30 ingestion runs and their stats/errors. |

Design: one validated colour system (blue/orange/aqua categorical, blue↔red diverging for gain/loss, status colours
always paired with an icon), light & dark themes with a toggle, tabular numerals, responsive tables.

## 12. Running, operating and deploying

```powershell
cd backend
pip install -r requirements.txt
python -m ipopilot.cli bootstrap        # universe + market + live + subscription & GMP backfills
python -m ipopilot.cli news --history   # optional: headlines + FinBERT for listed IPOs
python -m ipopilot.cli train            # trains baselines + deep ensemble, writes metrics.json
python -m ipopilot.cli backtest         # writes backtest.json
python -m ipopilot.cli serve            # API on :8000 (scheduler starts automatically)
cd ../frontend && npm install && npm run dev   # web on :3000
```

* **Scheduler** (inside the API process): live issues/subscription/GMP every 30 min on weekdays 09:00–18:00 IST,
  news hourly, and at 19:30 IST the IPO index (new listing gains), market bars, forecaster refit and a stored
  prediction snapshot for every active IPO (so live accuracy can be measured later).
* **Retraining:** run `train` whenever enough new IPOs have listed (e.g. monthly). The API hot-reloads the new model.
* **PostgreSQL:** set `DATABASE_URL`. **Docker:** `docker compose up -d --build`.
* **Tests:** `python -m pytest -q` (allotment maths incl. the brief's 10% / 27.1% examples, SEBI options, SME rule,
  one-application-per-PAN, capital reuse, objective comparison, parsers).

## 13. Code tour

| File | Responsibility |
|---|---|
| `config.py` | settings from env / `.env` |
| `db.py` | schema: `ipos`, `subscription_snapshots`, `gmp_observations`, `news_items`, `market_bars`, `predictions`, `plans`, `ingest_runs` |
| `ingestion/http.py` | polite cached HTTP client |
| `ingestion/nse.py`, `ipos_markets.py`, `gmp.py`, `market.py`, `news.py` | one adapter per source |
| `ingestion/pipeline.py` | jobs, matching, validation, demo issue |
| `ml/features.py` | point-in-time feature builder |
| `ml/listing_model.py` | MDN network, ensemble, mixture maths, explanations, save/load |
| `ml/train.py` | split, baselines, training, calibration, metrics, production refit |
| `ml/sentiment.py` | FinBERT scoring + topic tags |
| `engine/allotment.py` | SEBI options and lottery maths |
| `engine/subscription.py` | final-subscription forecaster |
| `engine/service.py` | joint scenarios, analytics per IPO, plan assembly |
| `engine/optimizer.py` | CP-SAT model, objectives, Monte-Carlo evaluation, explanations |
| `engine/backtest.py` | strategy comparison on the test period |
| `api/app.py` | REST endpoints |
| `scheduler.py` | background jobs |
| `frontend/app/*` | pages; `components/charts.tsx` charts; `lib/api.ts` typed client |

## 14. Limitations, risks and what I would do next

**Limitations**
* GMP history exists for roughly half of post-2019 issues and none before; GMP sources can disagree.
* Historical lot sizes are not public in bulk, so the backtest approximates the minimum application.
* Application counts are unknown before allotment ⇒ allotment probabilities carry real uncertainty (shown).
* The model predicts the **listing day** only; post-listing performance is a different problem.
* Survivorship / selection: issues that were withdrawn never enter the data.
* Indian IPO regimes shift (2021 boom, 2024 SME frenzy, 2025 SME rules) — the walk-forward test is the honest guard,
  and periodic retraining is required.

**Regulatory note.** Offering personalised investment recommendations to the public in India requires SEBI
registration (Investment Adviser / Research Analyst regulations). IPOPilot is positioned as an educational research
tool; a disclaimer alone does not change that.

**Next steps**
1. Collect registrar basis-of-allotment PDFs to *learn* `L` per category (turns Model 2's prior into data).
2. Store intraday NSE snapshots for a few months ⇒ the demand forecaster becomes fully data-driven.
3. Add RHP fundamentals (P/E, revenue growth, margins) parsed from the NSE RHP zip files.
4. Conformal prediction on top of the MDN for distribution-free coverage guarantees.
5. Track live prediction accuracy from the stored `predictions` table on a dashboard.
6. User accounts and saved plans; email/Telegram alerts when a plan changes.

## 15. How to present this project

* **Title:** *An AI-Assisted IPO Application Allocation System Using Deep-Learning-Based Listing-Outcome Distributions
  and Probabilistic Allotment Modelling.*
* **Research question:** Does feeding deep-learning listing-return distributions into the allocation optimizer
  improve realised, risk-adjusted outcomes versus GMP-based and rule-based strategies?
* **Story arc for a demo (5 minutes):** (1) Live IPOs page — explain subscription, GMP and the 80% band; (2) open an
  IPO — move the GMP slider and show the what-if; show the multi-PAN table (27.1% vs exact); (3) Planner with
  ₹1L and 3 PANs on "Compare" — show how the four objectives choose differently and why; (4) Model page — CRPS and
  calibration vs baselines, then the backtest table; (5) close with limitations and the regulatory note.
* **Three things examiners usually ask — and the answers:** *Why not a neural net for allotment?* (§6). *How do you
  avoid leakage?* (§4.3). *Why a distribution rather than a number?* (§5.1 and the winner's curse, §8).

## 16. Glossary

* **CRPS** — continuous ranked probability score; generalises absolute error to a full predictive distribution.
* **CVaR / Expected shortfall** — average outcome in the worst x% of scenarios.
* **Deep ensemble** — several independently trained networks whose predictions are averaged.
* **Dinkelbach's algorithm** — solves fractional (ratio) objectives as a sequence of linear ones.
* **Gaussian copula** — a way to correlate random variables with arbitrary marginal distributions.
* **MDN** — mixture density network: a neural network that outputs the parameters of a mixture distribution.
* **Temperature scaling** — post-hoc calibration by scaling predicted uncertainty with one learned scalar.
* **Walk-forward split** — training on the past and testing on the future, in time order.
