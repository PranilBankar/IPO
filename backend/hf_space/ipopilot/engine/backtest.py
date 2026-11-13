"""Strategy backtest — the project's research question:

  Does feeding deep-learning listing-return DISTRIBUTIONS into the allocation optimizer improve
  realised, risk-adjusted outcomes versus GMP-ranking, rule-based and gradient-boosting strategies?

Protocol (out-of-sample test period only, see ml/train.py split):
  * Issues are grouped into weekly decision windows by closing date.
  * Investor: capital ₹2,00,000, three distinct PANs, moderate risk, sell on listing.
  * Decision information: model predictions + final-day subscription observed with forecast
    noise (decisions are made on the closing day, before the final numbers are known).
  * Outcome: the SEBI lottery is simulated with the ACTUAL final retail subscription and the
    ACTUAL listing return; averaged over 400 lottery draws per window.
Strategies
  apply_all      apply the minimum retail application to every issue (budget permitting)
  gmp_rank       rank by GMP, apply where GMP > 0
  rule_based     apply where final total subscription >= 10x, ranked by subscription
  lightgbm_opt   optimizer (max expected profit) fed with LightGBM quantile predictions
  deep_mdn_opt   optimizer fed with the deep ensemble's predictive distribution  (ours)
  oracle         optimizer with perfect knowledge of the listing return (upper bound)
Historical lot sizes are not in the dataset, so the minimum application is approximated by
the SEBI-guided ticket size (≈₹15k mainboard; ≈₹1.2L per SME application, 2 lots after Jul 2025).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import settings
from .allotment import allot_given, sample_L
from .optimizer import Applicant, Candidate, PlanRequest, build_plan, next_business_day

log = logging.getLogger(__name__)
CAPITAL = 200_000
N_APPLICANTS = 3
COST = settings.sell_cost_pct


def approx_lot(price: float, board: str, close: dt.date) -> int:
    if board == "sme":
        unit_value = 120_000 if close < dt.date(2025, 7, 1) else 120_000  # per lot (2 lots >= ₹2L after Jul-25)
        return max(1, math.ceil(unit_value / price))
    return max(1, math.ceil(15_000 / price))


def _samples_from_quantiles(q10, q50, q90, mean, S, rng):
    """Two-piece lognormal in (1+r) space matched to q10/q50/q90."""
    y10, y50, y90 = (np.log1p(max(v, -0.95)) for v in (q10, q50, q90))
    sl = max((y50 - y10) / 1.2816, 1e-3)
    sh = max((y90 - y50) / 1.2816, 1e-3)
    z = rng.standard_normal(S)
    y = np.where(z < 0, y50 + sl * z, y50 + sh * z)
    return np.expm1(y)


def run_backtest(seed: int = 3) -> dict:
    path = settings.models_dir / "test_predictions.pkl"
    if not path.exists():
        raise RuntimeError("Run `python -m ipopilot.cli train` first (creates test_predictions.pkl).")
    df = pd.read_pickle(path)
    df["close_date"] = pd.to_datetime(df["close_date"]).dt.date
    df["price"] = np.exp(df["log_price"])
    df["rii_T"] = np.expm1(df["log_sub_rii"]).where(df["log_sub_rii"].notna(), np.expm1(df["log_sub_total"]))
    df = df[df["rii_T"].notna() & df["price"].notna()].copy()
    df["week"] = pd.to_datetime(df["close_date"]).dt.to_period("W").astype(str)
    rng = np.random.default_rng(seed)
    applicants = [Applicant(str(k), f"PAN {k + 1}", f"P{k}") for k in range(N_APPLICANTS)]
    S = 1500

    strategies = ["apply_all", "gmp_rank", "rule_based", "lightgbm_opt", "deep_mdn_opt", "oracle"]
    window_profit = {s: [] for s in strategies}
    window_capital = {s: [] for s in strategies}
    window_allot = {s: [] for s in strategies}

    for week, g in df.groupby("week"):
        g = g.reset_index(drop=True)
        rows = g.to_dict("records")
        for r in rows:
            r["lot"] = approx_lot(r["price"], r["board"], r["close_date"])
            r["unit"] = 2 if (r["board"] == "sme" and r["close_date"] >= dt.date(2025, 7, 1)) else 1
            r["amount"] = r["unit"] * r["lot"] * r["price"]

        def realised(selection: list[tuple[dict, int]]) -> tuple[float, float, float]:
            """selection: list of (row, n_applications). Returns mean profit, capital, P(any allot)."""
            if not selection:
                return 0.0, 0.0, 0.0
            draws = 400
            tot = np.zeros(draws)
            anyw = np.zeros(draws, dtype=bool)
            cap = 0.0
            for r, n in selection:
                L = sample_L(r["board"], "RII", draws, rng)
                p, _ = allot_given(np.full(draws, r["rii_T"]), L, 1)
                val = r["amount"]
                gain = r["gain_pct"] / 100
                prof_if = val * gain - COST / 100 * val * (1 + gain)
                for _ in range(n):
                    won = rng.random(draws) < p
                    tot += np.where(won, prof_if, 0.0)
                    anyw |= won
                cap += n * val
            return float(tot.mean()), cap, float(anyw.mean())

        def greedy(order: list[dict]) -> list[tuple[dict, int]]:
            # capital is reused across days; within a week windows overlap, so be conservative
            budget, sel = CAPITAL, []
            for r in order:
                n = min(N_APPLICANTS, int(budget // r["amount"]))
                if n > 0:
                    sel.append((r, n))
                    budget -= n * r["amount"]
            return sel

        sels = {
            "apply_all": greedy(sorted(rows, key=lambda r: r["amount"])),
            "gmp_rank": greedy(sorted([r for r in rows if r["gmp_missing"] == 0 and r["gmp_last_pct"] > 0],
                                      key=lambda r: -r["gmp_last_pct"])),
            "rule_based": greedy(sorted([r for r in rows if np.expm1(r["log_sub_total"] or 0) >= 10],
                                        key=lambda r: -(r["log_sub_total"] or 0))),
        }
        for strat, key in (("lightgbm_opt", "lightgbm"), ("deep_mdn_opt", "deep_mdn"), ("oracle", None)):
            cands = []
            for k, r in enumerate(rows):
                if key is None:
                    ret = np.full(S, r["gain_pct"] / 100)
                    pl = float(r["gain_pct"] < 0)
                    mean = r["gain_pct"] / 100
                else:
                    ret = _samples_from_quantiles(r[f"{key}_q10"], r[f"{key}_q50"], r[f"{key}_q90"], r[f"{key}_mean"], S, rng)
                    pl, mean = r[f"{key}_ploss"], r[f"{key}_mean"]
                noisy_T = r["rii_T"] * np.exp(0.25 * rng.standard_normal(S))  # closing-day forecast error
                cands.append(Candidate(
                    slug=f"{r['slug']}", name=r["slug"], board=r["board"], price=r["price"], lot=r["lot"],
                    close_date=r["close_date"], unblock_date=next_business_day(r["close_date"]),
                    ret_samples=ret, T_samples={"RII": noisy_T, "SNII": noisy_T, "BNII": noisy_T},
                    p_loss=pl, mean_ret=mean))
            req = PlanRequest(capital=CAPITAL, applicants=applicants, objective="max_ev", risk="moderate",
                              allow_nii=False, seed=seed)
            plan = build_plan(cands, req)
            by_slug = {r["slug"]: r for r in rows}
            sels[strat] = [(by_slug[p["slug"]], p["n_applications"]) for p in plan["selected"]]

        for strat in strategies:
            prof, cap, pa = realised(sels[strat])
            window_profit[strat].append(prof)
            window_capital[strat].append(cap)
            window_allot[strat].append(pa)

    summary = []
    for strat in strategies:
        p = np.array(window_profit[strat])
        c = np.array(window_capital[strat])
        active = c > 0
        summary.append({
            "strategy": strat, "windows": int(len(p)), "active_windows": int(active.sum()),
            "total_expected_profit": float(p.sum()), "mean_profit_per_window": float(p.mean()),
            "std_profit_per_window": float(p.std()),
            "sharpe_like": float(p.mean() / p.std()) if p.std() > 0 else None,
            "loss_window_share": float((p < 0).mean()),
            "worst_window": float(p.min()), "p5_window": float(np.quantile(p, 0.05)),
            "avg_capital_deployed": float(c[active].mean()) if active.any() else 0.0,
            "return_on_capital_pct": float(100 * p.sum() / c.sum()) if c.sum() else 0.0,
            "mean_p_any_allotment": float(np.mean(window_allot[strat])),
        })
    report = {"created": dt.datetime.now().isoformat(timespec="seconds"), "capital": CAPITAL,
              "applicants": N_APPLICANTS, "n_issues": int(len(df)), "summary": summary,
              "period": [str(df["close_date"].min()), str(df["close_date"].max())]}
    (settings.models_dir / "backtest.json").write_text(json.dumps(report, indent=2))
    return report
