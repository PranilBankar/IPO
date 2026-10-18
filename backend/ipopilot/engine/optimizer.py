"""Investment-plan optimizer (integer programming with OR-Tools CP-SAT) + Monte-Carlo
portfolio simulator.

Decision variables
    x[a, i, o] ∈ {0,1}  applicant a submits application option o (category + lots) to IPO i
Constraints
    Σ_o x[a,i,o] <= 1                       one application per PAN per IPO (SEBI)
    Σ amount·x over IPOs whose funds are     capital is BLOCKED (UPI/ASBA) from the bid until
      blocked on day d  <=  B   ∀ days d     allotment (T+1); freed capital is reusable
    per-applicant capital caps (optional)
    risk filters (max P(loss)), board filters, exclusions
Objectives (user selectable — or "compare" runs all four)
    max_ev               maximise Σ E[profit]
    max_allot_prob       maximise P(at least one allotment)  ≈ max Σ -log(1-p)   (EV >= 0 by default)
    capital_efficiency   maximise Σ E[profit] / Σ capital blocked   (Dinkelbach iterations)
    risk_adjusted        maximise Σ (E[profit] - λ·Downside),  Downside = -CVaR10 of the application
Evaluation
    The chosen plan is simulated jointly: listing returns are correlated across IPOs through a
    Gaussian copula (common market factor), allotment probabilities share uncertainty within
    an IPO, and lottery draws are independent across applicants.
"""
from __future__ import annotations

import datetime as dt
import math
import zlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from ortools.sat.python import cp_model

from ..config import settings
from .allotment import CATEGORY_LABEL, AppOption, allot_given, application_options, sample_L

Objective = Literal["max_ev", "max_allot_prob", "capital_efficiency", "risk_adjusted"]
OBJECTIVES: list[Objective] = ["max_ev", "max_allot_prob", "capital_efficiency", "risk_adjusted"]
OBJECTIVE_LABEL = {
    "max_ev": "A · Maximum expected profit",
    "max_allot_prob": "B · Maximum probability of ≥1 allotment",
    "capital_efficiency": "C · Maximum return on blocked capital",
    "risk_adjusted": "D · Risk-adjusted return",
}
RISK_LAMBDA = {"conservative": 1.0, "moderate": 0.5, "aggressive": 0.15}
RISK_MAX_PLOSS = {"conservative": 0.30, "moderate": 0.45, "aggressive": 0.75}


@dataclass
class Candidate:
    slug: str
    name: str
    board: str
    price: float
    lot: int
    close_date: dt.date
    unblock_date: dt.date
    ret_samples: np.ndarray                 # (S,) listing return (fraction), copula-correlated
    T_samples: dict[str, np.ndarray]        # category -> (S,) final subscription multiple
    p_loss: float
    mean_ret: float
    is_demo: bool = False
    known_applications: dict[str, float] = field(default_factory=dict)


@dataclass
class Applicant:
    id: str
    name: str
    pan_key: str
    capital: float | None = None
    allow_nii: bool = True


@dataclass
class OptionEval:
    cand: Candidate
    opt: AppOption
    ep: float            # expected profit (rupees)
    p: float             # P(allotment)
    p_s: np.ndarray      # per-scenario allotment probability
    profit_if: np.ndarray  # per-scenario profit if allotted
    downside: float      # -CVaR10 of the application's profit (>=0)
    ev_per_rupee: float
    value_if: np.ndarray  # per-scenario allotted value (rupees at issue price) if allotted


def next_business_day(d: dt.date, n: int = 1) -> dt.date:
    while n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def evaluate_option(c: Candidate, o: AppOption, rng: np.random.Generator, cost_pct: float) -> OptionEval:
    S = len(c.ret_samples)
    T = c.T_samples.get(o.category)
    if T is None:
        T = c.T_samples.get("RII")
    known = c.known_applications.get(o.category)
    if known:
        L = np.full(S, 1.0)  # exact: P = units available / applications, encoded via T_eff
    else:
        L = sample_L(c.board, o.category, S, rng)
    p_s, units_if = allot_given(T, L, o.lots / o.unit_lots)
    value_if = units_if * o.unit_lots * c.lot * c.price
    r = c.ret_samples
    profit_if = value_if * r - (cost_pct / 100.0) * value_if * (1 + r)
    ep = float(np.mean(p_s * profit_if))
    # downside: simulate the application's own profit distribution
    won = rng.random(S) < p_s
    prof = np.where(won, profit_if, 0.0)
    q = np.quantile(prof, 0.10)
    cvar = float(prof[prof <= q].mean()) if np.any(prof <= q) else float(q)
    return OptionEval(c, o, ep, float(p_s.mean()), p_s, profit_if, max(0.0, -cvar), ep / o.amount, value_if)


@dataclass
class PlanRequest:
    capital: float
    applicants: list[Applicant]
    objective: Objective = "max_ev"
    risk: str = "moderate"
    boards: tuple[str, ...] = ("mainboard", "sme")
    allow_negative_ev: bool = False
    max_p_loss: float | None = None
    max_ipos: int | None = None
    exclude: tuple[str, ...] = ()
    allow_nii: bool = True
    seed: int = 7


def _solve(evals: dict[tuple[int, int], list[OptionEval]], req: PlanRequest, coef_fn, days: list[dt.date],
           time_limit: float = 8.0) -> dict[tuple[int, int, int], int]:
    m = cp_model.CpModel()
    x: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for (a, i), opts in evals.items():
        for k, ev in enumerate(opts):
            x[(a, i, k)] = m.NewBoolVar(f"x_{a}_{i}_{k}")
        m.Add(sum(x[(a, i, k)] for k in range(len(opts))) <= 1)
    # day-by-day blocked capital (shared pool)
    B = int(req.capital)
    for d in days:
        terms = [int(ev.opt.amount) * x[(a, i, k)] for (a, i), opts in evals.items()
                 for k, ev in enumerate(opts) if ev.cand.close_date <= d <= ev.cand.unblock_date]
        if terms:
            m.Add(sum(terms) <= B)
    # per-applicant caps
    for a, ap in enumerate(req.applicants):
        if ap.capital:
            for d in days:
                terms = [int(ev.opt.amount) * x[(aa, i, k)] for (aa, i), opts in evals.items() if aa == a
                         for k, ev in enumerate(opts) if ev.cand.close_date <= d <= ev.cand.unblock_date]
                if terms:
                    m.Add(sum(terms) <= int(ap.capital))
    if req.max_ipos:
        ipo_used = {}
        for (a, i), opts in evals.items():
            ipo_used.setdefault(i, m.NewBoolVar(f"u_{i}"))
            for k in range(len(opts)):
                m.AddImplication(x[(a, i, k)], ipo_used[i])
        m.Add(sum(ipo_used.values()) <= req.max_ipos)
    obj = []
    for (a, i), opts in evals.items():
        for k, ev in enumerate(opts):
            c = coef_fn(ev)
            if c != 0:
                obj.append(int(round(c)) * x[(a, i, k)])
    m.Maximize(sum(obj) if obj else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}
    return {key: 1 for key, var in x.items() if solver.Value(var)}


def simulate_portfolio(chosen: list[tuple[int, OptionEval]], rng: np.random.Generator, capital: float) -> dict:
    if not chosen:
        return {"expected_profit": 0.0, "p_any_allotment": 0.0, "p_loss": 0.0, "histogram": [], "n_applications": 0}
    S = len(chosen[0][1].cand.ret_samples)
    total = np.zeros(S)
    any_allot = np.zeros(S, dtype=bool)
    n_allot = np.zeros(S)
    for _, ev in chosen:
        won = rng.random(S) < ev.p_s       # shared p_s within an IPO; independent lottery draws
        total += np.where(won, ev.profit_if, 0.0)
        any_allot |= won
        n_allot += won
    q05 = float(np.quantile(total, 0.05))
    hist_counts, edges = np.histogram(total, bins=40)
    return {
        "expected_profit": float(total.mean()), "median_profit": float(np.median(total)),
        "std_profit": float(total.std()), "p_loss": float((total < -1e-6).mean()),
        "p_profit": float((total > 1e-6).mean()), "p_any_allotment": float(any_allot.mean()),
        "expected_allotments": float(n_allot.mean()), "var5": q05,
        "cvar5": float(total[total <= q05].mean()) if np.any(total <= q05) else q05,
        "q95_profit": float(np.quantile(total, 0.95)),
        "histogram": [{"lo": float(edges[j]), "hi": float(edges[j + 1]), "count": int(hist_counts[j])}
                      for j in range(len(hist_counts))],
        "n_applications": len(chosen),
    }


def build_plan(cands: list[Candidate], req: PlanRequest, objective: Objective | None = None) -> dict:
    objective = objective or req.objective
    rng = np.random.default_rng(req.seed)
    max_pl = req.max_p_loss if req.max_p_loss is not None else RISK_MAX_PLOSS.get(req.risk, 0.45)
    lam = RISK_LAMBDA.get(req.risk, 0.5)

    excluded: dict[str, str] = {}
    usable: list[Candidate] = []
    for c in cands:
        if c.slug in req.exclude:
            excluded[c.slug] = "Excluded by you."
        elif c.board not in req.boards:
            excluded[c.slug] = f"{c.board.upper()} issues are filtered out by your category preference."
        elif not c.price or not c.lot:
            excluded[c.slug] = "Price band or lot size not yet published — application cost unknown."
        elif c.p_loss > max_pl:
            excluded[c.slug] = (f"Model P(listing below issue price) = {100 * c.p_loss:.0f}% exceeds your "
                                f"{req.risk} risk limit of {100 * max_pl:.0f}%.")
        else:
            usable.append(c)

    # evaluate options per (applicant, ipo); common random numbers across applicants
    evals: dict[tuple[int, int], list[OptionEval]] = {}
    option_cache: dict[tuple[str, str, int], OptionEval] = {}
    for i, c in enumerate(usable):
        cap = min(req.capital, max((ap.capital or req.capital) for ap in req.applicants))
        opts = application_options(c.board, c.price, c.lot, c.close_date, budget_cap=cap)
        for a, ap in enumerate(req.applicants):
            lst = []
            for o in opts:
                if o.category != "RII" and not (req.allow_nii and ap.allow_nii):
                    continue
                if ap.capital and o.amount > ap.capital:
                    continue
                key = (c.slug, o.category, o.lots)
                if key not in option_cache:
                    option_cache[key] = evaluate_option(c, o, np.random.default_rng(req.seed + zlib.crc32(repr(key).encode())),
                                                        settings.sell_cost_pct)
                ev = option_cache[key]
                if not req.allow_negative_ev and ev.ep <= 0:
                    continue
                lst.append(ev)
            if lst:
                evals[(a, i)] = lst
    for i, c in enumerate(usable):
        if not any(k[1] == i for k in evals):
            best = max((v for k, v in option_cache.items() if k[0] == c.slug), key=lambda e: e.ep, default=None)
            if best is None:
                excluded[c.slug] = "Even the minimum application exceeds the available capital."
            else:
                excluded[c.slug] = (f"Negative expected value after costs (best option E[profit] = ₹{best.ep:,.0f}; "
                                    f"P(allotment) {100 * best.p:.1f}%, model mean return {100 * c.mean_ret:+.1f}%).")

    days = sorted({c.close_date + dt.timedelta(days=k) for c in usable
                   for k in range((c.unblock_date - c.close_date).days + 1)})

    if objective == "max_ev":
        sol = _solve(evals, req, lambda ev: ev.ep * 10, days)
    elif objective == "max_allot_prob":
        sol = _solve(evals, req, lambda ev: 1e6 * -math.log(max(1e-9, 1 - min(ev.p, 0.999999))) + ev.ep / 50, days)
    elif objective == "risk_adjusted":
        sol = _solve(evals, req, lambda ev: (ev.ep - lam * ev.downside) * 10 if ev.ep - lam * ev.downside > 0 else 0, days)
    elif objective == "capital_efficiency":
        rho, sol = 0.0, {}
        for _ in range(12):  # Dinkelbach: max Σ(ep - rho*amount) until rho converges
            cur = _solve(evals, req, lambda ev, r=rho: (ev.ep - r * ev.opt.amount) * 1000, days, time_limit=4)
            if not cur:
                break
            ep = sum(evals[(a, i)][k].ep for (a, i, k) in cur)
            amt = sum(evals[(a, i)][k].opt.amount for (a, i, k) in cur)
            new_rho = ep / amt if amt else 0
            sol = cur
            if abs(new_rho - rho) < 1e-6:
                break
            rho = new_rho
    else:
        raise ValueError(objective)

    chosen = [(a, evals[(a, i)][k]) for (a, i, k) in sorted(sol)]
    sim = simulate_portfolio(chosen, np.random.default_rng(req.seed + 1), req.capital)

    # capital timeline (blocked per day)
    timeline = []
    for d in days:
        blocked = sum(ev.opt.amount for _, ev in chosen if ev.cand.close_date <= d <= ev.cand.unblock_date)
        timeline.append({"date": d.isoformat(), "blocked": blocked, "free": req.capital - blocked})
    peak = max((t["blocked"] for t in timeline), default=0.0)

    # per-IPO summary
    rows = []
    by_ipo: dict[str, list[tuple[int, OptionEval]]] = {}
    for a, ev in chosen:
        by_ipo.setdefault(ev.cand.slug, []).append((a, ev))
    for c in usable:
        apps = by_ipo.get(c.slug, [])
        if not apps:
            if c.slug not in excluded:
                best = max((e for e in option_cache.values() if e.cand.slug == c.slug), key=lambda e: e.ev_per_rupee, default=None)
                excluded[c.slug] = (
                    "Positive expected value but not selected: other applications score higher on the "
                    f"'{OBJECTIVE_LABEL[objective]}' objective within your capital limit"
                    + (f" (best: ₹{best.ep:,.0f} expected on ₹{best.opt.amount:,.0f} blocked)." if best else "."))
            continue
        p_samples = np.stack([ev.p_s for _, ev in apps])
        p_any = float(np.mean(1 - np.prod(1 - p_samples, axis=0)))
        ep = sum(ev.ep for _, ev in apps)
        # break-even: how many percentage points lower than predicted the listing return can be
        # before this IPO's expected profit turns negative  (EP(δ) = EP − δ·E[p·allotted value])
        mean_alloc_val = sum(float(np.mean(ev.p_s * ev.value_if)) for _, ev in apps)
        breakeven = (ep / mean_alloc_val) if mean_alloc_val else None
        rows.append({
            "slug": c.slug, "name": c.name, "board": c.board, "price": c.price, "lot": c.lot,
            "close_date": c.close_date.isoformat(), "unblock_date": c.unblock_date.isoformat(),
            "is_demo": c.is_demo, "model_mean_return_pct": 100 * c.mean_ret, "model_p_loss": c.p_loss,
            "applications": [{
                "applicant_id": req.applicants[a].id, "applicant": req.applicants[a].name,
                "category": ev.opt.category, "category_label": CATEGORY_LABEL[ev.opt.category],
                "option": ev.opt.label, "lots": ev.opt.lots, "shares": ev.opt.lots * c.lot,
                "amount": ev.opt.amount, "p_allot": ev.p, "expected_profit": ev.ep,
                "downside_cvar10": ev.downside,
            } for a, ev in apps],
            "n_applications": len(apps), "capital_blocked": sum(ev.opt.amount for _, ev in apps),
            "p_any_allotment": p_any, "expected_profit": ep,
            "breakeven_shift_pct": 100 * breakeven if breakeven is not None else None,
            "why": _why(c, apps, objective),
        })
    rows.sort(key=lambda r: -r["expected_profit"])
    names = {c.slug: c.name for c in cands}
    return {
        "objective": objective, "objective_label": OBJECTIVE_LABEL[objective], "risk": req.risk,
        "capital": req.capital, "peak_capital_blocked": peak, "min_free_cash": req.capital - peak,
        "total_applications": len(chosen), "selected": rows,
        "excluded": [{"slug": k, "name": names.get(k, k), "reason": v} for k, v in excluded.items()],
        "simulation": sim, "capital_timeline": timeline,
        "assumptions": [
            "Bids are placed on each issue's closing day; blocked funds are released on allotment day (T+1).",
            "One application per PAN per IPO; applications from the same PAN are not independent chances.",
            f"Sale on listing day at the predicted listing price; selling costs ≈ {settings.sell_cost_pct}% of sale value.",
            "Allotment follows SEBI lottery rules for RII / sNII / bNII (SME: 2-lot minimum since 1 Jul 2025).",
            "Listing returns of different IPOs are positively correlated (common market factor).",
        ],
    }


def _why(c: Candidate, apps: list[tuple[int, OptionEval]], objective: str) -> str:
    ev = apps[0][1]
    parts = [f"Model expects {100 * c.mean_ret:+.1f}% listing return (P(loss) {100 * c.p_loss:.0f}%)."]
    parts.append(f"Each {ev.opt.category} application has ≈{100 * ev.p:.1f}% allotment odds"
                 + (" — oversubscribed, so the minimum application maximises odds per rupee." if ev.p < 0.999 and ev.opt.lots == ev.opt.unit_lots else "."))
    if len(apps) > 1:
        parts.append(f"{len(apps)} distinct applicants (PANs) raise the chance of ≥1 allotment.")
    parts.append(f"Expected ₹{ev.ep:,.0f} per application on ₹{ev.opt.amount:,.0f} blocked "
                 f"({100 * ev.ev_per_rupee:.2f}% of blocked capital).")
    return " ".join(parts)


def compare_objectives(cands: list[Candidate], req: PlanRequest) -> dict:
    plans = {o: build_plan(cands, req, o) for o in OBJECTIVES}
    table = [{
        "objective": o, "label": OBJECTIVE_LABEL[o],
        "expected_profit": p["simulation"].get("expected_profit", 0), "p_any_allotment": p["simulation"].get("p_any_allotment", 0),
        "p_loss": p["simulation"].get("p_loss", 0), "cvar5": p["simulation"].get("cvar5", 0),
        "peak_capital": p["peak_capital_blocked"], "applications": p["total_applications"],
        "return_on_peak_capital": (p["simulation"].get("expected_profit", 0) / p["peak_capital_blocked"]) if p["peak_capital_blocked"] else 0,
    } for o, p in plans.items()]
    return {"comparison": table, "plans": plans}
