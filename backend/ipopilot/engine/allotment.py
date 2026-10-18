"""Model 2 — Probabilistic allotment model (explicit mathematics, not a black box).

SEBI ICDR allotment rules modelled
----------------------------------
Mainboard
  * Retail Individual Investor (RII, bid <= Rs 2 lakh): if oversubscribed, the maximum number
    of RIIs get ONE minimum bid lot, chosen by lottery; any lots left after everyone gets one
    are allotted proportionately.
  * Small NII (Rs 2-10 lakh) and big NII (> Rs 10 lakh) (SEBI circular, Apr 2022): allotment
    of the *minimum NII application size* by lottery, remainder proportionate.
SME
  * Before 1 Jul 2025: retail applies exactly one application size (lot set so it is >= Rs 1 lakh).
  * From 1 Jul 2025 (SEBI ICDR Amendment 2025 / NSE & BSE circulars): the minimum application
    is TWO lots (> Rs 2 lakh); "Individual Investors" apply for the minimum application and are
    allotted it by lottery; NII split 1/3 (2 lots .. Rs 10 lakh) and 2/3 (> Rs 10 lakh).

Probability
-----------
For a category with subscription T (times), let the minimum application be one "unit" and let
L = the average application size in units (L >= 1). Then
    available units  A = offered / unit,  applications N = T*A / L
    P(allot) = min(1, A/N) = min(1, L / T)
and, when L/T >= 1 (everyone gets a unit and lots remain), an application of u units receives
    1 + (u-1) * (L/T - 1)/(L - 1)  units in expectation.
L is not published until the basis of allotment, so it is a random variable with a prior per
category (lognormal; SME individual applications are fixed-size so L = 1 exactly). If the
actual number of applications is known, it is used instead and the uncertainty collapses.

Correlation across applications: all of a family's applications to one IPO share the same
(unknown) p, so P(at least one of n) = E_p[1 - (1-p)^n], which is LOWER than the naive
1-(1-E[p])^n. The Monte-Carlo below handles this exactly.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

SME_RULE_CHANGE = dt.date(2025, 7, 1)
RII_LIMIT = 200_000
SNII_LIMIT = 1_000_000

# prior for L (average application size in minimum-application units): (median, log-sd)
L_PRIORS = {
    ("mainboard", "RII"): (1.7, 0.35),
    ("mainboard", "SNII"): (1.8, 0.40),
    ("mainboard", "BNII"): (3.0, 0.60),
    ("sme", "RII"): (1.0, 0.0),   # fixed application size (individual investors)
    ("sme", "SNII"): (1.6, 0.40),
    ("sme", "BNII"): (2.5, 0.60),
}

CATEGORY_LABEL = {"RII": "Retail / Individual", "SNII": "Small NII (₹2–10L)", "BNII": "Big NII (>₹10L)"}


@dataclass(frozen=True)
class AppOption:
    category: str          # RII | SNII | BNII
    lots: int              # lots applied
    unit_lots: int         # lots in the minimum application of this category
    amount: float          # rupees blocked
    label: str


def application_options(board: str, price: float, lot: int, close_date: dt.date | None,
                        budget_cap: float | None = None) -> list[AppOption]:
    """Valid application choices for ONE applicant (one PAN) in one IPO."""
    if not price or not lot:
        return []
    lot_value = price * lot
    opts: list[AppOption] = []
    sme_new = board == "sme" and (close_date is None or close_date >= SME_RULE_CHANGE)
    if board == "sme":
        unit = 2 if sme_new else 1
        opts.append(AppOption("RII", unit, unit, unit * lot_value,
                              f"Individual – {unit} lot{'s' if unit > 1 else ''} (minimum application)"))
        snii_min = max(unit + 1, math.floor(RII_LIMIT / lot_value) + 1) if sme_new else max(2, math.floor(RII_LIMIT / lot_value) + 1)
    else:
        max_rii = max(1, math.floor(RII_LIMIT / lot_value))
        opts.append(AppOption("RII", 1, 1, lot_value, "Retail – 1 lot (best odds per rupee)"))
        if max_rii > 1:
            opts.append(AppOption("RII", max_rii, 1, max_rii * lot_value, f"Retail – {max_rii} lots (max ₹2L)"))
        snii_min = math.floor(RII_LIMIT / lot_value) + 1
    snii_max = max(snii_min, math.floor(SNII_LIMIT / lot_value))
    opts.append(AppOption("SNII", snii_min, snii_min, snii_min * lot_value, f"Small NII – {snii_min} lots (min sNII)"))
    bnii_min = math.floor(SNII_LIMIT / lot_value) + 1
    opts.append(AppOption("BNII", bnii_min, bnii_min, bnii_min * lot_value, f"Big NII – {bnii_min} lots (min bNII)"))
    if budget_cap is not None:
        opts = [o for o in opts if o.amount <= budget_cap]
    # de-duplicate equal-size options (tiny lots)
    seen, out = set(), []
    for o in opts:
        if (o.category, o.lots) not in seen:
            seen.add((o.category, o.lots))
            out.append(o)
    return out


def sample_L(board: str, category: str, n: int, rng: np.random.Generator) -> np.ndarray:
    med, sd = L_PRIORS.get((board, category), (1.7, 0.4))
    if sd == 0:
        return np.full(n, med)
    return np.maximum(1.0, med * np.exp(sd * rng.standard_normal(n)))


def allot_given(T: np.ndarray, L: np.ndarray, units_applied: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over scenarios. Returns (P(at least one unit), expected units if allotted).

    T <= 1  : full allotment of every unit applied
    L/T < 1 : lottery, winners receive exactly one unit
    L/T >= 1: every applicant gets one unit + proportional share of the rest
    """
    T = np.maximum(T, 1e-6)
    ratio = L / T
    p = np.where(T <= 1, 1.0, np.minimum(1.0, ratio))
    extra_share = np.where(L > 1, np.clip((ratio - 1) / np.maximum(L - 1, 1e-9), 0, 1), 1.0)
    units_if = np.where(T <= 1, units_applied,
                        np.where(ratio >= 1, 1 + (units_applied - 1) * extra_share, 1.0))
    return p, units_if


def p_allot_point(board: str, category: str, times: float | None, applications: float | None = None,
                  offered_shares: float | None = None, unit_shares: float | None = None) -> float | None:
    """Point estimate (median L) — for display tables."""
    if times is None:
        return None
    if applications and offered_shares and unit_shares:
        return float(min(1.0, (offered_shares / unit_shares) / applications))
    med, _ = L_PRIORS.get((board, category), (1.7, 0.4))
    return float(1.0 if times <= 1 else min(1.0, med / times))


def prob_at_least_one(p_samples: np.ndarray, n: int) -> float:
    """Exact under shared-p uncertainty: E[1-(1-p)^n]."""
    return float(np.mean(1 - (1 - p_samples) ** n))


def naive_at_least_one(p: float, n: int) -> float:
    return 1 - (1 - p) ** n
