"""Final-subscription forecaster.

While an IPO is still open, the *final* category subscription (which drives both allotment
odds and the listing model) is unknown. Indian IPO books fill late: QIBs and NIIs bid mostly
on the last day. We model

    log(final / current)_cat ~ Normal( a_cat * (1 - f),  s0 + s1 * (1 - f) )

where f in [0,1] is the fraction of bidding time elapsed. The priors below are deliberately
conservative; `fit_from_history()` re-estimates a_cat by least squares from stored intraday
snapshots (every live refresh is stored, so the forecaster improves automatically as data
accumulates).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select

from ..config import settings
from ..db import Ipo, SubscriptionSnapshot, session_scope

log = logging.getLogger(__name__)
PARAMS_PATH = settings.models_dir / "subscription_forecaster.json"

DEFAULT_A = {"total": 2.9, "qib": 3.7, "nii": 3.0, "rii": 1.8}
S0, S1 = 0.12, 0.55
BID_OPEN_H, BID_CLOSE_H = 10, 17


def _business_days(a: dt.date, b: dt.date) -> list[dt.date]:
    days, d = [], a
    while d <= b:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days or [a]


def elapsed_fraction(open_date: dt.date | None, close_date: dt.date | None, at: dt.datetime) -> float:
    """Fraction of the bidding window (10:00-17:00 IST on weekdays) elapsed at IST time `at`."""
    if not open_date or not close_date:
        return 1.0
    days = _business_days(open_date, close_date)
    total = len(days) * (BID_CLOSE_H - BID_OPEN_H)
    done = 0.0
    for d in days:
        if at.date() > d:
            done += BID_CLOSE_H - BID_OPEN_H
        elif at.date() == d:
            h = at.hour + at.minute / 60
            done += min(max(h - BID_OPEN_H, 0), BID_CLOSE_H - BID_OPEN_H)
    return float(min(1.0, max(0.0, done / total)))


def ist_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=5, minutes=30)


@dataclass
class Forecaster:
    a: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_A))
    s0: float = S0
    s1: float = S1
    fitted_on: int = 0

    @classmethod
    def load(cls) -> "Forecaster":
        if PARAMS_PATH.exists():
            d = json.loads(PARAMS_PATH.read_text())
            return cls(d["a"], d["s0"], d["s1"], d.get("fitted_on", 0))
        return cls()

    def save(self) -> None:
        PARAMS_PATH.write_text(json.dumps({"a": self.a, "s0": self.s0, "s1": self.s1, "fitted_on": self.fitted_on}))

    def params(self, cat: str, f: float) -> tuple[float, float]:
        rem = 1.0 - f
        return self.a.get(cat, DEFAULT_A["total"]) * rem, (self.s0 + self.s1 * rem) if rem > 1e-6 else 0.0

    def quantiles(self, cat: str, current: float | None, f: float, qs=(0.1, 0.5, 0.9)) -> list[float] | None:
        if current is None:
            return None
        from scipy.stats import norm
        mu, sd = self.params(cat, f)
        base = max(current, 0.01)
        return [float(base * math.exp(mu + sd * norm.ppf(q))) for q in qs]

    def sample(self, cat: str, current: float | None, f: float, n: int, rng: np.random.Generator,
               z: np.ndarray | None = None) -> np.ndarray | None:
        if current is None:
            return None
        mu, sd = self.params(cat, f)
        z = rng.standard_normal(n) if z is None else z
        return max(current, 0.01) * np.exp(mu + sd * z)


def fit_from_history(min_pairs: int = 30) -> Forecaster:
    """Re-estimate a_cat from (intermediate snapshot, final snapshot) pairs."""
    fc = Forecaster.load()
    pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in DEFAULT_A}
    with session_scope() as s:
        for ipo in s.scalars(select(Ipo).where(Ipo.listing_date.is_not(None))):
            snaps = list(s.scalars(select(SubscriptionSnapshot).where(
                SubscriptionSnapshot.ipo_slug == ipo.slug, SubscriptionSnapshot.source == "nse")
                .order_by(SubscriptionSnapshot.as_of)))
            if len(snaps) < 2:
                continue
            final = snaps[-1]
            if final.as_of.date() < (ipo.close_date or final.as_of.date()):
                continue
            for sn in snaps[:-1]:
                f = elapsed_fraction(ipo.open_date, ipo.close_date, sn.as_of)
                if f >= 0.98:
                    continue
                for cat, col in (("total", "total_times"), ("qib", "qib_times"), ("nii", "nii_times"), ("rii", "rii_times")):
                    c, fin = getattr(sn, col), getattr(final, col)
                    if c and fin and c > 0.01:
                        pairs[cat].append((1 - f, math.log(fin / c)))
    n_total = 0
    for cat, pts in pairs.items():
        if len(pts) >= min_pairs:
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            fc.a[cat] = float((x * y).sum() / (x * x).sum())
            n_total += len(pts)
    if n_total:
        fc.fitted_on = n_total
        fc.save()
        log.info("subscription forecaster refit on %d pairs: %s", n_total, fc.a)
    return fc
