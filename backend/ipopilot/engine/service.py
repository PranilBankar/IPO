"""Service layer: assembles live IPO analytics, model predictions and optimizer inputs.

Joint scenario model for one IPO (S Monte-Carlo scenarios):
    demand shock  z_d  ~ N(0,1)   (partly driven by a common market factor m)
    final subscription per category  T_cat = forecast(current, elapsed, z_d)
    listing return r ~ DeepMDN( features | subscription = T(z_d) )   via a Gaussian copula on m
So returns and allotment odds are COUPLED: in high-demand scenarios the listing pop is larger
but allotment odds are lower ("winner's curse" / adverse selection for the lottery winner).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import threading
import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import GmpObservation, Ipo, NewsItem, Prediction, SubscriptionSnapshot, session_scope
from ..ml.features import FeatureBuilder, load_tables
from ..ml.listing_model import ListingEnsemble, MixtureDist
from .allotment import L_PRIORS, application_options, p_allot_point, prob_at_least_one, naive_at_least_one, sample_L, allot_given
from .optimizer import Applicant, Candidate, PlanRequest, build_plan, compare_objectives, next_business_day
from .subscription import Forecaster, elapsed_fraction, ist_now

log = logging.getLogger(__name__)

ACTIVE = ("upcoming", "rhp_filed", "open", "closed", "allotment", "pre_listing")
DEMAND_QS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
RHO_MARKET = 0.30      # copula correlation of listing returns across IPOs
RHO_DEMAND = 0.20      # share of demand shock explained by the common market factor
N_SCEN = 4000


# --------------------------------------------------------------------------- model registry
class _Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self.model: ListingEnsemble | None = None
        self.mtime = 0.0
        self.fb: FeatureBuilder | None = None
        self.fb_time = 0.0

    def get_model(self) -> ListingEnsemble | None:
        path = settings.models_dir / "listing_current"
        cfg = path / "config.json"
        if not cfg.exists():
            return None
        m = cfg.stat().st_mtime
        with self._lock:
            if self.model is None or m != self.mtime:
                self.model = ListingEnsemble.load(path)
                self.mtime = m
                log.info("loaded listing model %s", self.model.meta.get("version"))
        return self.model

    def get_fb(self, max_age: float = 300) -> FeatureBuilder:
        with self._lock:
            if self.fb is None or time.time() - self.fb_time > max_age:
                with session_scope() as s:
                    self.fb = FeatureBuilder(load_tables(s))
                self.fb_time = time.time()
            return self.fb

    def invalidate(self):
        with self._lock:
            self.fb = None


registry = _Registry()


def model_version() -> str:
    m = registry.get_model()
    return m.meta.get("version", "deep_mdn") if m else "fallback-heuristic"


def _fallback_dist(feat: dict) -> MixtureDist:
    """Used only before the first training run: GMP if present, else board climatology."""
    if feat.get("gmp_missing", 1) == 0 and not pd.isna(feat.get("gmp_last_pct")):
        mu = math.log1p(max(-0.5, feat["gmp_last_pct"]))
        return MixtureDist(np.array([1.0]), np.array([mu * 0.8]), np.array([0.15]))
    mu = 0.08 if feat.get("board") == "mainboard" else 0.12
    return MixtureDist(np.array([1.0]), np.array([mu]), np.array([0.30]))


def _merge(dists: list[MixtureDist]) -> MixtureDist:
    return MixtureDist(np.concatenate([d.w for d in dists]) / len(dists),
                       np.concatenate([d.mu for d in dists]), np.concatenate([d.sigma for d in dists]))


# --------------------------------------------------------------------------- data access
def effective_stage(i: Ipo, today: dt.date | None = None) -> str | None:
    """Source stage labels can lag; the issue calendar is authoritative."""
    today = today or dt.date.today()
    st = i.stage
    if i.close_date and today > i.close_date and st in ("open", "upcoming", "rhp_filed"):
        return "closed"
    if i.open_date and i.close_date and i.open_date <= today <= i.close_date and st in ("upcoming", "rhp_filed"):
        return "open"
    return st


def ipo_dict(i: Ipo) -> dict[str, Any]:
    return {
        "slug": i.slug, "name": i.name, "symbol": i.symbol, "board": i.board, "stage": effective_stage(i),
        "open_date": i.open_date, "close_date": i.close_date, "listing_date": i.listing_date,
        "price_low": i.price_low, "price_high": i.price_high, "price": i.price, "lot_size": i.lot_size,
        "issue_size_cr": i.issue_size_cr, "anchor_pct": i.anchor_pct, "sector": i.sector,
        "registrar": i.registrar, "is_demo": i.is_demo, "listing_gain_pct": i.listing_gain_pct,
        "sub_total_index": (i.extra or {}).get("sub_total_index"), "partly_paid": i.partly_paid,
        "updated_at": i.updated_at,
    }


def latest_subscription(s: Session, slug: str) -> SubscriptionSnapshot | None:
    rows = list(s.scalars(select(SubscriptionSnapshot).where(SubscriptionSnapshot.ipo_slug == slug)
                          .order_by(SubscriptionSnapshot.as_of.desc()).limit(6)))
    if not rows:
        return None
    # prefer the most recent; on ties prefer NSE (official, has sNII/bNII split)
    rows.sort(key=lambda r: (r.as_of, r.source == "nse"), reverse=True)
    return rows[0]


def snapshot_dict(sn: SubscriptionSnapshot | None) -> dict | None:
    if sn is None:
        return None
    return {"source": sn.source, "as_of": sn.as_of, "final": sn.final, "total": sn.total_times, "qib": sn.qib_times,
            "nii": sn.nii_times, "snii": sn.snii_times, "bnii": sn.bnii_times, "rii": sn.rii_times,
            "emp": sn.emp_times, "rii_applications": sn.rii_applications}


def gmp_history(s: Session, slug: str) -> list[dict]:
    return [{"date": g.obs_date, "source": g.source, "gmp_rs": g.gmp_rs, "gmp_pct": g.gmp_pct,
             "fetched_at": g.fetched_at}
            for g in s.scalars(select(GmpObservation).where(GmpObservation.ipo_slug == slug).order_by(GmpObservation.obs_date))]


@lru_cache(maxsize=1)
def _hist_final_subs(_day: str) -> dict[tuple[str, str], np.ndarray]:
    """Empirical distribution of FINAL subscription by board & category (for not-yet-open issues)."""
    out: dict[tuple[str, str], list[float]] = {}
    with session_scope() as s:
        q = (select(Ipo.board, SubscriptionSnapshot.total_times, SubscriptionSnapshot.qib_times,
                    SubscriptionSnapshot.nii_times, SubscriptionSnapshot.rii_times, Ipo.close_date)
             .join(Ipo, Ipo.slug == SubscriptionSnapshot.ipo_slug).where(SubscriptionSnapshot.final.is_(True)))
        for board, tot, qib, nii, rii, cd in s.execute(q):
            if cd and cd.year < 2021:
                continue
            for cat, v in (("total", tot), ("qib", qib), ("nii", nii), ("rii", rii)):
                if v is not None and v > 0:
                    out.setdefault((board, cat), []).append(float(v))
    defaults = {"total": 15.0, "qib": 20.0, "nii": 15.0, "rii": 10.0}
    res = {}
    for board in ("mainboard", "sme"):
        for cat in defaults:
            vals = out.get((board, cat))
            res[(board, cat)] = np.sort(np.array(vals)) if vals and len(vals) >= 20 else \
                np.sort(defaults[cat] * np.exp(1.1 * np.random.default_rng(0).standard_normal(500)))
    return res


# --------------------------------------------------------------------------- demand scenarios
def demand_model(ipo: dict, snap: dict | None, fc: Forecaster, overrides: dict | None = None) -> dict:
    """Returns callables/params to generate final-subscription draws for each category."""
    overrides = overrides or {}
    now = ist_now()
    f = elapsed_fraction(ipo.get("open_date"), ipo.get("close_date"), now)
    if snap and snap.get("final"):
        f = 1.0
    cur = {}
    if snap:
        cur = {"total": snap.get("total"), "qib": snap.get("qib"), "nii": snap.get("nii"), "rii": snap.get("rii"),
               "snii": snap.get("snii") or snap.get("nii"), "bnii": snap.get("bnii") or snap.get("nii")}
    # SME issues often publish only an overall multiple: use it as the proxy for missing categories
    if cur.get("total") is not None:
        for k in ("rii", "nii", "snii", "bnii"):
            if cur.get(k) is None:
                cur[k] = cur["total"]
    for k in ("total", "qib", "nii", "rii"):
        if overrides.get(f"sub_{k}") is not None:
            cur[k] = float(overrides[f"sub_{k}"])
            if k == "nii":
                cur["snii"] = cur["bnii"] = cur[k]
    if overrides.get("final_subscription"):
        f = 1.0
    has_live = any(cur.get(k) for k in ("total", "rii"))
    stage = "final" if f >= 0.999 and has_live else ("live" if has_live else "prior")
    return {"f": f, "cur": cur, "stage": stage, "board": ipo.get("board", "mainboard")}


def draw_T(dm: dict, cat: str, z: np.ndarray, fc: Forecaster) -> np.ndarray:
    """Final subscription draws for a category given demand shocks z (monotone in z)."""
    fc_cat = {"rii": "rii", "snii": "nii", "bnii": "nii", "nii": "nii", "qib": "qib", "total": "total"}[cat]
    cur = dm["cur"].get(cat)
    if dm["stage"] != "prior" and cur is not None:
        mu, sd = fc.params(fc_cat, dm["f"])
        return max(cur, 0.01) * np.exp(mu + sd * z)
    hist = _hist_final_subs(dt.date.today().isoformat())[(dm["board"], fc_cat)]
    u = norm.cdf(z)
    return np.quantile(hist, np.clip(u, 0.001, 0.999))


def predict_ipo(ipo: dict, snap: dict | None, fb: FeatureBuilder, fc: Forecaster,
                overrides: dict | None = None) -> dict:
    """Predictive listing-return distribution integrated over demand uncertainty.
    Returns {'buckets': [MixtureDist x5], 'merged': MixtureDist, 'dm': demand model, 'features': row}."""
    overrides = overrides or {}
    model = registry.get_model()
    dm = demand_model(ipo, snap, fc, overrides)
    gmp_override = {"gmp_pct": overrides["gmp_pct"]} if overrides.get("gmp_pct") is not None else None
    zq = norm.ppf(DEMAND_QS)
    rows = []
    n_buckets = 1 if dm["stage"] == "final" else len(DEMAND_QS)
    for b in range(n_buckets):
        z = np.array([0.0 if n_buckets == 1 else zq[b]])
        sub_ov = {k: float(draw_T(dm, k, z, fc)[0]) for k in ("total", "qib", "nii", "rii")}
        if dm["stage"] == "prior" and not overrides.get("assume_subscription"):
            pass  # use historical demand distribution as a prior (features filled with draws)
        rows.append(fb.row(ipo, sub_override=sub_ov, gmp_override=gmp_override,
                           when=ipo.get("close_date") or dt.date.today()))
    df = pd.DataFrame(rows)
    if model is not None:
        dists = model.predict(df)
    else:
        dists = [_fallback_dist(r) for r in rows]
    if n_buckets == 1:
        dists = dists * len(DEMAND_QS)
    return {"buckets": dists, "merged": _merge(dists), "dm": dm, "features": rows[len(rows) // 2]}


def scenario_draws(pred: dict, fc: Forecaster, m: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Coupled (return, subscription) scenarios for one IPO given the common market factor m."""
    S = len(m)
    z_d = math.sqrt(RHO_DEMAND) * m + math.sqrt(1 - RHO_DEMAND) * rng.standard_normal(S)
    dm = pred["dm"]
    T = {"RII": draw_T(dm, "rii", z_d, fc), "SNII": draw_T(dm, "snii", 0.8 * z_d + 0.6 * rng.standard_normal(S), fc),
         "BNII": draw_T(dm, "bnii", 0.8 * z_d + 0.6 * rng.standard_normal(S), fc)}
    bucket = np.digitize(norm.cdf(z_d), [0.2, 0.4, 0.6, 0.8])
    u = norm.cdf(math.sqrt(RHO_MARKET) * m + math.sqrt(1 - RHO_MARKET) * rng.standard_normal(S))
    r = np.empty(S)
    for b, dist in enumerate(pred["buckets"]):
        idx = np.where(bucket == b)[0]
        if len(idx) == 0:
            continue
        samp = np.sort(dist.sample_return(20000, np.random.default_rng(1000 + b)))
        r[idx] = samp[np.clip((u[idx] * len(samp)).astype(int), 0, len(samp) - 1)]
    return np.clip(r, -0.95, 5.0), T


# --------------------------------------------------------------------------- public API
def _ipo_analytics(s: Session, i: Ipo, fb: FeatureBuilder, fc: Forecaster, overrides: dict | None = None,
                   detail: bool = False) -> dict:
    d = ipo_dict(i)
    sn = latest_subscription(s, i.slug)
    snap = snapshot_dict(sn)
    gh = gmp_history(s, i.slug)
    pred = predict_ipo(d, snap, fb, fc, overrides)
    dist: MixtureDist = pred["merged"]
    summ = dist.summary()
    dm = pred["dm"]
    lot_value = (d["price"] or 0) * (d["lot_size"] or 0)

    # allotment for the minimum individual application (MC over demand + L uncertainty)
    rng = np.random.default_rng(11)
    z = rng.standard_normal(4000)
    T_rii = draw_T(dm, "rii", z, fc)
    unit = 2 if (d["board"] == "sme" and (d["close_date"] or dt.date.today()) >= dt.date(2025, 7, 1)) else 1
    L = sample_L(d["board"], "RII", 4000, rng)
    p_s, _ = allot_given(T_rii, L, 1)
    allot = {
        "category": "RII", "unit_lots": unit, "min_amount": unit * lot_value if lot_value else None,
        "p_mean": float(p_s.mean()), "p_q10": float(np.quantile(p_s, 0.1)), "p_q90": float(np.quantile(p_s, 0.9)),
        "rii_final_forecast": [float(np.quantile(T_rii, q)) for q in (0.1, 0.5, 0.9)],
        "multi": [{"n": n, "p_at_least_one": prob_at_least_one(p_s, n), "naive": naive_at_least_one(float(p_s.mean()), n)}
                  for n in range(1, 6)],
        "L_prior": L_PRIORS.get((d["board"], "RII")),
    }
    # expected profit of the minimum application (coupled scenarios, no other IPOs)
    ev_min = None
    if lot_value:
        r, T = scenario_draws(pred, fc, rng.standard_normal(4000), rng)
        L2 = sample_L(d["board"], "RII", 4000, rng)
        p2, units_if = allot_given(T["RII"], L2, 1)
        val = units_if * unit * lot_value
        prof = val * r - settings.sell_cost_pct / 100 * val * (1 + r)
        ev = float(np.mean(p2 * prof))
        naive = float(p2.mean() * np.mean(prof))
        ev_min = {"expected_profit": ev, "amount": unit * lot_value, "ev_per_rupee": ev / (unit * lot_value),
                  "naive_expected_profit": naive,
                  "winners_curse": naive - ev,
                  "return_given_allotted_pct": 100 * float(np.sum(p2 * r) / max(np.sum(p2), 1e-9))}
    news_rows = list(s.scalars(select(NewsItem).where(NewsItem.ipo_slug == i.slug).order_by(NewsItem.published_at.desc()).limit(25)))
    latest_gmp = {}
    for g in gh:
        latest_gmp[g["source"]] = g
    gmp_vals = [g["gmp_pct"] for g in latest_gmp.values() if g.get("gmp_pct") is not None]
    out = {
        **d,
        "min_investment": unit * lot_value if lot_value else None,
        "subscription": snap,
        "demand_stage": dm["stage"], "bid_elapsed": dm["f"],
        "gmp_latest": list(latest_gmp.values()),
        "gmp_consensus_pct": float(np.mean(gmp_vals)) if gmp_vals else None,
        "prediction": {**summ, "model_version": model_version()},
        "allotment": allot, "min_application_ev": ev_min,
        "news_count": len(news_rows),
        "news_sentiment": (float(np.mean([(n.p_pos or 0) - (n.p_neg or 0) for n in news_rows if n.p_pos is not None]))
                           if any(n.p_pos is not None for n in news_rows) else None),
    }
    if detail:
        model = registry.get_model()
        feat_df = pd.DataFrame([pred["features"]])
        out["gmp_history"] = gh
        out["news"] = [{"title": n.title, "url": n.url, "publisher": n.publisher, "published_at": n.published_at,
                        "p_pos": n.p_pos, "p_neg": n.p_neg, "p_neu": n.p_neu, "topic": n.topic} for n in news_rows]
        out["explanation"] = model.explain(feat_df) if model else []
        out["features"] = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in pred["features"].items()
                           if k not in ("slug",)}
        out["distribution"] = _density(dist)
        out["subscription_history"] = [snapshot_dict(x) for x in s.scalars(
            select(SubscriptionSnapshot).where(SubscriptionSnapshot.ipo_slug == i.slug).order_by(SubscriptionSnapshot.as_of))]
        out["options"] = [o.__dict__ for o in application_options(d["board"], d["price"] or 0, d["lot_size"] or 0, d["close_date"])]
    return out


def _density(dist: MixtureDist, n: int = 121) -> list[dict]:
    lo = max(-0.6, dist.quantile_return(0.005))
    hi = min(3.0, dist.quantile_return(0.995))
    xs = np.linspace(lo, hi, n)
    y = np.log1p(xs)
    pdf_y = (dist.w[None, :] * norm.pdf((y[:, None] - dist.mu[None, :]) / dist.sigma[None, :]) / dist.sigma[None, :]).sum(1)
    pdf_r = pdf_y / (1 + xs)  # change of variables
    return [{"ret_pct": float(100 * x), "density": float(p)} for x, p in zip(xs, pdf_r)]


def list_active(include_demo: bool = True, overrides: dict[str, dict] | None = None) -> list[dict]:
    fb, fc = registry.get_fb(), Forecaster.load()
    out = []
    with session_scope() as s:
        today = dt.date.today()
        q = select(Ipo).where(Ipo.stage.in_(ACTIVE))
        for i in s.scalars(q):
            if i.is_demo and not include_demo:
                continue
            if i.offer_type not in (None, "ipo"):
                continue
            if i.price_low is None and i.price_high is None and i.issue_price is None and effective_stage(i) == "closed":
                continue  # no equity price band: not an equity IPO we can analyse (e.g. NCDs)
            if i.close_date and i.close_date < today - dt.timedelta(days=10):
                continue  # stale stage label
            try:
                out.append(_ipo_analytics(s, i, fb, fc, (overrides or {}).get(i.slug)))
            except Exception:  # never let one bad record break the dashboard
                log.exception("analytics failed for %s", i.slug)
    order = {"open": 0, "closed": 1, "allotment": 1, "pre_listing": 1, "upcoming": 2, "rhp_filed": 3}
    out.sort(key=lambda r: (order.get(r["stage"], 9), r["close_date"] or dt.date.max))
    return out


def ipo_detail(slug: str, overrides: dict | None = None) -> dict | None:
    fb, fc = registry.get_fb(), Forecaster.load()
    with session_scope() as s:
        i = s.get(Ipo, slug)
        if i is None:
            return None
        d = _ipo_analytics(s, i, fb, fc, overrides, detail=True)
        if i.listing_gain_pct is not None:
            d["actual_listing_gain_pct"] = i.listing_gain_pct
        return d


def build_candidates(slugs: list[str] | None, overrides: dict[str, dict] | None, include_demo: bool,
                     seed: int = 7) -> list[Candidate]:
    fb, fc = registry.get_fb(), Forecaster.load()
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(N_SCEN)  # common market factor shared by all IPOs
    cands = []
    with session_scope() as s:
        today = dt.date.today()
        q = select(Ipo).where(Ipo.stage.in_(("open", "upcoming", "rhp_filed")))
        for i in s.scalars(q):
            if slugs and i.slug not in slugs:
                continue
            if i.offer_type not in (None, "ipo") or effective_stage(i) not in ("open", "upcoming", "rhp_filed"):
                continue
            if i.is_demo and not include_demo:
                continue
            if not i.close_date or i.close_date < today:
                continue
            d = ipo_dict(i)
            snap = snapshot_dict(latest_subscription(s, i.slug))
            pred = predict_ipo(d, snap, fb, fc, (overrides or {}).get(i.slug))
            r, T = scenario_draws(pred, fc, m, np.random.default_rng(seed + len(cands) + 1))
            summ = pred["merged"].summary()
            cands.append(Candidate(
                slug=i.slug, name=i.name, board=i.board, price=d["price"] or 0, lot=d["lot_size"] or 0,
                close_date=max(i.close_date, today), unblock_date=next_business_day(max(i.close_date, today)),
                ret_samples=r, T_samples=T, p_loss=summ["p_loss"], mean_ret=summ["mean_pct"] / 100, is_demo=i.is_demo))
    return cands


def make_plan(payload: dict) -> dict:
    applicants, warnings = [], []
    seen_pan: dict[str, str] = {}
    for k, a in enumerate(payload.get("applicants") or [{"name": "Applicant 1"}]):
        pan = (a.get("pan") or a.get("pan_last4") or f"anon-{k}").strip().upper()
        if pan in seen_pan:
            warnings.append(f"'{a.get('name')}' shares a PAN with '{seen_pan[pan]}' — SEBI allows one application per PAN "
                            "per IPO, so they are merged (multiple demat accounts on one PAN are NOT extra chances).")
            continue
        seen_pan[pan] = a.get("name") or f"Applicant {k + 1}"
        applicants.append(Applicant(id=str(k), name=a.get("name") or f"Applicant {k + 1}", pan_key=pan,
                                    capital=a.get("capital"), allow_nii=a.get("allow_nii", True)))
    req = PlanRequest(
        capital=float(payload["capital"]), applicants=applicants, objective=payload.get("objective", "max_ev"),
        risk=payload.get("risk", "moderate"), boards=tuple(payload.get("boards") or ("mainboard", "sme")),
        allow_negative_ev=bool(payload.get("allow_negative_ev", False)), max_p_loss=payload.get("max_p_loss"),
        max_ipos=payload.get("max_ipos"), exclude=tuple(payload.get("exclude") or ()),
        allow_nii=bool(payload.get("allow_nii", True)),
    )
    cands = build_candidates(payload.get("slugs"), payload.get("overrides"), bool(payload.get("include_demo", False)))
    if payload.get("objective") == "compare":
        res = compare_objectives(cands, req)
    else:
        res = build_plan(cands, req)
    res["warnings"] = warnings + ([] if cands else ["No IPOs are currently open or upcoming with known dates."])
    if payload.get("holding") == "long_term":
        res["warnings"].append("Holding preference is long-term: the model predicts LISTING-day returns only; "
                               "post-listing performance is not modelled and can differ substantially.")
    res["model_version"] = model_version()
    res["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    res["n_candidates"] = len(cands)
    return res


def refresh_predictions() -> dict:
    """Persist a prediction snapshot for every active IPO (for later live evaluation)."""
    registry.invalidate()
    rows = list_active(include_demo=False)
    with session_scope() as s:
        for r in rows:
            p = r["prediction"]
            s.add(Prediction(ipo_slug=r["slug"], model_version=p["model_version"], mean_pct=p["mean_pct"],
                             median_pct=p["median_pct"], q10_pct=p["q10_pct"], q90_pct=p["q90_pct"], p_loss=p["p_loss"],
                             payload={"allotment": r["allotment"], "demand_stage": r["demand_stage"],
                                      "gmp_consensus_pct": r["gmp_consensus_pct"]}))
    return {"stored": len(rows)}
