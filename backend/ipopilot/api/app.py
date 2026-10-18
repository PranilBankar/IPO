"""FastAPI application — REST API for the IPOPilot frontend.

    GET  /api/health                     service + data freshness
    GET  /api/ipos                       active IPOs with analytics (subscription, GMP, model, allotment)
    GET  /api/ipos/{slug}                full detail: distribution, explanation, GMP/subscription history, news
    POST /api/ipos/{slug}/whatif         re-run the model with GMP / subscription overrides
    POST /api/plan                       optimise an application plan (objective A/B/C/D or "compare")
    POST /api/allotment/calc             stand-alone allotment-probability calculator
    GET  /api/history                    listed IPOs (searchable) with actual listing gains
    GET  /api/model                      model card: metrics, backtest, features
    GET  /api/sources                    data sources + last ingestion runs
    POST /api/admin/refresh              trigger live ingestion (requires X-Admin-Key if configured)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from typing import Any, Literal

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import GmpObservation, IngestRun, Ipo, PlanRecord, SubscriptionSnapshot, get_session, init_db, session_scope
from ..engine import service
from ..engine.allotment import allot_given, naive_at_least_one, prob_at_least_one, sample_L

log = logging.getLogger("ipopilot.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DISCLAIMER = ("IPOPilot AI is an educational research and decision-support tool, not investment advice. "
              "GMP is unofficial; model outputs are probabilistic estimates that can be wrong. "
              "It is not a SEBI-registered investment adviser.")


def _clean(o: Any) -> Any:
    """JSON-safe conversion (NaN/inf -> None, numpy -> python, dates -> iso)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    return o


def ok(data: Any) -> JSONResponse:
    return JSONResponse(_clean(data))


app = FastAPI(title="IPOPilot AI", version="1.0.0", description=DISCLAIMER)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    init_db()
    from ..ingestion.pipeline import ensure_demo_ipo
    ensure_demo_ipo()
    if settings.enable_scheduler:
        from ..scheduler import start_scheduler
        start_scheduler()


# --------------------------------------------------------------------------- schemas
class Overrides(BaseModel):
    gmp_pct: float | None = Field(None, ge=-90, le=500)
    sub_total: float | None = Field(None, ge=0, le=5000)
    sub_qib: float | None = Field(None, ge=0, le=5000)
    sub_nii: float | None = Field(None, ge=0, le=5000)
    sub_rii: float | None = Field(None, ge=0, le=5000)
    final_subscription: bool = False


class ApplicantIn(BaseModel):
    name: str = Field(..., max_length=60)
    pan: str | None = Field(None, max_length=12, description="PAN or any unique label (only used to de-duplicate)")
    capital: float | None = Field(None, gt=0)
    allow_nii: bool = True


class PlanIn(BaseModel):
    capital: float = Field(..., gt=0, le=1e9)
    applicants: list[ApplicantIn] = Field(..., min_length=1, max_length=20)
    objective: Literal["max_ev", "max_allot_prob", "capital_efficiency", "risk_adjusted", "compare"] = "compare"
    risk: Literal["conservative", "moderate", "aggressive"] = "moderate"
    boards: list[Literal["mainboard", "sme"]] = ["mainboard", "sme"]
    holding: Literal["listing", "long_term"] = "listing"
    allow_negative_ev: bool = False
    allow_nii: bool = True
    max_p_loss: float | None = Field(None, ge=0, le=1)
    max_ipos: int | None = Field(None, ge=1, le=50)
    exclude: list[str] = []
    slugs: list[str] | None = None
    include_demo: bool = False
    overrides: dict[str, Overrides] = {}

    @field_validator("boards")
    @classmethod
    def _nonempty(cls, v):
        if not v:
            raise ValueError("choose at least one board")
        return v


class AllotCalcIn(BaseModel):
    board: Literal["mainboard", "sme"] = "mainboard"
    category: Literal["RII", "SNII", "BNII"] = "RII"
    subscription_times: float = Field(..., gt=0, le=10000)
    applications: float | None = Field(None, gt=0)
    offered_shares: float | None = Field(None, gt=0)
    unit_shares: float | None = Field(None, gt=0)
    applicants: int = Field(3, ge=1, le=20)


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health(s: Session = Depends(get_session)):
    last = s.scalar(select(func.max(IngestRun.finished_at)).where(IngestRun.ok.is_(True)))
    return ok({"status": "ok", "model_version": service.model_version(), "last_ingest": last,
               "ipos": s.scalar(select(func.count(Ipo.slug))), "disclaimer": DISCLAIMER})


@app.get("/api/ipos")
def ipos(include_demo: bool = True):
    return ok({"generated_at": dt.datetime.now(), "items": service.list_active(include_demo=include_demo),
               "disclaimer": DISCLAIMER})


@app.get("/api/ipos/{slug}")
def ipo_detail(slug: str):
    d = service.ipo_detail(slug)
    if d is None:
        raise HTTPException(404, "IPO not found")
    return ok(d)


@app.post("/api/ipos/{slug}/whatif")
def whatif(slug: str, ov: Overrides):
    d = service.ipo_detail(slug, ov.model_dump(exclude_none=True))
    if d is None:
        raise HTTPException(404, "IPO not found")
    return ok(d)


def _store_plan(payload: dict, res: dict) -> None:
    try:
        with session_scope() as s:
            s.add(PlanRecord(request=_clean(payload), result=_clean({k: v for k, v in res.items() if k != "plans"})))
    except Exception:  # persistence is best-effort and never delays the response
        log.exception("could not store plan")


@app.post("/api/plan")
def plan(body: PlanIn, bg: BackgroundTasks):
    payload = body.model_dump()
    payload["overrides"] = {k: v.model_dump(exclude_none=True) for k, v in body.overrides.items()}
    payload["applicants"] = [a.model_dump() for a in body.applicants]
    res = service.make_plan(payload)
    res["disclaimer"] = DISCLAIMER
    bg.add_task(_store_plan, payload, res)
    return ok(res)


@app.post("/api/allotment/calc")
def allot_calc(b: AllotCalcIn):
    rng = np.random.default_rng(5)
    n = 20000
    if b.applications and b.offered_shares and b.unit_shares:
        p_exact = min(1.0, (b.offered_shares / b.unit_shares) / b.applications)
        p_s = np.full(n, p_exact)
        method = "exact (units available / applications)"
    else:
        L = sample_L(b.board, b.category, n, rng)
        p_s, _ = allot_given(np.full(n, b.subscription_times), L, 1)
        method = "estimated (L prior: average application size uncertain)"
    return ok({
        "method": method, "p_mean": float(p_s.mean()), "p_q10": float(np.quantile(p_s, .1)),
        "p_q90": float(np.quantile(p_s, .9)),
        "table": [{"n": k, "p_at_least_one": prob_at_least_one(p_s, k), "naive": naive_at_least_one(float(p_s.mean()), k)}
                  for k in range(1, b.applicants + 1)],
    })


@app.get("/api/history")
def history(q: str | None = None, board: str | None = None, year: int | None = None,
            limit: int = Query(100, le=500), offset: int = 0, s: Session = Depends(get_session)):
    stmt = select(Ipo).where(Ipo.listing_gain_pct.is_not(None), Ipo.is_demo.is_(False))
    if q:
        stmt = stmt.where(or_(Ipo.name.ilike(f"%{q}%"), Ipo.symbol.ilike(f"%{q}%")))
    if board:
        stmt = stmt.where(Ipo.board == board)
    if year:
        stmt = stmt.where(Ipo.close_date >= dt.date(year, 1, 1), Ipo.close_date <= dt.date(year, 12, 31))
    total = s.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = s.scalars(stmt.order_by(Ipo.listing_date.desc()).limit(limit).offset(offset)).all()
    gmp_slugs = {r for (r,) in s.execute(select(GmpObservation.ipo_slug).distinct())}
    sub_map = {}
    for sn in s.scalars(select(SubscriptionSnapshot).where(
            SubscriptionSnapshot.ipo_slug.in_([r.slug for r in rows]), SubscriptionSnapshot.final.is_(True))):
        sub_map[sn.ipo_slug] = sn.total_times
    return ok({"total": total, "items": [{
        "slug": r.slug, "name": r.name, "board": r.board, "close_date": r.close_date, "listing_date": r.listing_date,
        "price": r.price, "issue_size_cr": r.issue_size_cr, "listing_gain_pct": r.listing_gain_pct,
        "sub_total": sub_map.get(r.slug) or (r.extra or {}).get("sub_total_index"), "has_gmp": r.slug in gmp_slugs,
    } for r in rows]})


@app.get("/api/history/stats")
def history_stats(s: Session = Depends(get_session)):
    rows = s.execute(select(Ipo.board, Ipo.close_date, Ipo.listing_gain_pct).where(
        Ipo.listing_gain_pct.is_not(None), Ipo.close_date.is_not(None), Ipo.is_demo.is_(False))).all()
    by: dict[tuple[int, str], list[float]] = {}
    for board, cd, g in rows:
        if cd.year >= 2012:
            by.setdefault((cd.year, board), []).append(g)
    return ok({"by_year": [{"year": y, "board": b, "n": len(v), "mean_gain": float(np.mean(v)),
                            "median_gain": float(np.median(v)), "share_negative": float(np.mean(np.array(v) < 0))}
                           for (y, b), v in sorted(by.items())]})


@app.get("/api/model")
def model_card():
    md = settings.models_dir
    metrics = json.loads((md / "metrics.json").read_text()) if (md / "metrics.json").exists() else None
    bt = json.loads((md / "backtest.json").read_text()) if (md / "backtest.json").exists() else None
    return ok({"model_version": service.model_version(), "metrics": metrics, "backtest": bt})


@app.get("/api/sources")
def sources(s: Session = Depends(get_session)):
    runs = s.scalars(select(IngestRun).order_by(IngestRun.started_at.desc()).limit(30)).all()
    counts = {
        "ipos": s.scalar(select(func.count(Ipo.slug))),
        "listed_with_gain": s.scalar(select(func.count(Ipo.slug)).where(Ipo.listing_gain_pct.is_not(None))),
        "subscription_snapshots": s.scalar(select(func.count(SubscriptionSnapshot.id))),
        "gmp_observations": s.scalar(select(func.count(GmpObservation.id))),
        "ipos_with_gmp": s.scalar(select(func.count(func.distinct(GmpObservation.ipo_slug)))),
    }
    return ok({"counts": counts, "runs": [{"job": r.job, "started_at": r.started_at, "finished_at": r.finished_at,
                                           "ok": r.ok, "stats": r.stats, "error": (r.error or "").splitlines()[0] if r.error else None}
                                          for r in runs],
               "sources": [
                   {"name": "NSE India", "url": "https://www.nseindia.com/market-data/all-upcoming-issues-ipo", "kind": "official",
                    "fields": "current/upcoming issues, bid lot, price band, category-wise subscription"},
                   {"name": "BSE India (via ipos.markets)", "url": "https://www.bseindia.com/publicissue.html", "kind": "official",
                    "fields": "SME issues, BSE codes"},
                   {"name": "SEBI data resources", "url": "https://www.sebi.gov.in/", "kind": "official",
                    "fields": "allotment rules (ICDR), SME rule changes"},
                   {"name": "ipos.markets API", "url": "https://ipos.markets/api-docs", "kind": "aggregator (official data)",
                    "fields": "issue history since 1990s, listing gains, final subscription"},
                   {"name": "IPOWatch GMP", "url": "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/", "kind": "unofficial GMP",
                    "fields": "live GMP table, day-wise GMP history"},
                   {"name": "MarketLiveIPOGMP", "url": "https://marketliveipogmp.com/", "kind": "unofficial GMP",
                    "fields": "GMP, price, lot size"},
                   {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/quote/%5ENSEI", "kind": "market data",
                    "fields": "NIFTY 50, India VIX, NIFTY Smallcap 250"},
                   {"name": "Google News RSS", "url": "https://news.google.com/", "kind": "news", "fields": "headlines for FinBERT"},
               ]})


@app.post("/api/admin/refresh")
def admin_refresh(x_admin_key: str | None = Header(None)):
    if settings.admin_api_key and x_admin_key != settings.admin_api_key:
        raise HTTPException(401, "invalid admin key")
    from ..ingestion import pipeline
    res = pipeline.run_all_live()
    service.registry.invalidate()
    return ok({"results": res})
