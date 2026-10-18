"""Ingestion orchestration: pulls every source, validates, cross-matches and stores.

Jobs (all idempotent, safe to re-run):
  sync_index          ipos.markets issues index  -> ipos (universe + listing-gain labels)
  sync_live           NSE + ipos.markets open/upcoming issues, lot sizes, live subscription
  sync_gmp_live       IPOWatch + MarketLiveIPOGMP GMP for active issues
  backfill_subscriptions   final category-wise subscription for listed issues
  backfill_gmp        IPOWatch day-wise GMP history for listed issues
  sync_market         NIFTY / VIX / Smallcap daily bars
  sync_news           Google News headlines (+ FinBERT scoring) for active issues
"""
from __future__ import annotations

import datetime as dt
import logging
import traceback
from collections import defaultdict
from typing import Any, Callable, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import GmpObservation, IngestRun, Ipo, MarketBar, NewsItem, SubscriptionSnapshot, session_scope, utcnow
from . import gmp as gmp_src
from . import ipos_markets as im
from . import market, news, nse
from .util import name_similarity, norm_name, parse_band, parse_date, parse_datetime, to_float

log = logging.getLogger(__name__)

ACTIVE_STAGES = {"upcoming", "rhp_filed", "open", "closed", "allotment", "pre_listing"}


# --------------------------------------------------------------------------- helpers
def _run(job: str, fn: Callable[[Session], dict[str, Any]]) -> dict[str, Any]:
    with session_scope() as s:
        run = IngestRun(job=job)
        s.add(run)
        s.flush()
        run_id = run.id
    stats: dict[str, Any] = {}
    ok, err = False, None
    try:
        with session_scope() as s:
            stats = fn(s) or {}
        ok = True
    except Exception as e:  # noqa: BLE001 - we log and persist the failure
        err = f"{e}\n{traceback.format_exc()}"
        log.exception("Job %s failed", job)
    with session_scope() as s:
        run = s.get(IngestRun, run_id)
        run.finished_at, run.ok, run.stats, run.error = utcnow(), ok, stats, err
    log.info("job=%s ok=%s stats=%s", job, ok, stats)
    return {"job": job, "ok": ok, "stats": stats, "error": err and err.splitlines()[0]}


class Matcher:
    """Resolve an external (name, symbol, date) to a canonical Ipo slug."""

    def __init__(self, ipos: Iterable[Ipo]):
        self.by_symbol: dict[str, str] = {}
        self.by_norm: dict[str, list[Ipo]] = defaultdict(list)
        self.items = list(ipos)
        for i in self.items:
            if i.symbol:
                self.by_symbol[i.symbol.upper()] = i.slug
            self.by_norm[norm_name(i.name)].append(i)

    def match(self, name: str | None, symbol: str | None = None, near: dt.date | None = None,
              pool: Iterable[Ipo] | None = None, threshold: float = 0.86) -> str | None:
        if symbol and symbol.upper() in self.by_symbol:
            return self.by_symbol[symbol.upper()]
        if not name:
            return None
        cands = self.by_norm.get(norm_name(name)) or []
        if not cands:
            scored = [(name_similarity(name, i.name), i) for i in (pool or self.items)]
            scored = [x for x in scored if x[0] >= threshold]
            scored.sort(key=lambda x: -x[0])
            cands = [i for _, i in scored[:5]]
        if not cands:
            return None
        if near is not None and len(cands) > 1:
            cands.sort(key=lambda i: abs(((i.close_date or i.open_date or near) - near).days))
        return cands[0].slug


def _upsert_gmp(s: Session, slug: str, obs: dict[str, Any], price: float | None) -> bool:
    """Daily granularity; repeated intra-day fetches update the day's value and keep an
    intraday trail in `raw` so nothing is lost."""
    d = obs["obs_date"]
    g = obs.get("gmp_rs")
    pct = obs.get("gmp_pct")
    ref = obs.get("ref_price") or price
    if pct is None and g is not None and ref:
        pct = 100.0 * g / ref
    row = s.scalar(select(GmpObservation).where(
        GmpObservation.ipo_slug == slug, GmpObservation.source == obs["source"], GmpObservation.obs_date == d))
    now = utcnow().isoformat(timespec="seconds")
    if row is None:
        s.add(GmpObservation(ipo_slug=slug, source=obs["source"], obs_date=d, gmp_rs=g, gmp_pct=pct,
                             ref_price=ref, raw={**(obs.get("raw") or {}), "intraday": [[now, g]]}))
        return True
    if row.gmp_rs != g and d == dt.date.today():
        raw = dict(row.raw or {})
        raw["intraday"] = (raw.get("intraday") or []) + [[now, g]]
        row.gmp_rs, row.gmp_pct, row.ref_price, row.raw, row.fetched_at = g, pct, ref, raw, utcnow()
    return False


# --------------------------------------------------------------------------- jobs
def sync_index() -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        issues = im.fetch_issues_index()
        if not issues:
            raise RuntimeError("ipos.markets issues index unavailable")
        existing = {i.slug: i for i in s.scalars(select(Ipo))}
        n_new = n_upd = 0
        for x in issues:
            if x.get("board") not in ("mainboard", "sme") or x.get("offer_type") not in ("ipo", None):
                continue
            lo, hi = parse_band(x.get("band"))
            fields = dict(
                name=x.get("name") or x["slug"], symbol=x.get("symbol"), board=x["board"],
                offer_type=x.get("offer_type"), stage=x.get("stage"),
                open_date=parse_date(x.get("open")), close_date=parse_date(x.get("close")),
                listing_date=parse_date(x.get("listing")), price_low=lo, price_high=hi,
                issue_price=to_float(x.get("price")), issue_size_cr=to_float(x.get("size")),
                sector=x.get("sector"), anchor_pct=to_float(x.get("anchor_pct")),
                partly_paid=bool(x.get("partly_paid")), listing_gain_pct=to_float(x.get("gain")),
                cmp=to_float(x.get("cmp")),
            )
            extra = {"sub_total_index": to_float(x.get("sub")), "status": x.get("status")}
            row = existing.get(x["slug"])
            if row is None:
                s.add(Ipo(slug=x["slug"], extra=extra, **fields))
                n_new += 1
            else:
                for k, v in fields.items():
                    if v is not None or k in ("stage",):
                        setattr(row, k, v)
                row.extra = {**(row.extra or {}), **extra}
                n_upd += 1
        return {"issues": len(issues), "new": n_new, "updated": n_upd}

    return _run("sync_index", job)


def _snapshot_from_nse(slug: str, det: dict[str, Any], final: bool) -> SubscriptionSnapshot | None:
    cats = det.get("categories") or {}
    if not cats:
        return None
    as_of = parse_datetime(det.get("as_of")) or utcnow().replace(minute=0, second=0, microsecond=0)

    def t(k):
        return (cats.get(k) or {}).get("times")

    return SubscriptionSnapshot(
        ipo_slug=slug, source="nse", as_of=as_of, final=final,
        total_times=t("TOTAL"), qib_times=t("QIB"), nii_times=t("NII"), snii_times=t("SNII"),
        bnii_times=t("BNII"), rii_times=t("RII"), emp_times=t("EMP"),
        rii_offered=(cats.get("RII") or {}).get("offered"), rii_bid=(cats.get("RII") or {}).get("bid"),
        snii_offered=(cats.get("SNII") or {}).get("offered"), snii_bid=(cats.get("SNII") or {}).get("bid"),
        raw={"categories": cats},
    )


def _snapshot_from_im(slug: str, sub: dict[str, Any]) -> SubscriptionSnapshot | None:
    if not sub or not (sub.get("categories") or sub.get("total_times") or sub.get("lower_bound_times")):
        return None
    c = im.category_times(sub)
    as_of = parse_datetime(sub.get("as_of")) or utcnow()
    return SubscriptionSnapshot(
        ipo_slug=slug, source="ipos_markets", as_of=as_of, final=bool(sub.get("final")),
        total_times=c["total_times"], qib_times=c["qib_times"], nii_times=c["nii_times"],
        rii_times=c["rii_times"], emp_times=c["emp_times"], rii_offered=c["rii_offered"],
        rii_bid=c["rii_bid"], rii_applications=c["rii_applications"],
        raw={"categories": sub.get("categories"), "basis": sub.get("basis")},
    )


def _add_snapshot(s: Session, snap: SubscriptionSnapshot | None) -> bool:
    if snap is None:
        return False
    exists = s.scalar(select(SubscriptionSnapshot.id).where(
        SubscriptionSnapshot.ipo_slug == snap.ipo_slug, SubscriptionSnapshot.source == snap.source,
        SubscriptionSnapshot.as_of == snap.as_of))
    if exists:
        return False
    s.add(snap)
    return True


def sync_live() -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        stats = defaultdict(int)
        ov = im.fetch_overview()
        for bucket in ("open", "closed", "upcoming", "recent"):
            for x in ov.get(bucket, []) or []:
                if x.get("board") not in ("mainboard", "sme") or x.get("offer_type") not in ("ipo", None):
                    continue
                row = s.get(Ipo, x["slug"])
                lo, hi = parse_band(x.get("band"))
                if row is None:
                    row = Ipo(slug=x["slug"], name=x.get("name") or x["slug"], board=x["board"],
                              offer_type=x.get("offer_type"))
                    s.add(row)
                    stats["new"] += 1
                row.symbol = x.get("symbol") or row.symbol
                row.stage = x.get("stage") or row.stage
                row.open_date = parse_date(x.get("open")) or row.open_date
                row.close_date = parse_date(x.get("close")) or row.close_date
                row.listing_date = parse_date(x.get("listing")) or row.listing_date
                row.price_low, row.price_high = lo or row.price_low, hi or row.price_high
                row.issue_price = to_float(x.get("price")) or row.issue_price
                row.issue_size_cr = to_float(x.get("size")) or row.issue_size_cr
                row.anchor_pct = to_float(x.get("anchor_pct")) or row.anchor_pct
                if x.get("gain") is not None:
                    row.listing_gain_pct = to_float(x.get("gain"))
                stats[bucket] += 1
        s.flush()

        active = list(s.scalars(select(Ipo).where(Ipo.stage.in_(ACTIVE_STAGES))))
        matcher = Matcher(active)
        # --- NSE: official current / upcoming issues
        nse_rows = {r.get("symbol"): r for r in (nse.fetch_current() + nse.fetch_upcoming()) if r.get("symbol")}
        for sym, r in nse_rows.items():
            slug = matcher.match(r.get("companyName"), sym, parse_date(r.get("issueEndDate")), pool=active)
            if not slug:
                stats["nse_unmatched"] += 1
                continue
            ipo = s.get(Ipo, slug)
            if (r.get("series") or "EQ").upper() not in ("EQ", "SME", "BE", "SM", "ST"):
                ipo.offer_type = "debt"  # NCDs / bonds occasionally mislabelled upstream as IPOs
                stats["non_equity"] += 1
                continue
            if not ipo.symbol:
                ipo.symbol = sym
            series = "SME" if ipo.board == "sme" else (r.get("series") or "EQ")
            det = nse.fetch_detail(sym, series)
            if not det:
                continue
            p = nse.parse_detail(det)
            if p.get("lot_size"):
                ipo.lot_size = p["lot_size"]
            if p.get("registrar"):
                ipo.registrar = p["registrar"]
            lo, hi = parse_band(p.get("price_range_text"))
            if hi:
                ipo.price_low, ipo.price_high = lo, hi
            ipo.extra = {**(ipo.extra or {}), "nse_issue_size_text": p.get("issue_size_text"),
                         "retail_max_amount": p.get("retail_max_amount")}
            stats["nse_snapshots"] += _add_snapshot(s, _snapshot_from_nse(slug, p, final=False))

        # --- ipos.markets live subscription for open / closed issues
        for ipo in active:
            if ipo.stage not in ("open", "closed", "allotment", "pre_listing"):
                continue
            sub = im.fetch_subscription(ipo.symbol or ipo.slug)
            stats["im_snapshots"] += _add_snapshot(s, _snapshot_from_im(ipo.slug, sub or {}))
        return dict(stats)

    return _run("sync_live", job)


def sync_gmp_live() -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        stats = defaultdict(int)
        today = dt.date.today()
        pool = [i for i in s.scalars(select(Ipo)) if (i.stage in ACTIVE_STAGES) or (
            i.listing_date and abs((i.listing_date - today).days) <= 5)]
        matcher = Matcher(pool)
        for obs in gmp_src.fetch_ipowatch_live():
            slug = matcher.match(obs["name"], near=today, pool=pool, threshold=0.8)
            if not slug:
                stats["ipowatch_unmatched"] += 1
                continue
            ipo = s.get(Ipo, slug)
            if obs.get("url"):
                ipo.extra = {**(ipo.extra or {}), "ipowatch_url": obs["url"] if obs["url"].startswith("http")
                             else "https://ipowatch.in" + obs["url"]}
            stats["ipowatch"] += _upsert_gmp(s, slug, obs, ipo.price) or 0
        for page in gmp_src.list_marketlive_pages():
            slug = matcher.match(page["name"], near=today, pool=pool, threshold=0.8)
            if not slug:
                stats["marketlive_unmatched"] += 1
                continue
            obs = gmp_src.fetch_marketlive_page(page["url"])
            if not obs:
                continue
            ipo = s.get(Ipo, slug)
            if obs.get("lot_size") and not ipo.lot_size:
                ipo.lot_size = obs["lot_size"]
            stats["marketlive"] += _upsert_gmp(s, slug, obs, ipo.price) or 0
        return dict(stats)

    return _run("sync_gmp_live", job)


def backfill_subscriptions(limit: int | None = None) -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        stats = defaultdict(int)
        idx = im.fetch_subscription_index()
        ipos = list(s.scalars(select(Ipo)))
        by_slug = {i.slug: i for i in ipos}
        matcher = Matcher(ipos)
        have = {r for (r,) in s.execute(select(SubscriptionSnapshot.ipo_slug).where(
            SubscriptionSnapshot.source == "ipos_markets", SubscriptionSnapshot.final.is_(True)))}
        todo = []
        for x in idx:
            if x.get("board") not in ("mainboard", "sme") or x.get("basis") == "bids_only":
                continue
            slug = x["slug"] if x.get("slug") in by_slug else matcher.match(x.get("name"), x.get("symbol"))
            if not slug or slug in have:
                continue
            todo.append((slug, x))
        if limit:
            todo = todo[:limit]
        for n, (slug, x) in enumerate(todo, 1):
            key = x.get("path", "").rsplit("/", 1)[-1].removesuffix(".json") or x.get("symbol") or x.get("slug")
            historical = x.get("status") in ("post_listing", "listed")
            sub = im.fetch_subscription(key, historical=historical)
            snap = _snapshot_from_im(slug, sub or {})
            if snap is not None:
                snap.final = snap.final or historical
            stats["added"] += _add_snapshot(s, snap)
            if n % 100 == 0:
                s.commit()
                log.info("backfill_subscriptions %d/%d", n, len(todo))
        stats["candidates"] = len(todo)
        return dict(stats)

    return _run("backfill_subscriptions", job)


def backfill_gmp(since_year: int = 2019, limit: int | None = None) -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        stats = defaultdict(int)
        ipos = [i for i in s.scalars(select(Ipo)) if i.close_date and i.close_date.year >= since_year]
        have = {r for (r,) in s.execute(select(GmpObservation.ipo_slug).where(GmpObservation.source == "ipowatch"))}
        by_year: dict[int, list[Ipo]] = defaultdict(list)
        for i in ipos:
            by_year[i.close_date.year].append(i)
        urls = gmp_src.list_ipowatch_gmp_pages()
        stats["pages"] = len(urls)
        exact: dict[str, list[Ipo]] = defaultdict(list)
        by_token: dict[str, list[Ipo]] = defaultdict(list)
        for i in ipos:
            nn = norm_name(i.name)
            exact[nn].append(i)
            for tok in set(nn.split()):
                by_token[tok].append(i)
        jobs = []
        for u in urls:
            pname = gmp_src.page_name_from_url(u)
            cands = exact.get(norm_name(pname))
            if not cands:
                # only compare against IPOs sharing at least one name token (fast pre-filter)
                pool = {i.slug: i for tok in norm_name(pname).split() for i in by_token.get(tok, [])}
                cands = [i for i in pool.values() if name_similarity(pname, i.name) >= 0.9]
            if len(cands) != 1:
                stats["ambiguous_or_unmatched"] += 1
                continue
            ipo = cands[0]
            if ipo.slug in have:
                continue
            jobs.append((u, ipo))
        if limit:
            jobs = jobs[:limit]
        # network + HTML parsing in worker threads (the per-host throttle keeps it polite);
        # database writes stay on this thread
        from concurrent.futures import ThreadPoolExecutor

        def fetch(job):
            u, ipo = job
            try:
                return job, gmp_src.fetch_ipowatch_history(u, ipo.close_date.year, ipo.close_date)
            except Exception:  # noqa: BLE001 - one bad page must not stop the backfill
                log.exception("failed %s", u)
                return job, []

        with ThreadPoolExecutor(max_workers=4) as pool:
            for n, ((u, ipo), rows) in enumerate(pool.map(fetch, jobs), 1):
                # only keep observations dated before listing (GMP after listing is meaningless)
                cutoff = ipo.listing_date or (ipo.close_date + dt.timedelta(days=7))
                rows = [r for r in rows if ipo.close_date - dt.timedelta(days=40) <= r["obs_date"] <= cutoff]
                for r in rows:
                    stats["obs"] += _upsert_gmp(s, ipo.slug, r, ipo.price) or 0
                stats["ipos_with_gmp"] += bool(rows)
                have.add(ipo.slug)
                if n % 25 == 0:
                    s.commit()  # release the SQLite write lock periodically
                    log.info("backfill_gmp %d/%d", n, len(jobs))
        stats["fetched"] = len(jobs)
        return dict(stats)

    return _run("backfill_gmp", job)


def sync_market() -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        stats = {}
        for name, ysym in market.TICKERS.items():
            bars = market.fetch_daily(ysym)
            have = {d for (d,) in s.execute(select(MarketBar.date).where(MarketBar.ticker == name))}
            new = [MarketBar(ticker=name, date=d, close=c) for d, c in bars if d not in have]
            s.add_all(new)
            stats[name] = len(new)
        return stats

    return _run("sync_market", job)


def sync_news(slugs: list[str] | None = None, score: bool = True) -> dict[str, Any]:
    def job(s: Session) -> dict[str, Any]:
        stats = defaultdict(int)
        q = select(Ipo).where(Ipo.stage.in_(ACTIVE_STAGES)) if not slugs else select(Ipo).where(Ipo.slug.in_(slugs))
        targets = list(s.scalars(q))
        new_items: list[NewsItem] = []
        for ipo in targets:
            seen = {u for (u,) in s.execute(select(NewsItem.url).where(NewsItem.ipo_slug == ipo.slug))}
            for it in news.fetch_news(ipo.name):
                if not it.get("url") or it["url"] in seen:
                    continue
                seen.add(it["url"])
                n = NewsItem(ipo_slug=ipo.slug, **it)
                s.add(n)
                new_items.append(n)
        stats["new_items"] = len(new_items)
        if score and new_items:
            from ..ml.sentiment import score_items
            score_items(new_items)
            stats["scored"] = len(new_items)
        return dict(stats)

    return _run("sync_news", job)


def ensure_demo_ipo() -> None:
    """A clearly-labelled FICTIONAL issue so the UI can be demonstrated even on a day with
    no live IPOs, without misrepresenting real market data."""
    today = dt.date.today()
    with session_scope() as s:
        row = s.get(Ipo, "demo-aurora-robotics")
        if row is None:
            row = Ipo(slug="demo-aurora-robotics", name="Aurora Robotics Ltd (FICTIONAL DEMO)", board="mainboard",
                      is_demo=True, offer_type="ipo", symbol="DEMOAURORA")
            s.add(row)
        row.stage, row.open_date, row.close_date = "open", today - dt.timedelta(days=1), today + dt.timedelta(days=2)
        row.listing_date = today + dt.timedelta(days=5)
        row.price_low, row.price_high, row.lot_size, row.issue_size_cr = 380.0, 400.0, 37, 1250.0
        row.anchor_pct, row.sector = 30.0, "Industrials"
        s.flush()
        if not s.scalar(select(SubscriptionSnapshot.id).where(SubscriptionSnapshot.ipo_slug == row.slug)):
            s.add(SubscriptionSnapshot(ipo_slug=row.slug, source="demo", as_of=utcnow(), total_times=18.5,
                                       qib_times=31.0, nii_times=22.0, snii_times=17.0, bnii_times=24.5,
                                       rii_times=8.4, rii_offered=1_093_750 * 3, rii_bid=1_093_750 * 3 * 8.4))
        if not s.scalar(select(GmpObservation.id).where(GmpObservation.ipo_slug == row.slug)):
            for k, g in enumerate([40, 48, 55, 62]):
                s.add(GmpObservation(ipo_slug=row.slug, source="demo", obs_date=today - dt.timedelta(days=3 - k),
                                     gmp_rs=g, gmp_pct=100 * g / 400, ref_price=400))


def run_all_live() -> list[dict[str, Any]]:
    return [sync_live(), sync_gmp_live()]


def bootstrap(full: bool = True) -> list[dict[str, Any]]:
    from ..db import init_db
    init_db()
    out = [sync_index(), sync_market(), sync_live(), sync_gmp_live()]
    if full:
        out += [backfill_subscriptions(), backfill_gmp()]
    ensure_demo_ipo()
    return out
