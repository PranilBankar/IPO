"""Point-in-time feature engineering for the listing-return model.

Decision time t* = the issue's subscription CLOSE date (the last moment an investor can
still apply). Every feature uses only information observable on or before t*:
  * issue terms (board, price, size, anchor share, sector)
  * category-wise subscription (final figures at close; at inference during the bid window
    these are replaced by a forecast — see engine/subscription.py)
  * GMP observations dated <= t*
  * market regime from index closes strictly before t*
  * IPO-market "heat": listing gains of issues that LISTED before t*
  * FinBERT sentiment of headlines published <= t*
The label is the listing-day return  y = log(1 + gain%/100).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import GmpObservation, Ipo, MarketBar, NewsItem, SubscriptionSnapshot

NUMERIC_FEATURES = [
    "log_price", "log_size", "size_missing", "anchor_pct", "anchor_missing",
    "log_sub_total", "log_sub_qib", "log_sub_nii", "log_sub_rii", "sub_missing", "cat_missing",
    "gmp_last_pct", "gmp_max_pct", "gmp_trend_pct", "gmp_n_obs", "gmp_missing",
    "nifty_ret20", "nifty_ret60", "nifty_vol20", "vix", "smallcap_ret20", "market_missing",
    "heat_board10", "heat_all20", "heat_std10", "n_ipos_30d", "heat_missing",
    "sent_mean", "sent_neg_share", "news_count", "sent_missing",
    "is_sme", "post_2025_sme_rules", "days_open",
]
CATEGORICAL = ["board", "sector"]
SECTOR_TOP = 24


@dataclass
class RawTables:
    ipos: pd.DataFrame
    subs: pd.DataFrame
    gmp: pd.DataFrame
    bars: pd.DataFrame
    news: pd.DataFrame


def load_tables(s: Session) -> RawTables:
    ipos = pd.DataFrame([{
        "slug": i.slug, "name": i.name, "symbol": i.symbol, "board": i.board, "stage": i.stage,
        "open_date": i.open_date, "close_date": i.close_date, "listing_date": i.listing_date,
        "price": i.price, "issue_size_cr": i.issue_size_cr, "anchor_pct": i.anchor_pct, "sector": i.sector,
        "listing_gain_pct": i.listing_gain_pct, "is_demo": i.is_demo, "partly_paid": i.partly_paid,
        "sub_total_index": (i.extra or {}).get("sub_total_index"), "lot_size": i.lot_size,
    } for i in s.scalars(select(Ipo))])
    subs = pd.DataFrame([{
        "slug": r.ipo_slug, "source": r.source, "as_of": r.as_of, "final": r.final, "total": r.total_times,
        "qib": r.qib_times, "nii": r.nii_times, "rii": r.rii_times,
    } for r in s.scalars(select(SubscriptionSnapshot))])
    gmp = pd.DataFrame([{
        "slug": g.ipo_slug, "source": g.source, "date": g.obs_date, "gmp_pct": g.gmp_pct,
    } for g in s.scalars(select(GmpObservation)) if g.gmp_pct is not None])
    bars = pd.DataFrame([{"ticker": b.ticker, "date": b.date, "close": b.close} for b in s.scalars(select(MarketBar))])
    news = pd.DataFrame([{
        "slug": n.ipo_slug, "published_at": n.published_at, "p_pos": n.p_pos, "p_neg": n.p_neg,
    } for n in s.scalars(select(NewsItem)) if n.p_pos is not None])
    return RawTables(ipos, subs, gmp, bars, news)


def _ln1p(x):
    return math.log1p(x) if x is not None and not pd.isna(x) and x >= 0 else np.nan


class FeatureBuilder:
    def __init__(self, t: RawTables):
        self.t = t
        self.market = {}
        if not t.bars.empty:
            for tk, g in t.bars.groupby("ticker"):
                ser = g.sort_values("date").set_index("date")["close"]
                ser.index = pd.to_datetime(ser.index)
                self.market[tk] = ser
        listed = t.ipos[t.ipos["listing_gain_pct"].notna() & t.ipos["listing_date"].notna() & ~t.ipos["is_demo"]]
        self.listed = listed.assign(listing_date=pd.to_datetime(listed["listing_date"])).sort_values("listing_date")
        self.closes = pd.to_datetime(t.ipos["close_date"].dropna()).sort_values().values
        self.subs_by = {k: g for k, g in t.subs.groupby("slug")} if not t.subs.empty else {}
        self.gmp_by = {k: g for k, g in t.gmp.groupby("slug")} if not t.gmp.empty else {}
        self.news_by = {k: g for k, g in t.news.groupby("slug")} if not t.news.empty else {}
        top = t.ipos["sector"].dropna().value_counts().head(SECTOR_TOP).index.tolist()
        self.sectors = top

    # ---- components -------------------------------------------------------
    def _market(self, when: pd.Timestamp) -> dict:
        out = {}
        ok = True
        for tk, n20, key20, key60 in (("NIFTY", 20, "nifty_ret20", "nifty_ret60"), ("SMALLCAP", 20, "smallcap_ret20", None)):
            ser = self.market.get(tk)
            if ser is None:
                ok = False
                continue
            hist = ser[ser.index < when]
            if len(hist) < 61:
                ok = False
                continue
            out[key20] = float(hist.iloc[-1] / hist.iloc[-21] - 1)
            if key60:
                out[key60] = float(hist.iloc[-1] / hist.iloc[-61] - 1)
                r = np.log(hist.iloc[-21:]).diff().dropna()
                out["nifty_vol20"] = float(r.std() * math.sqrt(252))
        vix = self.market.get("INDIAVIX")
        if vix is not None:
            h = vix[vix.index < when]
            out["vix"] = float(h.iloc[-1]) if len(h) else np.nan
        out["market_missing"] = 0.0 if ok and "vix" in out else 1.0
        return out

    def _heat(self, when: pd.Timestamp, board: str) -> dict:
        prior = self.listed[self.listed["listing_date"] < when]
        if len(prior) < 5:
            return {"heat_missing": 1.0}
        same = prior[prior["board"] == board].tail(10)["listing_gain_pct"]
        allp = prior.tail(20)["listing_gain_pct"]
        lo = np.searchsorted(self.closes, np.datetime64(when - pd.Timedelta(days=30)))
        hi = np.searchsorted(self.closes, np.datetime64(when))
        return {
            "heat_board10": float(np.clip(same.mean(), -50, 150)) / 100 if len(same) else np.nan,
            "heat_all20": float(np.clip(allp.mean(), -50, 150)) / 100,
            "heat_std10": float(same.std()) / 100 if len(same) > 2 else np.nan,
            "n_ipos_30d": float(hi - lo),
            "heat_missing": 0.0,
        }

    def _subs(self, slug: str, when: pd.Timestamp, idx_total, override: dict | None) -> dict:
        if override:
            tot, q, n, r = (override.get(k) for k in ("total", "qib", "nii", "rii"))
        else:
            g = self.subs_by.get(slug)
            tot = q = n = r = None
            if g is not None and len(g):
                g = g.sort_values(["final", "as_of"])  # prefer final, then latest
                row = g.iloc[-1]
                tot, q, n, r = row["total"], row["qib"], row["nii"], row["rii"]
            if tot is None or pd.isna(tot):
                tot = idx_total
        f = {"log_sub_total": _ln1p(tot), "log_sub_qib": _ln1p(q), "log_sub_nii": _ln1p(n), "log_sub_rii": _ln1p(r)}
        f["sub_missing"] = 1.0 if np.isnan(f["log_sub_total"]) else 0.0
        f["cat_missing"] = 1.0 if all(np.isnan(f[k]) for k in ("log_sub_qib", "log_sub_nii", "log_sub_rii")) else 0.0
        return f

    def _gmp(self, slug: str, when: pd.Timestamp, override: dict | None) -> dict:
        if override and override.get("gmp_pct") is not None:
            v = float(override["gmp_pct"]) / 100
            return {"gmp_last_pct": v, "gmp_max_pct": v, "gmp_trend_pct": 0.0, "gmp_n_obs": 1.0, "gmp_missing": 0.0}
        g = self.gmp_by.get(slug)
        if g is None:
            return {"gmp_missing": 1.0}
        g = g[pd.to_datetime(g["date"]) <= when]
        if g.empty:
            return {"gmp_missing": 1.0}
        daily = g.groupby("date")["gmp_pct"].mean().sort_index()  # average sources per day
        v = daily.clip(-60, 300) / 100
        return {"gmp_last_pct": float(v.iloc[-1]), "gmp_max_pct": float(v.max()),
                "gmp_trend_pct": float(v.iloc[-1] - v.iloc[0]), "gmp_n_obs": float(min(len(v), 15)),
                "gmp_missing": 0.0}

    def _news(self, slug: str, when: pd.Timestamp) -> dict:
        g = self.news_by.get(slug)
        if g is None:
            return {"sent_missing": 1.0, "news_count": 0.0}
        g = g[pd.to_datetime(g["published_at"]) <= when + pd.Timedelta(hours=17)]
        if g.empty:
            return {"sent_missing": 1.0, "news_count": 0.0}
        net = g["p_pos"] - g["p_neg"]
        return {"sent_mean": float(net.mean()), "sent_neg_share": float((g["p_neg"] > 0.5).mean()),
                "news_count": float(min(len(g), 30)), "sent_missing": 0.0}

    # ---- public -----------------------------------------------------------
    def row(self, ipo: dict, *, sub_override: dict | None = None, gmp_override: dict | None = None,
            when: dt.date | None = None) -> dict:
        close = when or ipo.get("close_date") or dt.date.today()
        whenT = pd.Timestamp(close)
        price = ipo.get("price")
        size = ipo.get("issue_size_cr")
        anchor = ipo.get("anchor_pct")
        is_sme = 1.0 if ipo.get("board") == "sme" else 0.0
        open_d = ipo.get("open_date")
        f = {
            "slug": ipo["slug"], "board": ipo.get("board") or "mainboard",
            "sector": ipo.get("sector") if ipo.get("sector") in self.sectors else "unknown",
            "close_date": close,
            "log_price": math.log(price) if price and price > 0 else np.nan,
            "log_size": math.log1p(size) if size and size > 0 else np.nan,
            "size_missing": 0.0 if size and size > 0 else 1.0,
            "anchor_pct": (anchor / 100) if anchor is not None and not pd.isna(anchor) else np.nan,
            "anchor_missing": 0.0 if anchor is not None and not pd.isna(anchor) else 1.0,
            "is_sme": is_sme,
            "post_2025_sme_rules": 1.0 if is_sme and close >= dt.date(2025, 7, 1) else 0.0,
            "days_open": float((close - open_d).days) if open_d else np.nan,
        }
        f.update(self._subs(ipo["slug"], whenT, ipo.get("sub_total_index"), sub_override))
        f.update(self._gmp(ipo["slug"], whenT, gmp_override))
        f.update(self._market(whenT))
        f.update(self._heat(whenT, f["board"]))
        f.update(self._news(ipo["slug"], whenT))
        for k in NUMERIC_FEATURES:
            f.setdefault(k, np.nan)
        return f

    def training_frame(self, min_year: int = 2012) -> pd.DataFrame:
        df = self.t.ipos
        df = df[df["listing_gain_pct"].notna() & df["close_date"].notna() & df["price"].notna() & ~df["is_demo"]
                & ~df["partly_paid"].fillna(False)]
        df = df[pd.to_datetime(df["close_date"]).dt.year >= min_year]
        rows = []
        for ipo in df.to_dict("records"):
            r = self.row(ipo)
            g = float(np.clip(ipo["listing_gain_pct"], -90, 400))
            r["gain_pct"] = g
            r["y"] = math.log1p(g / 100)
            rows.append(r)
        out = pd.DataFrame(rows).sort_values("close_date").reset_index(drop=True)
        return out
