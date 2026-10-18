"""Grey Market Premium (GMP) collectors.

GMP is UNOFFICIAL: an indication of the premium at which IPO shares trade informally before
listing. It is not an exchange price and may be stale, inconsistent across sources, or
manipulated. IPOPilot therefore stores every observation with source + date and never
overwrites history, and the model learns how much to trust it.

Sources
  * IPOWatch      live table  https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/
                  per-IPO day-wise history pages (used for historical backfill)
  * MarketLiveIPOGMP  https://marketliveipogmp.com/  (per-IPO page meta: GMP, price, lot)
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from .http import client
from .util import parse_day_month, to_float

log = logging.getLogger(__name__)

IPOWATCH_LIVE = "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"
IPOWATCH_SITEMAPS = [f"https://ipowatch.in/post-sitemap{'' if i == 1 else i}.xml" for i in range(1, 16)]
MLG_HOME = "https://marketliveipogmp.com/"


# --------------------------------------------------------------------------- IPOWatch
def _clean_company(cell: str) -> tuple[str, str | None]:
    status = None
    for st in ("Upcoming", "Open", "Closed", "Listed", "Allotment"):
        if cell.endswith(" " + st) or cell.endswith(st):
            status = st.lower()
            cell = cell[: -len(st)].strip()
            break
    return cell.strip(), status


def fetch_ipowatch_live() -> list[dict[str, Any]]:
    html = client().get_text(IPOWATCH_LIVE)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, Any]] = []
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        if not rows:
            continue
        head = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])]
        if not head or "Company" not in head[0] or not any("GMP" in h for h in head):
            continue
        idx_gmp = next(i for i, h in enumerate(head) if "GMP" in h)
        idx_price = next((i for i, h in enumerate(head) if "Price" in h), None)
        for r in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
            if len(cells) <= idx_gmp:
                continue
            a = r.find("a", href=True)
            name, status = _clean_company(cells[0])
            gmp = to_float(cells[idx_gmp]) if "-" != cells[idx_gmp].replace("₹", "").strip() else None
            price = to_float(cells[idx_price]) if idx_price is not None and idx_price < len(cells) else None
            out.append({
                "source": "ipowatch", "name": name, "status": status, "gmp_rs": gmp, "ref_price": price,
                "gmp_pct": (100.0 * gmp / price) if gmp is not None and price else None,
                "obs_date": dt.date.today(), "url": (a["href"] if a else None),
                "dates_text": cells[-1] if cells else None,
            })
    return out


def list_ipowatch_gmp_pages() -> list[str]:
    urls: list[str] = []
    for sm in IPOWATCH_SITEMAPS:
        xml = client().get_text(sm, cache_ttl=7 * 86400)
        if not xml:
            break
        urls += [u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if "gmp" in u or "grey-market" in u]
    return urls


def page_name_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.split(r"-ipo-(gmp|grey)", slug)[0]
    return slug.replace("-", " ")


def fetch_ipowatch_history(url: str, year_hint: int, ref_date: dt.date | None, *,
                           historical: bool = True) -> list[dict[str, Any]]:
    """Parse the day-wise GMP table of an IPOWatch per-IPO page."""
    html = client().get_text(url, cache_ttl=float("inf") if historical else 3600)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        if not rows:
            continue
        head = [c.get_text(" ", strip=True) for c in rows[0].find_all(["td", "th"])]
        if len(head) < 2 or "Date" not in head[0] or "GMP" not in head[1]:
            continue
        for r in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            d = parse_day_month(cells[0], year_hint, ref_date)
            raw = cells[1].replace("₹", "").replace("+-", "").strip()
            if d is None or raw in ("-", "", "–"):
                continue
            g = to_float(raw)
            if g is None:
                continue
            out.append({"source": "ipowatch", "obs_date": d, "gmp_rs": g, "raw": {"row": cells, "url": url}})
        if out:
            break
    return out


# --------------------------------------------------------------------------- MarketLiveIPOGMP
_MLG_META = re.compile(
    r"GMP today is ₹\s*(-?[\d,.]+)\s*\(([-+]?[\d.]+)%\)\.\s*Price ₹\s*([\d,.]+)\.\s*Lot\s*([\d,]+)\s*shares", re.I)


def list_marketlive_pages() -> list[dict[str, str]]:
    html = client().get_text(MLG_HOME)
    if not html:
        return []
    items = re.findall(r'"name":"([^"]+?) IPO GMP","url":"(https://marketliveipogmp\.com/ipo/[^"]+)"', html)
    seen, out = set(), []
    for name, url in items:
        if url not in seen:
            seen.add(url)
            out.append({"name": name, "url": url})
    return out


def fetch_marketlive_page(url: str) -> dict[str, Any] | None:
    html = client().get_text(url)
    if not html:
        return None
    m = _MLG_META.search(html)
    if not m:
        return None
    gmp, pct, price, lot = (to_float(m.group(i)) for i in range(1, 5))
    mod = re.search(r'"dateModified":"([^"]+)"', html)
    obs = dt.date.today()
    if mod:
        try:
            obs = dt.datetime.fromisoformat(mod.group(1)).date()
        except ValueError:
            pass
    return {"source": "marketliveipogmp", "gmp_rs": gmp, "gmp_pct": pct, "ref_price": price,
            "lot_size": int(lot) if lot else None, "obs_date": obs, "url": url}
