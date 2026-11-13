"""NSE India public IPO endpoints (the official reference source).

  /api/ipo-current-issue                  currently open issues + overall subscription
  /api/all-upcoming-issues?category=ipo   upcoming / active issues
  /api/public-past-issues                 every past issue with listing date
  /api/ipo-detail?symbol=X&series=EQ|SME  issue info (bid lot, price band, registrar) and
                                          category-wise subscription (QIB, sNII, bNII, RII, EMP)
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .http import client
from .util import to_float

log = logging.getLogger(__name__)
BASE = "https://www.nseindia.com/api"


def fetch_current() -> list[dict[str, Any]]:
    return client().get_json(f"{BASE}/ipo-current-issue") or []


def fetch_upcoming() -> list[dict[str, Any]]:
    return client().get_json(f"{BASE}/all-upcoming-issues", params={"category": "ipo"}) or []


def fetch_past() -> list[dict[str, Any]]:
    return client().get_json(f"{BASE}/public-past-issues") or []


def fetch_detail(symbol: str, series: str = "EQ") -> dict[str, Any] | None:
    d = client().get_json(f"{BASE}/ipo-detail", params={"symbol": symbol, "series": series})
    if not d or not isinstance(d, dict):
        return None
    return d


def _num(v) -> float | None:
    f = to_float(v)
    return f if f is not None else None


def parse_detail(d: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields IPOPilot needs from an ipo-detail payload."""
    out: dict[str, Any] = {}
    info = {(x.get("title") or "").strip(): (x.get("value") or "") for x in (d.get("issueInfo") or {}).get("dataList", [])}
    lot = info.get("Bid Lot") or info.get("Minimum Order Quantity") or ""
    m = re.search(r"([\d,]+)\s*Equity", lot)
    if m:
        out["lot_size"] = int(m.group(1).replace(",", ""))
    out["registrar"] = (info.get("Name of the Registrar") or "").strip() or None
    out["price_range_text"] = info.get("Price Range")
    out["issue_size_text"] = (info.get("Issue Size") or "").strip('"')[:600] or None
    out["issue_period"] = info.get("Issue Period")
    out["retail_max_amount"] = to_float(info.get("Maximum Subscription Amount for Retail Investor"))

    rows = (d.get("activeCat") or {}).get("dataList") or d.get("bidDetails") or []
    cats: dict[str, dict[str, float | None]] = {}
    for r in rows:
        name = (r.get("category") or "").strip()
        offered = _num(r.get("noOfShareOffered") or r.get("noOfSharesOffered"))
        bid = _num(r.get("noOfSharesBid") or r.get("noOfsharesBid"))
        times = _num(r.get("noOfTotalMeant") or r.get("noOfTime"))
        if name.startswith("Qualified Institutional"):
            key = "QIB"
        elif name == "Non Institutional Investors":
            key = "NII"
        elif "more than Ten Lakh" in name:
            key = "BNII"
        elif "Two Lakh" in name:
            key = "SNII"
        elif name.startswith("Retail Individual") or name.startswith("Individual Investors"):
            key = "RII"
        elif name.startswith("Employee"):
            key = "EMP"
        elif name == "Total":
            key = "TOTAL"
        else:
            continue
        if offered in (None, 0.0) and key != "TOTAL":
            continue  # category not applicable / data purged after close
        cats[key] = {"offered": offered, "bid": bid, "times": times}
    out["categories"] = cats
    out["as_of"] = None
    for k in ("demandDataNSE", "demandDataBSE"):
        rows = d.get(k) or []
        if rows and rows[0].get("timestamp"):
            out["as_of"] = rows[0]["timestamp"]
            break
    return out
