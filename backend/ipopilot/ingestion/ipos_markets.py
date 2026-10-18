"""ipos.markets — free, keyless public API re-publishing official NSE/BSE/SEBI IPO data.

Docs: https://ipos.markets/api-docs   Datasets: https://ipos.markets/data
  * /data/issues_index.json      every issue since the 1990s (dates, band, price, size, listing gain)
  * /data/overview.json          open / closed / upcoming / recent issues
  * /api/v1/subscription         index of issues with subscription data
  * /api/v1/subscription/<sym>   category-wise (QIB / NII / RII / EMP / SH) offered vs bid
"""
from __future__ import annotations

import logging
from typing import Any

from .http import client

log = logging.getLogger(__name__)
BASE = "https://ipos.markets"


def fetch_issues_index() -> list[dict[str, Any]]:
    d = client().get_json(f"{BASE}/data/issues_index.json")
    return (d or {}).get("issues", [])


def fetch_overview() -> dict[str, Any]:
    return client().get_json(f"{BASE}/data/overview.json") or {}


def fetch_subscription_index() -> list[dict[str, Any]]:
    d = client().get_json(f"{BASE}/api/v1/subscription")
    return (d or {}).get("issues", [])


def fetch_subscription(key: str, *, historical: bool = False) -> dict[str, Any] | None:
    """Per-issue category-wise subscription. Final figures of listed issues never change,
    so they are cached forever on disk when `historical` is set."""
    return client().get_json(
        f"{BASE}/api/v1/subscription/{key}.json",
        cache_ttl=float("inf") if historical else None,
    )


def category_times(sub: dict[str, Any]) -> dict[str, float | None]:
    cats = sub.get("categories") or {}

    def g(name: str, field: str = "times"):
        c = cats.get(name) or {}
        v = c.get(field)
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "total_times": sub.get("total_times") if sub.get("total_times") is not None else sub.get("lower_bound_times"),
        "qib_times": g("QIB"),
        "nii_times": g("NII"),
        "rii_times": g("RII") if g("RII") is not None else g("II"),  # SME post-2025 uses "Individual Investors"
        "emp_times": g("EMP"),
        "rii_offered": g("RII", "offered") or g("II", "offered"),
        "rii_bid": g("RII", "bid") or g("II", "bid"),
        "rii_applications": g("RII", "applications") or g("II", "applications"),
    }
