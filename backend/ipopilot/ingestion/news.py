"""Company / IPO news headlines from Google News RSS (India edition)."""
from __future__ import annotations

import datetime as dt
import email.utils
import logging
import re
from urllib.parse import quote_plus

import feedparser

from .http import client
from .util import norm_name

log = logging.getLogger(__name__)


def _query_name(name: str) -> str:
    n = re.sub(r"\b(limited|ltd\.?)\b", "", name, flags=re.I).strip()
    return re.sub(r"\s+", " ", n)


def fetch_news(name: str, max_items: int = 40) -> list[dict]:
    q = f'"{_query_name(name)}" IPO'
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"
    text = client().get_text(url)
    if not text:
        return []
    feed = feedparser.parse(text)
    key_tokens = set(norm_name(name).split()[:2])
    out = []
    for e in feed.entries[:max_items]:
        title = (e.get("title") or "").strip()
        if key_tokens and not key_tokens & set(norm_name(title).split()):
            continue  # filter results not actually about this company
        pub = None
        if e.get("published"):
            try:
                pub = email.utils.parsedate_to_datetime(e["published"]).astimezone(dt.timezone.utc).replace(tzinfo=None)
            except (TypeError, ValueError):
                pub = None
        src = e.get("source", {}).get("title") if isinstance(e.get("source"), dict) else None
        # Google appends " - Publisher" to titles
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        out.append({"title": title, "url": e.get("link"), "published_at": pub, "publisher": src})
    return out
