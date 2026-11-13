"""Polite HTTP client shared by all scrapers: per-host rate limiting, retries with
backoff, NSE cookie warm-up, and an optional on-disk cache for historical backfills."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ..config import settings

log = logging.getLogger(__name__)


class HttpClient:
    def __init__(self, min_interval: float | None = None, cache_dir: Path | None = None):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": settings.user_agent,
            "Accept-Language": "en-IN,en;q=0.9",
            "Accept": "text/html,application/json,application/xhtml+xml,*/*;q=0.8",
        })
        self.min_interval = settings.min_request_interval if min_interval is None else min_interval
        self.cache_dir = cache_dir or (settings.raw_dir / "http_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._nse_warm = 0.0

    def _throttle(self, host: str) -> None:
        with self._lock:
            wait = self._last.get(host, 0) + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def _warm_nse(self) -> None:
        # NSE's JSON API works best with the cookies set by a page visit; refresh every 10 min
        if time.monotonic() - self._nse_warm < 600:
            return
        try:
            self.s.get("https://www.nseindia.com/market-data/all-upcoming-issues-ipo", timeout=settings.http_timeout)
        except requests.RequestException:
            pass
        self._nse_warm = time.monotonic()

    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None,
            retries: int = 3, cache_ttl: float | None = None) -> requests.Response | None:
        """GET with retries. If cache_ttl (seconds) is given, a cached body younger than the
        TTL is served from disk (use float('inf') for immutable historical pages)."""
        key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        cpath = self.cache_dir / f"{key}.bin"
        if cache_ttl is not None and cpath.exists() and time.time() - cpath.stat().st_mtime < cache_ttl:
            r = requests.Response()
            r.status_code, r._content, r.url = 200, cpath.read_bytes(), url
            r.encoding = "utf-8"
            return r

        host = urlparse(url).netloc
        if "nseindia.com" in host:
            self._warm_nse()
        hdrs = dict(headers or {})
        if "nseindia.com" in host:
            hdrs.setdefault("Referer", "https://www.nseindia.com/market-data/all-upcoming-issues-ipo")
        for attempt in range(retries):
            self._throttle(host)
            try:
                r = self.s.get(url, params=params, headers=hdrs, timeout=settings.http_timeout)
            except requests.RequestException as e:
                log.warning("GET %s failed (%s), attempt %d", url, e, attempt + 1)
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                if cache_ttl is not None:
                    cpath.write_bytes(r.content)
                return r
            if r.status_code in (404, 410):
                return r
            if r.status_code in (401, 403) and "nseindia.com" in host:
                self._nse_warm = 0.0
                self._warm_nse()
            log.warning("GET %s -> %s, attempt %d", url, r.status_code, attempt + 1)
            time.sleep(1.5 * (attempt + 1))
        return None

    def get_json(self, url: str, **kw: Any) -> Any:
        r = self.get(url, **kw)
        if r is None or r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            try:
                return json.loads(r.content.decode("utf-8", errors="ignore"))
            except ValueError:
                log.warning("Non-JSON response from %s", url)
                return None

    def get_text(self, url: str, **kw: Any) -> str | None:
        r = self.get(url, **kw)
        if r is None or r.status_code != 200:
            return None
        return r.content.decode("utf-8", errors="ignore")


_client: HttpClient | None = None


def client() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient()
    return _client
