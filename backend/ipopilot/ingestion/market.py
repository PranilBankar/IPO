"""Broad-market context: daily closes of NIFTY 50, India VIX and NIFTY Smallcap 250
(Yahoo Finance chart API). Used for leakage-free market-regime features."""
from __future__ import annotations

import datetime as dt
import logging

from .http import client

log = logging.getLogger(__name__)

TICKERS = {"NIFTY": "^NSEI", "INDIAVIX": "^INDIAVIX", "SMALLCAP": "NIFTYSMLCAP250.NS"}


def fetch_daily(yahoo_symbol: str, start: dt.date = dt.date(2005, 1, 1)) -> list[tuple[dt.date, float]]:
    # explicit period1/period2: Yahoo silently down-samples `range=max` to weekly/monthly bars
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    p1 = int(dt.datetime(start.year, start.month, start.day).timestamp())
    p2 = int(dt.datetime.now().timestamp()) + 86400
    d = client().get_json(url, params={"period1": p1, "period2": p2, "interval": "1d"})
    try:
        res = d["chart"]["result"][0]
        ts = res["timestamp"]
        closes = res["indicators"]["quote"][0]["close"]
    except (TypeError, KeyError, IndexError):
        log.warning("No market data for %s", yahoo_symbol)
        return []
    out = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        # timestamps are market-open in IST; shift to IST date
        out.append(((dt.datetime.utcfromtimestamp(t) + dt.timedelta(hours=5, minutes=30)).date(), float(c)))
    return out
