"""Parsing helpers shared by the ingestion modules."""
from __future__ import annotations

import datetime as dt
import re
from difflib import SequenceMatcher

_STOP = {
    "limited", "ltd", "ltd.", "pvt", "private", "the", "ipo", "india", "industries", "company", "co",
    "corporation", "corp", "and", "&", "of", "sme", "mainboard", "upcoming", "open", "closed", "listed",
}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def norm_name(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower().replace("&amp;", "&")
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks)


def name_similarity(a: str, b: str) -> float:
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    jacc = len(ta & tb) / max(1, len(ta | tb))
    seq = SequenceMatcher(None, na, nb).ratio()
    prefix = 0.1 if na.split()[0] == nb.split()[0] else 0.0
    return max(jacc, seq) * 0.9 + prefix


def parse_date(s: str | None) -> dt.date | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("-", "", "NA", "N/A"):
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(s.title() if "%b" in fmt or "%B" in fmt else s, fmt).date()
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def parse_day_month(s: str, year_hint: int, ref: dt.date | None = None) -> dt.date | None:
    """Parse '16 October' / '6 May' (no year). Picks the year that puts the date closest to `ref`."""
    m = re.match(r"\s*(\d{1,2})\s+([A-Za-z]+)", s or "")
    if not m:
        return None
    day, mon = int(m.group(1)), _MONTHS.get(m.group(2)[:3].lower())
    if not mon:
        return None
    best = None
    for y in (year_hint - 1, year_hint, year_hint + 1):
        try:
            d = dt.date(y, mon, day)
        except ValueError:
            continue
        if ref is None:
            ref = dt.date(year_hint, 6, 30)
        if best is None or abs((d - ref).days) < abs((best - ref).days):
            best = d
    return best


def parse_datetime(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d.replace(tzinfo=None) if d.tzinfo is None else d.astimezone(dt.timezone.utc).replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in ("%b %d %Y %I:%M%p", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def to_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "").strip()
    s = s.replace("+-", "").replace("+", "")
    m = re.search(r"-?\d+(\.\d+)?(E[+-]?\d+)?", s, flags=re.I)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_band(band: str | None) -> tuple[float | None, float | None]:
    """'558.00 - 586.00', 'Rs.77 to Rs.82', '25-27.5', '402' -> (low, high)."""
    if not band:
        return None, None
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", band.replace(",", ""))]
    if not nums:
        return None, None
    return min(nums), max(nums)


def slugify(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
