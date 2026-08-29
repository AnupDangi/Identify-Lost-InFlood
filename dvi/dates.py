"""BS (Bikram Sambat) / AD (Gregorian) date normalization, per
docs/P0_5_IMPLEMENTATION.md Phase 5.

Nepal Police UDB records store event dates ("हराएको मिति" / missing date for AM,
"भेटिएको मिति" / found date for PM) as plain numeric strings with no calendar
marker, and in practice these mix BS and AD depending on how each record/importer
captured them -- scripts/import_pm_from_list_scrape.py in particular pulled PM
dates via a bare `\\d{4}-\\d{2}-\\d{2}` regex with no calendar awareness, so a BS
date like "2082-05-12" was stored looking identical to an AD one. Comparing those
directly (as the original date_score() did) can silently treat a same-day event as
a ~57-year gap, or vice versa.

This module cannot tell BS from AD from formatting alone (both are shown as
YYYY-MM-DD-ish numeric strings). It uses a year-range heuristic instead: BS years
for events in this dataset's plausible era (Nepal Police UDB records, roughly
2000s-2030s AD) fall in ~2057-2100 BS, which never overlaps a plausible AD year for
the same events. This is a provisional engineering heuristic, not a validated
calendar-detection algorithm -- see BS_YEAR_MIN/AD_YEAR_MAX below for the exact
cutoffs and CALENDAR_MARKERS for the (rare) explicit-marker override.
"""
from __future__ import annotations

import re
from datetime import date
from typing import TypedDict

import nepali_datetime as nd

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Provisional, not scientifically validated -- see module docstring. Nepal
# Police UDB records are all recent (this project's scrape covers 2020s-2026
# AD events); a bare 4-digit year at or above BS_YEAR_MIN is treated as BS,
# one at or below AD_YEAR_MAX as AD. There is deliberately no gap between them
# left ambiguous because BS runs ~56-57 years ahead of AD, so for *this*
# dataset's era the two ranges never actually collide.
AD_YEAR_MAX = 2050
BS_YEAR_MIN = 2051

# nepali_datetime's supported BS year range (see its `date.min`/`date.max`).
BS_YEAR_FLOOR = 1975
BS_YEAR_CEIL = 2100

CALENDAR_MARKERS = {
    "BS": ("बि.सं", "वि.सं", "बिस", "b.s", "bs "),
    "AD": ("ई.सं", "ईस्वी", "a.d", "ad "),
}


class NormalizedDate(TypedDict):
    raw_event_date: str
    calendar_type: str  # "BS" | "AD" | "unknown"
    event_date_normalized: str  # ISO YYYY-MM-DD (Gregorian), or "" if unparseable


def _to_latin_digits(s: str) -> str:
    return s.translate(DEVANAGARI_DIGITS)


def _detect_marker(raw_lower: str) -> str | None:
    for calendar, markers in CALENDAR_MARKERS.items():
        if any(m in raw_lower for m in markers):
            return calendar
    return None


def normalize_date(raw: str | None) -> NormalizedDate:
    """Best-effort BS/AD detection + conversion to a Gregorian ISO date.

    Never raises. Unparseable or out-of-range input returns calendar_type
    "unknown" and event_date_normalized "" -- callers (dvi.scoring.date_score)
    must treat that as neutral, not as a penalty, per the "don't hard-filter
    on unknown metadata" rule.
    """
    raw = (raw or "").strip()
    result: NormalizedDate = {
        "raw_event_date": raw, "calendar_type": "unknown", "event_date_normalized": "",
    }
    if not raw:
        return result

    cleaned = _to_latin_digits(raw)
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", cleaned)
    if not m:
        return result
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))

    forced = _detect_marker(raw.lower())

    if forced == "AD" or (forced is None and year <= AD_YEAR_MAX):
        try:
            g = date(year, month, day)
        except ValueError:
            return result
        result["calendar_type"] = "AD"
        result["event_date_normalized"] = g.isoformat()
        return result

    if forced == "BS" or (forced is None and year >= BS_YEAR_MIN):
        if not (BS_YEAR_FLOOR <= year <= BS_YEAR_CEIL):
            return result
        try:
            b = nd.date(year, month, day)
        except ValueError:
            return result
        result["calendar_type"] = "BS"
        result["event_date_normalized"] = b.to_datetime_date().isoformat()
        return result

    return result


def get_normalized(record: dict) -> NormalizedDate:
    """Read a record's precomputed raw_event_date/calendar_type/
    event_date_normalized columns if present (populated by
    scripts/normalize_dates.py or scrape_udb.py's normalize()); otherwise
    compute on the fly from `event_date`. Works whether or not the backfill
    script has been run."""
    normalized = record.get("event_date_normalized")
    calendar_type = record.get("calendar_type")
    if normalized:
        return {
            "raw_event_date": record.get("raw_event_date") or record.get("event_date") or "",
            "calendar_type": calendar_type or "unknown",
            "event_date_normalized": normalized,
        }
    raw = record.get("event_date", "")
    return normalize_date(raw)
