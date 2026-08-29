"""Location normalization and scoring, per docs/P0_5_IMPLEMENTATION.md Phase 11.

AM records carry structured province/जिल्ला/गाउँपालिका fields extracted from the
raw detail-page JSON (see scripts/backfill_locations.py). PM records scraped via
scripts/import_pm_from_list_scrape.py only have a single free-text "found place"
string (भेटिएको ठाउँ) -- there is no detail-page JSON to split it from, so PM has
no structured province/district/municipality of its own. location_score() is
written around that asymmetry: it checks whether AM's structured values appear as
substrings of PM's free-text location before falling back to plain token overlap.

This is intentionally NOT a geocoder: no coordinates, no river/hydrology
reasoning (explicitly out of scope for P0.5), and the province alias table below
is a best-effort convenience for the handful of common spellings this project's
data has been seen to use, not an exhaustive gazetteer. Do not treat it as
validated ground truth -- if it fails to canonicalize a name, callers fall back
to plain string comparison, which still works correctly as long as both AM and PM
values came from the same source site (same spelling for the same place).
"""
from __future__ import annotations

import re
import unicodedata
from typing import TypedDict

# Best-effort aliases for Nepal's 7 provinces (Devanagari names + numbered
# forms seen on the UDB site). Not exhaustive; canonicalization silently
# no-ops for anything not listed here, per the module docstring.
PROVINCE_ALIASES: dict[str, tuple[str, ...]] = {
    "koshi": ("कोशी प्रदेश", "कोशी", "प्रदेश नं १", "प्रदेश न १", "प्रदेश १"),
    "madhesh": ("मधेश प्रदेश", "मधेश", "प्रदेश नं २", "प्रदेश न २", "प्रदेश २"),
    "bagmati": ("बागमती प्रदेश", "बागमती", "प्रदेश नं ३", "प्रदेश न ३", "प्रदेश ३"),
    "gandaki": ("गण्डकी प्रदेश", "गण्डकी", "प्रदेश नं ४", "प्रदेश न ४", "प्रदेश ४"),
    "lumbini": ("लुम्बिनी प्रदेश", "लुम्बिनी", "प्रदेश नं ५", "प्रदेश न ५", "प्रदेश ५"),
    "karnali": ("कर्णाली प्रदेश", "कर्णाली", "प्रदेश नं ६", "प्रदेश न ६", "प्रदेश ६"),
    "sudurpashchim": ("सुदूरपश्चिम प्रदेश", "सुदूरपश्चिम", "प्रदेश नं ७", "प्रदेश न ७", "प्रदेश ७"),
}

_PUNCT_RE = re.compile(r"[:.\-,।/|]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(raw: str | None) -> str:
    """Unicode (NFKC) + punctuation + whitespace cleanup. No transliteration,
    no casing changes to Devanagari (casefold only affects Latin script)."""
    raw = (raw or "")
    raw = unicodedata.normalize("NFKC", raw)
    raw = _PUNCT_RE.sub(" ", raw)
    raw = _WS_RE.sub(" ", raw).strip()
    return raw.casefold()


def canonical_province(value: str | None) -> str | None:
    norm = normalize_text(value)
    if not norm:
        return None
    for canonical, aliases in PROVINCE_ALIASES.items():
        for alias in aliases:
            if normalize_text(alias) in norm:
                return canonical
    return None


class LocationProfile(TypedDict):
    raw_location: str
    normalized_location: str
    province: str
    district: str
    municipality: str
    ward: str


def build_location_profile(record: dict) -> LocationProfile:
    """Reads a record dict's location fields. `province`/`district`/
    `municipality`/`ward` are populated only when the record actually carries
    them (AM, after scripts/backfill_locations.py) -- left "" otherwise, never
    guessed from the free-text location string."""
    raw = record.get("location") or ""
    return {
        "raw_location": raw,
        "normalized_location": normalize_text(raw),
        "province": normalize_text(record.get("province") or ""),
        "district": normalize_text(record.get("district") or ""),
        "municipality": normalize_text(record.get("municipality") or ""),
        "ward": normalize_text(record.get("ward") or ""),
    }


# Provisional engineering scoring tiers -- not scientifically calibrated.
SCORE_SAME_MUNICIPALITY = 1.0
SCORE_SAME_DISTRICT = 0.8
SCORE_SAME_PROVINCE = 0.6
SCORE_TOKEN_OVERLAP_STRONG = 0.75
SCORE_TOKEN_OVERLAP_WEAK = 0.55
SCORE_NEUTRAL_UNKNOWN = 0.5
SCORE_KNOWN_NO_OVERLAP = 0.35
SCORE_DIFFERENT_PROVINCE = 0.25


def location_score(am: dict, pm: dict) -> float:
    a = build_location_profile(am)
    p = build_location_profile(pm)

    for field, score in (
        ("municipality", SCORE_SAME_MUNICIPALITY),
        ("district", SCORE_SAME_DISTRICT),
        ("province", SCORE_SAME_PROVINCE),
    ):
        a_val, p_val = a[field], p[field]
        if a_val and p_val and a_val == p_val:
            return score
        if a_val and not p_val and p["normalized_location"] and a_val in p["normalized_location"]:
            return score
        if p_val and not a_val and a["normalized_location"] and p_val in a["normalized_location"]:
            return score

    if not a["normalized_location"] or not p["normalized_location"]:
        return SCORE_NEUTRAL_UNKNOWN

    a_tokens = set(a["normalized_location"].split())
    p_tokens = set(p["normalized_location"].split())
    overlap = a_tokens & p_tokens
    if len(overlap) >= 2:
        return SCORE_TOKEN_OVERLAP_STRONG
    if len(overlap) == 1:
        return SCORE_TOKEN_OVERLAP_WEAK

    a_prov = canonical_province(a["province"] or a["normalized_location"])
    p_prov = canonical_province(p["province"] or p["normalized_location"])
    if a_prov and p_prov and a_prov != p_prov:
        return SCORE_DIFFERENT_PROVINCE

    return SCORE_KNOWN_NO_OVERLAP
