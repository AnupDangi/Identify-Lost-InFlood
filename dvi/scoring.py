"""Centralized AM<->PM metadata/fusion scoring, per
docs/P0_5_IMPLEMENTATION.md Phase 12.

Previously scripts/compare_am_pm.py defined sex/age/height/date/location scoring
and scripts/search_candidates.py imported it directly (a single source, but living
inside a "metadata-only comparison" script rather than a shared module). This
module is that shared home; compare_am_pm.py and search_candidates.py both import
from here now.

IMPORTANT: WEIGHTS and the score bands below are heuristic ranking features
carried over from the P0 prototype, not calibrated probabilities and not
validated against any ground-truth match dataset. Do not present final_score /
metadata_score to a reviewer as an identity confidence -- see
scripts/evaluate_retrieval.py for how these should eventually be validated, and
main.py / web/index.html for the disclaimers shown to reviewers.
"""
from __future__ import annotations

from dvi.dates import get_normalized
from dvi.location import location_score as _location_score


def to_float(v) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def band_score(diff: float | None, bands: list[tuple[float, float]]) -> float:
    """bands: list of (max_diff, score), checked in order; last is fallback."""
    if diff is None:
        return 0.5  # neutral when unknown -- never hard-filter on missing metadata
    for max_diff, score in bands:
        if diff <= max_diff:
            return score
    return bands[-1][1]


def sex_score(am: dict, pm: dict) -> float:
    """Blends scraped metadata sex with InsightFace-detected sex (when present
    as `detected_sex` on the record dict). Each available signal contributes
    equally; a soft cross-check, never a hard filter."""
    signals = []
    a_meta, p_meta = (am.get("sex") or "").lower(), (pm.get("sex") or "").lower()
    if a_meta and p_meta:
        signals.append(1.0 if a_meta == p_meta else 0.0)
    a_det, p_det = (am.get("detected_sex") or "").lower(), (pm.get("detected_sex") or "").lower()
    if a_det and p_det:
        signals.append(1.0 if a_det == p_det else 0.0)
    if not signals:
        return 0.5
    return sum(signals) / len(signals)


def age_score(am: dict, pm: dict) -> float:
    a_lo, a_hi = to_float(am.get("age_min")), to_float(am.get("age_max"))
    p_lo, p_hi = to_float(pm.get("age_min")), to_float(pm.get("age_max"))
    if a_lo is None or p_lo is None:
        return 0.5
    a_mid = (a_lo + (a_hi or a_lo)) / 2
    p_mid = (p_lo + (p_hi or p_lo)) / 2
    return band_score(abs(a_mid - p_mid), [(5, 1.0), (10, 0.7), (20, 0.4), (999, 0.1)])


def height_score(am: dict, pm: dict) -> float:
    a, p = to_float(am.get("height_cm")), to_float(pm.get("height_cm"))
    if a is None or p is None:
        return 0.5
    return band_score(abs(a - p), [(5, 1.0), (10, 0.7), (20, 0.4), (999, 0.1)])


def date_score(am: dict, pm: dict) -> float:
    """Compares normalized (Gregorian) event dates -- see dvi.dates -- instead
    of raw strings, so a BS-dated AM record and an AD-dated PM record are not
    scored as if decades apart. Falls back to neutral when either side can't be
    normalized (unparseable, or genuinely calendar-ambiguous)."""
    a = get_normalized(am)
    p = get_normalized(pm)
    if not a["event_date_normalized"] or not p["event_date_normalized"]:
        return 0.5
    from datetime import date as _date
    a_date = _date.fromisoformat(a["event_date_normalized"])
    p_date = _date.fromisoformat(p["event_date_normalized"])
    gap = (p_date - a_date).days
    if gap < 0:
        return 0.2  # PM found before AM reported missing -> implausible but not impossible (reporting lag)
    return band_score(gap, [(10, 1.0), (30, 0.7), (90, 0.4), (99999, 0.2)])


def location_score(am: dict, pm: dict) -> float:
    return _location_score(am, pm)


# Heuristic weights, NOT calibrated probabilities. Carried over from the P0
# prototype; revisit once scripts/evaluate_retrieval.py has ground-truth-backed
# Recall@K to tune against.
WEIGHTS = {
    "sex": 0.30,
    "age": 0.25,
    "height": 0.10,
    "date": 0.15,
    "location": 0.20,
}


def metadata_score(am: dict, pm: dict) -> dict:
    """Returns each component score plus the weighted metadata_score."""
    components = {
        "sex_score": sex_score(am, pm),
        "age_score": age_score(am, pm),
        "height_score": height_score(am, pm),
        "date_score": date_score(am, pm),
        "location_score": location_score(am, pm),
    }
    weighted = sum(WEIGHTS[k.split("_")[0]] * v for k, v in components.items())
    components["metadata_score"] = round(weighted, 4)
    return components


def fusion_score(face_score: float | None, metadata_score_value: float, face_weight: float = 0.6) -> float:
    """Combines face similarity with metadata compatibility into a single
    ranking score. Heuristic linear blend, not a calibrated probability -- see
    module docstring. When face_score is None (metadata-only candidate
    search, Phase 9), returns metadata_score_value unchanged."""
    if face_score is None:
        return metadata_score_value
    return face_weight * face_score + (1 - face_weight) * metadata_score_value


# ---------------------------------------------------------------------------
# Phase 6: explicit, separate conflict signals (replaces the single ambiguous
# `sex_conflict`). Unknown/missing data is never treated as a conflict -- these
# stay soft warnings for human review, never hard identity filters, and vision
# gender on PM (post-mortem/disaster) faces has not been benchmarked for
# reliability -- see docs/P0_5_IMPLEMENTATION.md.
# ---------------------------------------------------------------------------

def _conflict(a: str, b: str) -> bool:
    a, b = (a or "").lower(), (b or "").lower()
    if not a or not b:
        return False
    return a != b


def am_metadata_vision_conflict(am: dict) -> bool:
    """AM recorded sex != AM vision-estimated sex."""
    return _conflict(am.get("sex"), am.get("detected_sex"))


def pm_metadata_vision_conflict(pm: dict) -> bool:
    """PM recorded sex != PM vision-estimated sex."""
    return _conflict(pm.get("sex"), pm.get("detected_sex"))


def pair_metadata_conflict(am: dict, pm: dict) -> bool:
    """AM recorded sex != PM recorded sex."""
    return _conflict(am.get("sex"), pm.get("sex"))


def pair_vision_conflict(am: dict, pm: dict) -> bool:
    """AM vision-estimated sex != PM vision-estimated sex."""
    return _conflict(am.get("detected_sex"), pm.get("detected_sex"))


def compute_conflicts(am: dict, pm: dict) -> dict:
    return {
        "am_metadata_vision_conflict": am_metadata_vision_conflict(am),
        "pm_metadata_vision_conflict": pm_metadata_vision_conflict(pm),
        "pair_metadata_conflict": pair_metadata_conflict(am, pm),
        "pair_vision_conflict": pair_vision_conflict(am, pm),
    }
