"""
Metadata-only AM (missing person) <-> PM (unidentified body) comparison.

This is the P0 "metadata re-ranking" step from docs/project_requirement.md
section 13-14, run WITHOUT any face embeddings (no ArcFace/FAISS in this
repo yet) -- it only compares sex/age/height/date/location compatibility
using soft bands, not hard filters. It is a candidate-shortlisting aid for
a human investigator, not an identification. Never treat its output as a
match.

Usage:
    uv run python scripts/compare_am_pm.py
    uv run python scripts/compare_am_pm.py --top-k 10 --min-score 0.4
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_date(v: str) -> Optional[datetime]:
    v = (v or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(v[:10], fmt)
        except ValueError:
            continue
    return None


def band_score(diff: Optional[float], bands: list[tuple[float, float]]) -> float:
    """bands: list of (max_diff, score), checked in order; last is fallback."""
    if diff is None:
        return 0.5  # neutral when unknown, per plan section 14 (don't hard-filter)
    for max_diff, score in bands:
        if diff <= max_diff:
            return score
    return bands[-1][1]


def sex_score(am: dict, pm: dict) -> float:
    """Blends the scraped metadata sex with the InsightFace-detected sex
    (when a record dict carries a "detected_sex" key, as search_candidates.py
    merges in from the embeddings index). Each available signal contributes
    equally; two records that agree on metadata but whose photos read as
    different sexes score lower than a metadata-only match would, and vice
    versa -- a soft cross-check, not a hard filter, per the "don't hard-filter
    demographics" rule."""
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
    a, p = parse_date(am.get("event_date", "")), parse_date(pm.get("event_date", ""))
    if a is None or p is None:
        return 0.5
    gap = (p - a).days
    if gap < 0:
        return 0.2  # found before reported missing -> implausible but not impossible (reporting lag)
    return band_score(gap, [(10, 1.0), (30, 0.7), (90, 0.4), (99999, 0.2)])


def location_score(am: dict, pm: dict) -> float:
    a_tokens = set((am.get("location") or "").split())
    p_tokens = set((pm.get("location") or "").split())
    if not a_tokens or not p_tokens:
        return 0.5
    overlap = a_tokens & p_tokens
    if len(overlap) >= 2:
        return 1.0
    if len(overlap) == 1:
        return 0.6
    return 0.2


WEIGHTS = {
    "sex": 0.30,
    "age": 0.25,
    "height": 0.10,
    "date": 0.15,
    "location": 0.20,
}


def score_pair(am: dict, pm: dict) -> dict:
    s = {
        "sex_score": sex_score(am, pm),
        "age_score": age_score(am, pm),
        "height_score": height_score(am, pm),
        "date_score": date_score(am, pm),
        "location_score": location_score(am, pm),
    }
    final = sum(WEIGHTS[k.split("_")[0]] * v for k, v in s.items())
    s["final_score"] = round(final, 4)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--out", default=str(MANIFEST_DIR / "candidate_matches.csv"))
    args = ap.parse_args()

    am_records = load_csv(MANIFEST_DIR / "am_persons.csv")
    pm_records = load_csv(MANIFEST_DIR / "pm_persons.csv")

    print(f"Loaded {len(am_records)} AM (missing) and {len(pm_records)} PM (unidentified body) records")
    if not am_records or not pm_records:
        print("Nothing to compare -- run scripts/scrape_udb.py first.")
        return

    rows = []
    for pm in pm_records:
        scored = []
        for am in am_records:
            s = score_pair(am, pm)
            if s["final_score"] >= args.min_score:
                scored.append((am, s))
        scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
        for rank, (am, s) in enumerate(scored[: args.top_k], 1):
            rows.append({
                "pm_record_id": pm["record_id"],
                "am_record_id": am["record_id"],
                "rank": rank,
                **s,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pm_record_id", "am_record_id", "rank", "sex_score", "age_score",
                  "height_score", "date_score", "location_score", "final_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} candidate row(s) -> {out_path}")
    print()
    print("NOTE: metadata-only heuristic (no face embeddings in this repo yet).")
    print("This is a shortlist for human review, never an identification.")

    if rows:
        top = sorted(rows, key=lambda r: r["final_score"], reverse=True)[:5]
        print("\nTop overall candidate pairs across the whole comparison:")
        for r in top:
            print(f"  PM {r['pm_record_id']} <-> AM {r['am_record_id']}  score={r['final_score']:.2f}"
                  f"  (sex={r['sex_score']:.1f} age={r['age_score']:.1f} height={r['height_score']:.1f}"
                  f" date={r['date_score']:.1f} loc={r['location_score']:.1f})")


if __name__ == "__main__":
    main()
