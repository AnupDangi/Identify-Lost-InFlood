"""Metadata-only AM (missing person) <-> PM (unidentified body) comparison.

This is the P0 "metadata re-ranking" step from docs/project_requirement.md
section 13-14, run WITHOUT any face embeddings -- useful as a quick sanity
check before the face pipeline has run, or when neither record has a usable
face (see dvi.retrieval's metadata_only path, which main.py and
scripts/search_candidates.py use for the same purpose at query/candidate-search
time). It only compares sex/age/height/date/location compatibility using soft
bands, not hard filters. It is a candidate-shortlisting aid for a human
investigator, not an identification. Never treat its output as a match.

Scoring functions live in dvi/scoring.py (Phase 12: centralized, not duplicated
between this script and scripts/search_candidates.py / dvi/retrieval.py).

Usage:
    uv run python scripts/compare_am_pm.py
    uv run python scripts/compare_am_pm.py --top-k 10 --min-score 0.4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvi import scoring

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"

# Re-exported for any external caller/test that imported these names from this
# module before Phase 12 centralized them in dvi.scoring.
sex_score = scoring.sex_score
age_score = scoring.age_score
height_score = scoring.height_score
date_score = scoring.date_score
location_score = scoring.location_score


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_pair(am: dict, pm: dict) -> dict:
    meta = scoring.metadata_score(am, pm)
    meta["final_score"] = meta.pop("metadata_score")
    return meta


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
    print("NOTE: metadata-only heuristic (no face embeddings). This is a shortlist")
    print("for human review, never an identification. Weights are heuristic, not")
    print("calibrated probabilities -- see dvi/scoring.py.")

    if rows:
        top = sorted(rows, key=lambda r: r["final_score"], reverse=True)[:5]
        print("\nTop overall candidate pairs across the whole comparison:")
        for r in top:
            print(f"  PM {r['pm_record_id']} <-> AM {r['am_record_id']}  score={r['final_score']:.2f}"
                  f"  (sex={r['sex_score']:.1f} age={r['age_score']:.1f} height={r['height_score']:.1f}"
                  f" date={r['date_score']:.1f} loc={r['location_score']:.1f})")


if __name__ == "__main__":
    main()
