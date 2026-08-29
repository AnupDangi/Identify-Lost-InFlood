"""P0.5 bidirectional candidate search: AM<->PM face retrieval (FAISS) +
metadata re-ranking, per docs/project_requirement.md sections 12-15 and
docs/P0_5_IMPLEMENTATION.md Phases 7/8/9.

For every AM record it searches the PM FAISS index, and for every PM record it
searches the AM FAISS index -- both directions are independent FAISS searches
(Phase 7: cosine similarity is symmetric, Top-K neighborhoods are not). A
record with no usable face falls back to a metadata-only ranking over the
*entire* candidate gallery instead of being skipped (Phase 9) -- and by
default the FAISS search itself covers the entire gallery too (Phase 8:
--faiss-k all, the default, since IndexFlatIP is exact at this project's
current ~6k/2k scale; there's no accuracy cost to not truncating early).

Writes ONE normalized candidate table (per docs/P0_5_IMPLEMENTATION.md Phase 7's
"use the cleaner implementation" guidance) instead of separate pivoted files per
direction:

    data/manifests/candidates.csv

Columns: query_type, query_record_id, candidate_type, candidate_record_id, rank,
face_score, metadata_score, final_score, sex_score, age_score, height_score,
date_score, location_score, am_metadata_vision_conflict,
pm_metadata_vision_conflict, pair_metadata_conflict, pair_vision_conflict,
candidate_source.

This is a face+metadata SHORTLIST for human review, never an identification --
see docs/project_requirement.md section 15 ("Do not return identity/confidence").

Requires build_embeddings.py + build_index.py (both --record-type am and
--record-type pm) to have been run first for the face+metadata path; runs
metadata-only otherwise.

Usage:
    uv run python scripts/search_candidates.py --top-k 20
    uv run python scripts/search_candidates.py --top-k 20 --face-weight 0.7
    uv run python scripts/search_candidates.py --faiss-k 500   # cap instead of full gallery
    uv run python scripts/search_candidates.py --record-types am  # only AM->PM direction
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvi import retrieval

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"

FIELDNAMES = [
    "query_type", "query_record_id", "candidate_type", "candidate_record_id", "rank",
    "face_score", "metadata_score", "final_score",
    "sex_score", "age_score", "height_score", "date_score", "location_score",
    "am_metadata_vision_conflict", "pm_metadata_vision_conflict",
    "pair_metadata_conflict", "pair_vision_conflict",
    "candidate_source",
]


def run_direction(query_type: str, args) -> tuple[list[dict], int, int]:
    candidate_type = "pm" if query_type == "am" else "am"
    query_records = retrieval.load_records(query_type)
    rows = []
    n_face, n_metadata_only = 0, 0
    total = len(query_records)
    start = time.time()
    for i, query_record_id in enumerate(query_records, 1):
        result = retrieval.search_query(
            query_type, query_record_id, candidate_type,
            top_k=args.top_k, faiss_k=args.faiss_k, face_weight=args.face_weight,
        )
        if result["candidate_source"] == "face+metadata":
            n_face += 1
        elif result["candidate_source"] == "metadata_only":
            n_metadata_only += 1
        for c in result["candidates"]:
            rows.append({
                "query_type": query_type, "query_record_id": query_record_id,
                "candidate_type": candidate_type, "candidate_source": result["candidate_source"],
                **c,
            })
        if i % 200 == 0 or i == total:
            elapsed = time.time() - start
            print(f"  [{query_type}->{candidate_type}] {i}/{total} queries "
                  f"({elapsed:.0f}s elapsed, face={n_face} metadata_only={n_metadata_only})")
    return rows, n_face, n_metadata_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=20, help="candidates kept per query record")
    ap.add_argument("--faiss-k", default="all",
                     help='raw FAISS neighbors pulled before re-ranking; "all" (default) searches '
                          f"the full gallery when it's under {retrieval.FAISS_FULL_GALLERY_THRESHOLD:,} "
                          "vectors, otherwise pass an explicit integer")
    ap.add_argument("--face-weight", type=float, default=0.6,
                     help="blend weight for face_score vs (1 - face_weight) for metadata_score")
    ap.add_argument("--record-types", choices=["am", "pm", "both"], default="both",
                     help="which side(s) to run as the query (am->pm, pm->am, or both)")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "candidates.csv"))
    args = ap.parse_args()

    query_types = ["am", "pm"] if args.record_types == "both" else [args.record_types]

    all_rows = []
    for qt in query_types:
        print(f"[{qt}] searching {'pm' if qt == 'am' else 'am'} gallery "
              f"(faiss_k={args.faiss_k}, top_k={args.top_k})...")
        rows, n_face, n_meta = run_direction(qt, args)
        all_rows.extend(rows)
        print(f"[{qt}] {n_face} face+metadata query(ies), {n_meta} metadata_only fallback(s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nwrote {len(all_rows)} candidate row(s) -> {out_path}")
    print("NOTE: face+metadata/metadata-only shortlist for human review only -- never an identification.")


if __name__ == "__main__":
    main()
