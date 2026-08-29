"""
P0 candidate search: PM -> AM face retrieval (FAISS) + metadata re-ranking,
per docs/project_requirement.md sections 12-15.

For each PM record with a usable face embedding, retrieves the Top-N AM
identities by cosine similarity (FAISS IndexFlatIP over ArcFace embeddings),
then blends that face_score with the metadata compatibility score from
scripts/compare_am_pm.py (sex/age/height/date/location) into a final_score.

This produces a SHORTLIST for human review, never an identification -- see
docs/project_requirement.md section 15 ("Do not return identity/confidence").

Requires build_embeddings.py + build_index.py (--record-type am) to have
been run first.

Usage:
    uv run python scripts/search_candidates.py --top-k 20
    uv run python scripts/search_candidates.py --top-k 20 --face-weight 0.7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_am_pm import (  # noqa: E402
    sex_score, age_score, height_score, date_score, location_score,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
EMBED_DIR = ROOT / "data" / "embeddings"
INDEX_DIR = ROOT / "data" / "index"


def load_records(record_type: str) -> dict[str, dict]:
    # fillna("") up front: pandas turns blank CSV cells into NaN (a float),
    # and compare_am_pm's scoring functions call .strip()/.split() on
    # event_date/location -- those crash outright on NaN (blank `location`
    # is common for AM records), and age/height's to_float(nan) silently
    # returns nan instead of None, making band_score fall through to the
    # worst band instead of the intended neutral 0.5. "" round-trips
    # through those functions the same way csv.DictReader would present it.
    df = pd.read_csv(MANIFEST_DIR / f"{record_type}_persons.csv").fillna("")
    records = {row["record_id"]: row.to_dict() for _, row in df.iterrows()}

    idx_path = MANIFEST_DIR / f"{record_type}_embeddings_index.csv"
    if idx_path.exists():
        idx_df = pd.read_csv(idx_path)
        for _, r in idx_df.iterrows():
            rec = records.get(r["record_id"])
            if rec is not None:
                val = r.get("detected_sex")
                # a blank cell round-trips through pandas as NaN (a float), not "" --
                # and NaN is truthy, so `val or ""` would silently keep the NaN and
                # crash the first .lower() call downstream. Guard explicitly.
                rec["detected_sex"] = val if pd.notna(val) else ""
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=20, help="candidates kept per PM record")
    ap.add_argument("--faiss-k", type=int, default=100, help="raw FAISS neighbors pulled before re-ranking")
    ap.add_argument("--face-weight", type=float, default=0.6,
                     help="blend weight for face_score vs (1 - face_weight) for metadata_score")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "candidate_matches_face.csv"))
    args = ap.parse_args()

    am_index = faiss.read_index(str(INDEX_DIR / "am.index"))
    am_id_map = json.loads((INDEX_DIR / "am_id_map.json").read_text())

    am_records = load_records("am")
    pm_records = load_records("pm")

    pm_embed_dir = EMBED_DIR / "pm"
    pm_embed_index = pd.read_csv(MANIFEST_DIR / "pm_embeddings_index.csv")
    pm_usable = pm_embed_index[pm_embed_index["usable"] == True]  # noqa: E712

    print(f"AM index: {len(am_id_map)} embeddings")
    print(f"PM usable faces: {len(pm_usable)}")

    rows = []
    for _, prow in pm_usable.iterrows():
        pm_id = prow["record_id"]
        npy_path = pm_embed_dir / f"{pm_id}.npy"
        if not npy_path.exists():
            continue
        query = np.load(npy_path).astype(np.float32).reshape(1, -1)

        sims, idxs = am_index.search(query, min(args.faiss_k, len(am_id_map)))
        sims, idxs = sims[0], idxs[0]

        pm = pm_records.get(pm_id, {})
        scored = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            am_id = am_id_map[idx]
            am = am_records.get(am_id, {})
            face_score = float(max(0.0, min(1.0, (sim + 1) / 2)))  # cosine [-1,1] -> [0,1]
            meta = {
                "sex_score": sex_score(am, pm),
                "age_score": age_score(am, pm),
                "height_score": height_score(am, pm),
                "date_score": date_score(am, pm),
                "location_score": location_score(am, pm),
            }
            metadata_score = (0.30 * meta["sex_score"] + 0.25 * meta["age_score"] +
                               0.10 * meta["height_score"] + 0.15 * meta["date_score"] +
                               0.20 * meta["location_score"])
            final_score = args.face_weight * face_score + (1 - args.face_weight) * metadata_score
            am_det = (am.get("detected_sex") or "").lower()
            pm_det = (pm.get("detected_sex") or "").lower()
            sex_conflict = bool(am_det and pm_det and am_det != pm_det)
            scored.append((am_id, face_score, metadata_score, final_score, meta, sex_conflict))

        scored.sort(key=lambda x: x[3], reverse=True)
        for rank, (am_id, face_score, metadata_score, final_score, meta, sex_conflict) in enumerate(
                scored[: args.top_k], 1):
            rows.append({
                "pm_record_id": pm_id,
                "am_record_id": am_id,
                "rank": rank,
                "face_score": round(face_score, 4),
                "metadata_score": round(metadata_score, 4),
                "final_score": round(final_score, 4),
                "sex_conflict": sex_conflict,
                **{k: v for k, v in meta.items()},
            })

    out_path = Path(args.out)
    fieldnames = ["pm_record_id", "am_record_id", "rank", "face_score", "metadata_score",
                  "final_score", "sex_conflict", "sex_score", "age_score", "height_score",
                  "date_score", "location_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} candidate row(s) -> {out_path}")
    print("NOTE: face+metadata shortlist for human review only -- never an identification.")


if __name__ == "__main__":
    main()
