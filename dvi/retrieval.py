"""Bidirectional AM<->PM candidate retrieval, per docs/P0_5_IMPLEMENTATION.md
Phases 7, 8, and 9.

- Phase 7 (true bidirectional search): an AM query searches the PM FAISS index
  and a PM query searches the AM FAISS index -- each direction is its own FAISS
  search, never a pivot/lookup into the other direction's precomputed results
  (cosine similarity is symmetric, but a Top-K neighborhood is not).
- Phase 8 (no premature truncation): faiss_k="all" (the default) searches the
  entire candidate gallery when it's under FAISS_FULL_GALLERY_THRESHOLD vectors,
  so a correct-but-visually-degraded face can't be cut before metadata gets a
  chance to recover it. IndexFlatIP is exact at this scale; there is no
  approximate-index accuracy loss from setting k this high.
- Phase 9 (metadata fallback): a query with no usable face embedding is not an
  error -- it ranks the full candidate gallery by metadata_score alone
  (candidate_source="metadata_only") instead of refusing to search.

Shared by scripts/search_candidates.py (writes the batch
data/manifests/candidates.csv used as a fallback / for evaluation) and main.py
(computes candidates on demand per Phase 17, so the API doesn't only serve
whatever the last batch run produced).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from dvi import scoring

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
EMBED_DIR = ROOT / "data" / "embeddings"
INDEX_DIR = ROOT / "data" / "index"

# Below this many vectors, "all"/default --faiss-k searches the whole gallery
# (see module docstring, Phase 8). Above it, callers should pass an explicit
# --faiss-k -- IndexFlatIP is still exact, this is purely a cost control for
# scales this prototype was not built for.
FAISS_FULL_GALLERY_THRESHOLD = 100_000

VALID_TYPES = ("am", "pm")


def _other(record_type: str) -> str:
    return "pm" if record_type == "am" else "am"


@functools.lru_cache(maxsize=4)
def load_records(record_type: str) -> dict[str, dict]:
    """record_id -> merged dict of manifest columns + embeddings-index columns
    (usable/detected_sex/quality fields) + date/location normalization columns
    when those backfills have been run. Cached per record_type per process --
    call load_records.cache_clear() after mutating the underlying CSVs (e.g.
    between test cases, or after re-running a backfill script)."""
    manifest_path = MANIFEST_DIR / f"{record_type}_persons.csv"
    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path).fillna("")
    records = {row["record_id"]: row.to_dict() for _, row in df.iterrows()}

    idx_path = MANIFEST_DIR / f"{record_type}_embeddings_index.csv"
    if idx_path.exists():
        idx_df = pd.read_csv(idx_path).fillna("")
        for _, r in idx_df.iterrows():
            rec = records.get(r["record_id"])
            if rec is None:
                continue
            for col in ("usable", "detected_sex", "detector_score", "blur_score",
                        "quality_band", "face_width", "face_height"):
                if col in idx_df.columns:
                    rec[col] = r[col]
    return records


@functools.lru_cache(maxsize=4)
def load_index(record_type: str):
    """Returns (faiss.Index, id_map) or (None, []) if not built yet -- callers
    must treat a missing index as "fall back to metadata_only", not an error."""
    index_path = INDEX_DIR / f"{record_type}.index"
    id_map_path = INDEX_DIR / f"{record_type}_id_map.json"
    if not index_path.exists() or not id_map_path.exists():
        return None, []
    import json as _json

    import faiss
    index = faiss.read_index(str(index_path))
    id_map = _json.loads(id_map_path.read_text())
    return index, id_map


def clear_caches():
    load_records.cache_clear()
    load_index.cache_clear()


def resolve_faiss_k(faiss_k, gallery_size: int) -> int:
    if gallery_size <= 0:
        return 0
    if faiss_k in (None, "all"):
        return min(gallery_size, FAISS_FULL_GALLERY_THRESHOLD)
    return max(1, min(int(faiss_k), gallery_size))


def _am_pm(query_type: str, query_rec: dict, candidate_type: str, cand_rec: dict) -> tuple[dict, dict]:
    """dvi.scoring's component functions all take (am, pm) in that order."""
    if query_type == "am":
        return query_rec, cand_rec
    return cand_rec, query_rec


def _score_pair(query_type: str, query_rec: dict, candidate_type: str, cand_rec: dict,
                 face_score: float | None, face_weight: float) -> dict:
    am_rec, pm_rec = _am_pm(query_type, query_rec, candidate_type, cand_rec)
    meta = scoring.metadata_score(am_rec, pm_rec)
    final = scoring.fusion_score(face_score, meta["metadata_score"], face_weight)
    conflicts = scoring.compute_conflicts(am_rec, pm_rec)
    row = {
        "face_score": round(face_score, 4) if face_score is not None else None,
        "metadata_score": meta["metadata_score"],
        "final_score": round(final, 4),
        "sex_score": meta["sex_score"], "age_score": meta["age_score"],
        "height_score": meta["height_score"], "date_score": meta["date_score"],
        "location_score": meta["location_score"],
        **conflicts,
    }
    return row


def has_usable_face(record_type: str, record_id: str) -> bool:
    return (EMBED_DIR / record_type / f"{record_id}.npy").exists()


def metadata_only_ranking(query_type: str, query_record_id: str, candidate_type: str,
                           face_weight: float = 0.6) -> list[dict]:
    """Ranks the FULL candidate_type gallery by metadata alone, regardless of
    whether the query itself has a usable face -- used both by search_query's
    Phase-9 fallback and directly by scripts/evaluate_retrieval.py's ablation
    (Phase 16), where the metadata-only condition should not be limited to
    only the candidates that happen to have a usable face too."""
    query_records = load_records(query_type)
    if query_record_id not in query_records:
        raise KeyError(f"{query_record_id} not found in {query_type} records")
    query_rec = query_records[query_record_id]
    candidate_records = load_records(candidate_type)

    results = []
    for cand_id, cand_rec in candidate_records.items():
        row = _score_pair(query_type, query_rec, candidate_type, cand_rec, None, face_weight)
        row["candidate_record_id"] = cand_id
        results.append(row)
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


def face_plus_metadata_ranking(query_type: str, query_record_id: str, candidate_type: str,
                                faiss_k="all", face_weight: float = 0.6) -> list[dict] | None:
    """FAISS face search (over candidate_type's index) + metadata re-ranking,
    UNTRUNCATED and sorted by final_score. Returns None if the query has no
    usable face embedding or candidate_type's index hasn't been built --
    callers (search_query, and evaluate_retrieval.py's ablation) treat that as
    "fall back to metadata_only", not an error. Each row carries face_score,
    metadata_score, and final_score, so a caller doing an ablation can locally
    re-sort by whichever signal it wants without a second FAISS search."""
    query_records = load_records(query_type)
    if query_record_id not in query_records:
        raise KeyError(f"{query_record_id} not found in {query_type} records")
    query_rec = query_records[query_record_id]

    query_embed_path = EMBED_DIR / query_type / f"{query_record_id}.npy"
    if not query_embed_path.exists():
        return None
    index, id_map = load_index(candidate_type)
    if index is None or not id_map:
        return None

    candidate_records = load_records(candidate_type)
    qvec = np.load(query_embed_path).astype(np.float32).reshape(1, -1)
    k = resolve_faiss_k(faiss_k, len(id_map))
    sims, idxs = index.search(qvec, k)

    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0:
            continue
        cand_id = id_map[idx]
        cand_rec = candidate_records.get(cand_id)
        if cand_rec is None:
            continue
        face_score = float(max(0.0, min(1.0, (float(sim) + 1) / 2)))
        row = _score_pair(query_type, query_rec, candidate_type, cand_rec, face_score, face_weight)
        row["candidate_record_id"] = cand_id
        results.append(row)

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results


def search_query(query_type: str, query_record_id: str, candidate_type: str,
                  top_k: int = 20, faiss_k="all", face_weight: float = 0.6) -> dict:
    """Ranks `candidate_type` records for one `query_type` record.

    Returns {"candidate_source": "face+metadata"|"metadata_only"|"none",
             "has_face": bool, "candidates": [ {candidate_record_id, rank, ...} ]}.
    Never raises for "no usable face" -- that's the metadata_only path, not an
    error (Phase 9). Raises KeyError only if query_record_id truly doesn't
    exist in query_type's manifest -- callers should catch that themselves.
    """
    assert query_type in VALID_TYPES and candidate_type in VALID_TYPES and query_type != candidate_type

    has_face = has_usable_face(query_type, query_record_id)
    results = face_plus_metadata_ranking(query_type, query_record_id, candidate_type,
                                          faiss_k=faiss_k, face_weight=face_weight)
    if results is not None:
        source = "face+metadata"
    else:
        results = metadata_only_ranking(query_type, query_record_id, candidate_type, face_weight)
        source = "metadata_only"

    for rank, row in enumerate(results[:top_k], 1):
        row["rank"] = rank
    results = results[:top_k]

    return {
        "candidate_source": source if results else "none",
        "has_face": has_face,
        "candidates": results,
    }
