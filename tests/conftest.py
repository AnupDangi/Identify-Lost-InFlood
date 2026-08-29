"""Shared pytest fixtures. All test data here is synthetic -- no real scraped
AM/PM records, photos, or embeddings are ever committed or read by tests, per
docs/P0_5_IMPLEMENTATION.md Phase 13.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

AM_FIELDS = ["record_id", "record_type", "name", "sex", "age_min", "age_max",
             "height_cm", "event_date", "raw_event_date", "calendar_type",
             "event_date_normalized", "location", "province", "district",
             "municipality", "ward", "clothing", "distinguishing_marks",
             "image_path", "image_sha256", "source_ref", "scraped_at"]

EMBED_INDEX_FIELDS = ["record_id", "usable", "failure_reason", "detector_score",
                       "face_width", "face_height", "blur_score", "detected_sex",
                       "quality_band", "processed_at"]

DIM = 8


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    """Builds a tiny synthetic AM/PM dataset (3 records each, 2 with usable
    faces) under tmp_path/data, builds real FAISS indexes over the fake
    embeddings, and monkeypatches dvi.retrieval (and main.py's own
    MANIFEST_DIR, used only for reviews.csv) onto it. Returns a dict of useful
    paths/vectors for assertions."""
    import faiss

    from dvi import retrieval

    data_dir = tmp_path / "data"
    manifest_dir = data_dir / "manifests"
    embed_dir = data_dir / "embeddings"
    index_dir = data_dir / "index"
    for d in (manifest_dir, embed_dir / "am", embed_dir / "pm", index_dir):
        d.mkdir(parents=True, exist_ok=True)

    am_rows = [
        {"record_id": "AM1", "record_type": "AM", "name": "Test AM One", "sex": "male",
         "age_min": 30, "age_max": 35, "height_cm": 170, "event_date": "2082-05-12",
         "location": "Bagmati Kathmandu", "province": "Bagmati", "district": "Kathmandu",
         "municipality": "Kathmandu Metro"},
        {"record_id": "AM2", "record_type": "AM", "name": "Test AM Two", "sex": "female",
         "age_min": 20, "age_max": 25, "height_cm": 160, "event_date": "2026-01-05",
         "location": "Koshi Sunsari", "province": "Koshi", "district": "Sunsari"},
        {"record_id": "AM3", "record_type": "AM", "name": "Test AM Three", "sex": "male",
         "age_min": 40, "age_max": 45, "height_cm": 175, "event_date": "",
         "location": "", "province": "", "district": ""},
    ]
    pm_rows = [
        {"record_id": "PM1", "record_type": "PM", "sex": "male", "age_min": 28, "age_max": 33,
         "height_cm": 171, "event_date": "2082-05-20", "location": "Kathmandu Metro"},
        {"record_id": "PM2", "record_type": "PM", "sex": "female", "age_min": 55, "age_max": 60,
         "height_cm": 155, "event_date": "2026-01-08", "location": "Sunsari"},
        {"record_id": "PM3", "record_type": "PM", "sex": "other", "age_min": None, "age_max": None,
         "height_cm": None, "event_date": "", "location": ""},
    ]
    _write_csv(manifest_dir / "am_persons.csv", AM_FIELDS, am_rows)
    _write_csv(manifest_dir / "pm_persons.csv", AM_FIELDS, pm_rows)

    # AM1, AM2 and PM1 have usable faces; AM3/PM2/PM3 do not (metadata_only path).
    am_embed_rows = [
        {"record_id": "AM1", "usable": True, "detected_sex": "male", "detector_score": 0.9,
         "quality_band": "good"},
        {"record_id": "AM2", "usable": True, "detected_sex": "female", "detector_score": 0.85,
         "quality_band": "good"},
        {"record_id": "AM3", "usable": False, "failure_reason": "no_face_detected",
         "quality_band": "unusable"},
    ]
    pm_embed_rows = [
        {"record_id": "PM1", "usable": True, "detected_sex": "male", "detector_score": 0.88,
         "quality_band": "good"},
        {"record_id": "PM2", "usable": False, "failure_reason": "face_too_small",
         "quality_band": "unusable"},
        {"record_id": "PM3", "usable": False, "failure_reason": "no_face_detected",
         "quality_band": "unusable"},
    ]
    _write_csv(manifest_dir / "am_embeddings_index.csv", EMBED_INDEX_FIELDS, am_embed_rows)
    _write_csv(manifest_dir / "pm_embeddings_index.csv", EMBED_INDEX_FIELDS, pm_embed_rows)

    vectors = {"AM1": _unit_vec(1), "AM2": _unit_vec(2), "PM1": _unit_vec(3)}
    for record_type, rid in (("am", "AM1"), ("am", "AM2"), ("pm", "PM1")):
        np.save(embed_dir / record_type / f"{rid}.npy", vectors[rid])

    for record_type, id_map in (("am", ["AM1", "AM2"]), ("pm", ["PM1"])):
        matrix = np.vstack([vectors[i] for i in id_map]).astype(np.float32)
        index = faiss.IndexFlatIP(DIM)
        index.add(matrix)
        faiss.write_index(index, str(index_dir / f"{record_type}.index"))
        (index_dir / f"{record_type}_id_map.json").write_text(json.dumps(id_map))

    monkeypatch.setattr(retrieval, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(retrieval, "EMBED_DIR", embed_dir)
    monkeypatch.setattr(retrieval, "INDEX_DIR", index_dir)
    retrieval.clear_caches()

    return {
        "tmp_path": tmp_path, "manifest_dir": manifest_dir, "embed_dir": embed_dir,
        "index_dir": index_dir, "vectors": vectors,
    }
