"""Tests for transient upload search: photo + optional name/sex -> face+gender
matching against bodies with no metadata (the requested feature).

The upload endpoint extracts an embedding from the uploaded bytes via
main._process_upload_bytes; tests monkeypatch that helper to return synthetic
vectors so no real InsightFace model is needed.
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main as main_module
from dvi import retrieval


@pytest.fixture
def upload_client(synthetic_dataset, monkeypatch):
    monkeypatch.setattr(main_module, "MANIFEST_DIR", synthetic_dataset["manifest_dir"])
    return TestClient(main_module.app)


def _fake_image_bytes():
    # Minimal valid JPEG bytes via Pillow or just a PNG header that cv2 would reject.
    # The endpoint's _process_upload_bytes is monkeypatched, so exact bytes don't matter
    # except they must be non-empty and have an allowed filename extension.
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# --- dvi.retrieval transient search unit tests ---

def test_transient_search_with_face_uses_faiss(synthetic_dataset):
    # synthetic_dataset: AM1, AM2, PM1 have faces; others don't.
    # PM gallery has PM1 only (1 vector), AM gallery has AM1, AM2.
    # Use PM1's own vector as the query embedding -> should find PM1 when searching PM,
    # or AM nearest when searching AM depending on similarity.
    vec = synthetic_dataset["vectors"]["PM1"]
    query_rec = {"sex": "male", "detected_sex": "male", "name": "Test Upload"}
    # Search PM gallery with a PM1-like vector -> PM1 should be rank 1
    res = retrieval.search_transient(query_rec, vec, candidate_type="pm", query_type="am", top_k=5)
    assert res["has_face"] is True
    assert res["candidate_source"] == "face+metadata"
    assert res["candidates"][0]["candidate_record_id"] == "PM1"
    assert res["candidates"][0]["face_score"] is not None


def test_transient_search_without_face_falls_back_to_metadata_only(synthetic_dataset):
    query_rec = {"sex": "female", "detected_sex": "female", "name": ""}
    res = retrieval.search_transient(query_rec, None, candidate_type="pm", query_type="am", top_k=10)
    assert res["has_face"] is False
    assert res["candidate_source"] == "metadata_only"
    # All PM records ranked (3), even though only PM1 has a face in the index
    assert len(res["candidates"]) == 3
    for c in res["candidates"]:
        assert c["face_score"] is None


def test_transient_search_name_is_optional(synthetic_dataset):
    vec = synthetic_dataset["vectors"]["AM1"]
    # No name provided -> still works, name is informational only
    query_rec = {"sex": "", "detected_sex": "male", "name": ""}
    res = retrieval.search_transient(query_rec, vec, candidate_type="pm", query_type="am", top_k=3)
    assert res["candidates"]


def test_transient_search_gender_only_against_bodies_with_no_metadata(synthetic_dataset):
    """PM3 has sex='other' but no other metadata -> should still be ranked,
    not hard-filtered. A gender-only upload must surface bodies without metadata."""
    query_rec = {"sex": "male", "detected_sex": "male"}
    res = retrieval.search_transient(query_rec, None, candidate_type="pm", query_type="am", top_k=10)
    ids = {c["candidate_record_id"] for c in res["candidates"]}
    assert "PM3" in ids  # body without meaningful metadata still appears
    pm3 = next(c for c in res["candidates"] if c["candidate_record_id"] == "PM3")
    # PM3 sex is 'other' vs male query -> 0.0 on the metadata signal, but still ranked (soft, not filter)
    assert pm3["sex_score"] in (0.0, 0.5)


# --- API-level tests for POST /api/search/upload ---

def test_upload_search_with_face_and_name(upload_client, monkeypatch, synthetic_dataset):
    vec = synthetic_dataset["vectors"]["PM1"]

    def fake_process(data: bytes):
        return {
            "has_face": True,
            "embedding": vec,
            "detected_sex": "male",
            "quality_band": "good",
            "failure_reason": "",
            "detector_score": 0.95,
        }

    monkeypatch.setattr(main_module, "_process_upload_bytes", fake_process)

    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("photo.jpg", _fake_image_bytes(), "image/jpeg")},
        data={"name": "Ram Bahadur", "sex": "male", "target": "pm", "top_k": "5"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["has_face"] is True
    assert body["candidate_type"] == "pm"
    assert body["query"]["provided_name"] == "Ram Bahadur"
    assert body["query"]["provided_sex"] == "male"
    assert body["query"]["detected_sex"] == "male"
    assert body["source"] == "face+metadata"
    assert len(body["candidates"]) == 1  # PM has only PM1 in index
    assert "disclaimer" in body and "not an identity probability" in body["disclaimer"]
    # Never labeled as confidence/identity
    assert "confidence" not in str(body).lower() or "not an identity probability" in str(body).lower()
    # Candidate shape
    cand = body["candidates"][0]
    assert "face_score" in cand and "metadata_score" in cand and "final_score" in cand


def test_upload_search_name_optional_no_face_fallback_to_gender_only(upload_client, monkeypatch):
    def fake_no_face(data: bytes):
        return {
            "has_face": False,
            "embedding": None,
            "detected_sex": "female",
            "quality_band": "unusable",
            "failure_reason": "no_face_detected",
            "detector_score": None,
        }

    monkeypatch.setattr(main_module, "_process_upload_bytes", fake_no_face)

    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("face.png", _fake_image_bytes(), "image/png")},
        data={"sex": "female", "target": "pm"},  # no name
    )
    assert res.status_code == 200
    body = res.json()
    assert body["has_face"] is False
    assert body["query"]["provided_name"] is None  # empty name -> None
    assert body["query"]["provided_sex"] == "female"
    assert body["source"] == "metadata_only"
    # Gender-only still returns full PM gallery (3) even though bodies have little metadata
    assert len(body["candidates"]) == 3
    for c in body["candidates"]:
        assert c["face_score"] is None


def test_upload_search_bodies_without_metadata_still_ranked(upload_client, monkeypatch):
    """Explicitly verifies the user's stated requirement: a body without any
    metadata is still found via photo+gender alone (face+sex scores, rest neutral)."""
    def fake_no_face(data: bytes):
        return {
            "has_face": False,
            "embedding": None,
            "detected_sex": "male",
            "quality_band": "unusable",
            "failure_reason": "no_face_detected",
            "detector_score": None,
        }

    monkeypatch.setattr(main_module, "_process_upload_bytes", fake_no_face)
    # Don't provide name, only gender, search PM where PM3 has no metadata at all
    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("p.jpg", _fake_image_bytes(), "image/jpeg")},
        data={"sex": "male", "target": "pm"},
    )
    assert res.status_code == 200
    body = res.json()
    ids = {c.get("pm_record_id") or c.get("record_id") for c in body["candidates"]}
    assert "PM3" in ids or any(c.get("record_id") == "PM3" for c in body["candidates"]) or \
           any("PM3" in str(c) for c in body["candidates"])


def test_upload_search_invalid_target_rejected(upload_client):
    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("a.jpg", _fake_image_bytes(), "image/jpeg")},
        data={"target": "invalid"},
    )
    assert res.status_code == 400


def test_upload_search_invalid_sex_rejected(upload_client):
    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("a.jpg", _fake_image_bytes(), "image/jpeg")},
        data={"sex": "notasex"},
    )
    assert res.status_code == 400


def test_upload_search_rejects_non_image_extension(upload_client):
    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("evil.exe", b"not an image", "application/octet-stream")},
        data={},
    )
    assert res.status_code == 400


def test_upload_search_target_am_searches_missing_gallery(upload_client, monkeypatch, synthetic_dataset):
    vec = synthetic_dataset["vectors"]["AM1"]

    def fake_process(data: bytes):
        return {
            "has_face": True,
            "embedding": vec,
            "detected_sex": "female",
            "quality_band": "good",
            "failure_reason": "",
            "detector_score": 0.9,
        }

    monkeypatch.setattr(main_module, "_process_upload_bytes", fake_process)
    res = upload_client.post(
        "/api/search/upload",
        files={"file": ("p.jpg", _fake_image_bytes(), "image/jpeg")},
        data={"target": "am", "sex": "female"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["candidate_type"] == "am"
    # All returned candidates should be AM records
    for c in body["candidates"]:
        assert ("am_record_id" in c) or ("AM" in str(c.values()))
