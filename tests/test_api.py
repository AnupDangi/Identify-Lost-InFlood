import csv

import pytest
from fastapi.testclient import TestClient

import main as main_module


@pytest.fixture
def client(synthetic_dataset, monkeypatch):
    # main.py's review endpoint uses its own MANIFEST_DIR (for reviews.csv);
    # everything else goes through dvi.retrieval, already patched by the
    # synthetic_dataset fixture.
    monkeypatch.setattr(main_module, "MANIFEST_DIR", synthetic_dataset["manifest_dir"])
    return TestClient(main_module.app)


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_stats_returns_counts_without_sensitive_content(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["am_records"] == 3
    assert body["pm_records"] == 3
    assert body["am_usable_faces"] == 2
    assert body["pm_usable_faces"] == 1
    assert "face_model_version" in body and "ranking_model_version" in body
    # No raw image bytes/paths belong in a stats summary.
    assert "image_path" not in body


def test_list_am_and_pm(client):
    res = client.get("/api/am")
    assert res.status_code == 200
    assert res.json()["total"] == 3

    res = client.get("/api/pm")
    assert res.status_code == 200
    assert res.json()["total"] == 3


def test_no_face_query_gets_candidates_not_an_error(client):
    """Phase 9: a record with no usable face is NOT a 422 -- it gets a
    metadata_only shortlist."""
    res = client.get("/api/am/AM3/candidates")
    assert res.status_code == 200
    body = res.json()
    assert body["has_face"] is False
    assert body["source"] == "metadata_only"
    assert len(body["candidates"]) == 3


def test_unknown_record_is_404(client):
    res = client.get("/api/am/does-not-exist/candidates")
    assert res.status_code == 404


def test_candidates_never_labeled_as_probability_or_confidence(client):
    res = client.get("/api/pm/PM1/candidates")
    assert res.status_code == 200
    body = res.json()
    assert "not an identity probability" in body["disclaimer"]

    def _walk_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _walk_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from _walk_keys(item)

    forbidden = {"confidence", "identity_probability", "match_probability", "identity"}
    keys = set(_walk_keys(body))
    assert not (keys & forbidden)


def test_invalid_review_decision_is_rejected(client):
    res = client.post("/api/review", json={
        "pm_record_id": "PM1", "am_record_id": "AM1", "decision": "definitely_them",
    })
    assert res.status_code == 400


def test_valid_review_writes_full_audit_fields(client, synthetic_dataset):
    res = client.post("/api/review", json={
        "pm_record_id": "PM1", "am_record_id": "AM1", "decision": "potential", "notes": "test note",
    })
    assert res.status_code == 200

    reviews_path = synthetic_dataset["manifest_dir"] / "reviews.csv"
    with reviews_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    for field in ("review_id", "pm_record_id", "am_record_id", "decision", "notes",
                  "reviewed_at", "ranking_model_version", "face_model_version"):
        assert row.get(field)
