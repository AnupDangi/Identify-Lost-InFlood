"""Phase 3 acceptance test: run build_embeddings twice, the second run must
retain exactly the same metadata unless --force is given, an image sha256
change must invalidate the cache, and a model_version change must too.

InsightFace itself is never invoked here -- get_app()/process_image() are
monkeypatched with a deterministic fake so this test doesn't need the real
model bundle downloaded, and runs fast.
"""
from __future__ import annotations

import csv
import json
import sys

import build_embeddings as be
import numpy as np
import pytest


def _fake_process_image(app, image_path, **kwargs):
    return {
        "usable": True, "failure_reason": "",
        "detector_score": 0.9, "face_w": 100, "face_h": 100,
        "blur_score": 200.0, "detected_sex": "male",
        "bbox": [0, 0, 100, 100], "quality_band": "good", "quality_reasons": [],
        "landmarks_available": True,
        "embedding": (np.ones(4, dtype=np.float32) / 2),
    }


@pytest.fixture
def fake_manifest(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    embed_dir = tmp_path / "embeddings"
    manifest_dir.mkdir()
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake image bytes v1")

    with (manifest_dir / "am_persons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "image_path"])
        writer.writeheader()
        writer.writerow({"record_id": "AM1", "image_path": str(image_path)})

    monkeypatch.setattr(be, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(be, "EMBED_DIR", embed_dir)
    monkeypatch.setattr(be, "process_image", _fake_process_image)

    calls = {"n": 0}

    def _counting_get_app():
        calls["n"] += 1
        return object()

    monkeypatch.setattr(be, "get_app", _counting_get_app)

    return {"manifest_dir": manifest_dir, "embed_dir": embed_dir,
            "image_path": image_path, "get_app_calls": calls}


def _run(monkeypatch, argv_tail):
    monkeypatch.setattr(sys, "argv", ["build_embeddings.py", *argv_tail])
    be.main()


def test_second_run_without_force_preserves_metadata_exactly(fake_manifest, monkeypatch):
    _run(monkeypatch, ["--record-type", "am"])
    json_path = fake_manifest["embed_dir"] / "am" / "AM1.json"
    first = json.loads(json_path.read_text())
    assert first["usable"] is True
    assert fake_manifest["get_app_calls"]["n"] == 1

    _run(monkeypatch, ["--record-type", "am"])
    second = json.loads(json_path.read_text())

    assert first == second
    # The model should not have been reloaded for a pure cache hit.
    assert fake_manifest["get_app_calls"]["n"] == 1


def test_force_recomputes_even_when_nothing_changed(fake_manifest, monkeypatch):
    _run(monkeypatch, ["--record-type", "am"])
    json_path = fake_manifest["embed_dir"] / "am" / "AM1.json"
    first = json.loads(json_path.read_text())

    _run(monkeypatch, ["--record-type", "am", "--force"])
    second = json.loads(json_path.read_text())

    assert first["image_sha256"] == second["image_sha256"]
    assert first["processed_at"] != second["processed_at"]  # actually recomputed
    assert fake_manifest["get_app_calls"]["n"] == 2


def test_image_change_invalidates_cache(fake_manifest, monkeypatch):
    _run(monkeypatch, ["--record-type", "am"])
    json_path = fake_manifest["embed_dir"] / "am" / "AM1.json"
    first = json.loads(json_path.read_text())

    fake_manifest["image_path"].write_bytes(b"fake image bytes v2 -- different content")
    _run(monkeypatch, ["--record-type", "am"])
    second = json.loads(json_path.read_text())

    assert first["image_sha256"] != second["image_sha256"]
    assert first["processed_at"] != second["processed_at"]
    assert fake_manifest["get_app_calls"]["n"] == 2


def test_model_version_change_invalidates_cache(fake_manifest, monkeypatch):
    _run(monkeypatch, ["--record-type", "am"])
    json_path = fake_manifest["embed_dir"] / "am" / "AM1.json"
    first = json.loads(json_path.read_text())

    monkeypatch.setattr(be, "FACE_MODEL_VERSION", "some-other-model-version")
    _run(monkeypatch, ["--record-type", "am"])
    second = json.loads(json_path.read_text())

    assert first["image_sha256"] == second["image_sha256"]  # image unchanged
    assert second["model_version"] == "some-other-model-version"
    assert fake_manifest["get_app_calls"]["n"] == 2
