"""P0.5 vision step: run face detection + quality assessment + ArcFace embedding
(InsightFace buffalo_l) over one manifest's images (data/manifests/am_persons.csv
or pm_persons.csv), per docs/project_requirement.md sections 8-10 and
docs/P0_5_IMPLEMENTATION.md Phases 3-4.

Writes, per record:
    data/embeddings/{am,pm}/{record_id}.npy   -- L2-normalized 512-d embedding
                                                  of the highest-confidence face
    data/embeddings/{am,pm}/{record_id}.json  -- full metadata sidecar (Phase 3):
                                                  model name/version, image sha256,
                                                  detector_score, bbox, face size,
                                                  blur_score, detected_sex, usable,
                                                  failure_reason, quality_band,
                                                  quality_reasons, processed_at
And one summary rebuilt from the sidecars on every run:
    data/manifests/{am,pm}_embeddings_index.csv

Caching (Phase 3): a record is NOT recomputed if its image's sha256 is
unchanged AND FACE_MODEL_VERSION is unchanged AND (for a previously-usable
record) its .npy still exists -- the cached .json is reused byte-for-byte, so
re-running this script twice with no image/model changes produces identical
metadata. Pass --force to recompute everything regardless.

A record with no usable face (none detected, too small, low detector
confidence) gets usable=False and no .npy file -- it's excluded from FAISS
but still shows up in the index CSV with a failure_reason, per the plan's
"inconclusive is a valid result" rule (section 9). See dvi/quality.py for the
quality_band/quality_reasons logic and for why the thresholds below are
provisional engineering defaults, not validated forensic cutoffs.

Usage:
    uv run python scripts/build_embeddings.py --record-type am
    uv run python scripts/build_embeddings.py --record-type pm
    uv run python scripts/build_embeddings.py --record-type am --limit 20   # smoke test
    uv run python scripts/build_embeddings.py --record-type am --force      # recompute everything
    uv run python scripts/build_embeddings.py --record-type pm --min-blur-score 50
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvi.models import FACE_MODEL_NAME, FACE_MODEL_VERSION
from dvi.quality import (
    DEFAULT_MIN_BLUR_SCORE,
    DEFAULT_MIN_DET_SCORE,
    DEFAULT_MIN_FACE_SIZE,
    assess_quality,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
EMBED_DIR = ROOT / "data" / "embeddings"

CSV_FIELDNAMES = ["record_id", "usable", "failure_reason", "detector_score",
                   "face_width", "face_height", "blur_score", "detected_sex",
                   "quality_band", "processed_at"]


def get_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=FACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blur_score(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def process_image(app, image_path: Path, *, min_face_size: float, min_det_score: float,
                   min_blur_score: float | None) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"usable": False, "failure_reason": "unreadable_image",
                "quality_band": "unusable", "quality_reasons": ["unreadable_image"]}

    faces = app.get(img)
    if not faces:
        return {"usable": False, "failure_reason": "no_face_detected",
                "quality_band": "unusable", "quality_reasons": ["no_face_detected"]}

    face = max(faces, key=lambda f: f.det_score)
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    w, h = x2 - x1, y2 - y1

    # genderage.onnx runs as part of the same buffalo_l forward pass that
    # produced bbox/landmarks/embedding above -- reading it here is free,
    # not an extra detection pass. Used as an independent cross-check
    # against the scraped metadata's sex field, never as a hard filter.
    detected_sex = ""
    gender = getattr(face, "gender", None)
    if gender is not None:
        detected_sex = "male" if int(gender) == 1 else "female"
    landmarks_available = getattr(face, "kps", None) is not None

    quality = assess_quality(
        detector_score=float(face.det_score), face_w=w, face_h=h, blur_score=None,
        landmarks_available=landmarks_available,
        min_face_size=min_face_size, min_det_score=min_det_score, min_blur_score=min_blur_score,
    )
    if not quality.usable:
        return {"usable": False, "failure_reason": quality.quality_reasons[0],
                "detector_score": float(face.det_score), "face_w": w, "face_h": h,
                "detected_sex": detected_sex, "bbox": [x1, y1, x2, y2],
                "quality_band": quality.quality_band, "quality_reasons": quality.quality_reasons}

    crop_gray = cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
    b_score = blur_score(crop_gray)
    # re-assess now that blur_score is available, in case a blur threshold was configured
    quality = assess_quality(
        detector_score=float(face.det_score), face_w=w, face_h=h, blur_score=b_score,
        landmarks_available=landmarks_available,
        min_face_size=min_face_size, min_det_score=min_det_score, min_blur_score=min_blur_score,
    )

    embedding = face.normed_embedding.astype(np.float32)  # already L2-normalized

    return {
        "usable": quality.usable,
        "failure_reason": "" if quality.usable else (quality.quality_reasons[0] if quality.quality_reasons else ""),
        "detector_score": float(face.det_score),
        "face_w": w, "face_h": h,
        "blur_score": b_score,
        "detected_sex": detected_sex,
        "bbox": [x1, y1, x2, y2],
        "quality_band": quality.quality_band,
        "quality_reasons": quality.quality_reasons,
        "landmarks_available": landmarks_available,
        "embedding": embedding,
    }


def load_cached_metadata(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-type", choices=["am", "pm"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if a valid cache entry exists")
    ap.add_argument("--min-face-size", type=float, default=DEFAULT_MIN_FACE_SIZE,
                     help="px, either bbox dimension below this -> unusable (engineering default, not validated)")
    ap.add_argument("--min-det-score", type=float, default=DEFAULT_MIN_DET_SCORE,
                     help="InsightFace det_score below this -> unusable (engineering default, not validated)")
    ap.add_argument("--min-blur-score", type=float, default=DEFAULT_MIN_BLUR_SCORE,
                     help="Laplacian-variance floor for quality banding; unset (None) does not reject on blur")
    args = ap.parse_args()

    manifest_path = MANIFEST_DIR / f"{args.record_type}_persons.csv"
    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit)

    out_dir = EMBED_DIR / args.record_type
    out_dir.mkdir(parents=True, exist_ok=True)

    app = None  # lazy-loaded on first cache miss only
    rows: list[dict] = []
    n_usable = 0
    n_cached = 0
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False), 1):
        record_id = row.record_id
        image_path_raw = getattr(row, "image_path", None)
        npy_path = out_dir / f"{record_id}.npy"
        json_path = out_dir / f"{record_id}.json"

        if not image_path_raw or isinstance(image_path_raw, float):
            rows.append({"record_id": record_id, "usable": False, "failure_reason": "no_image_path"})
            continue

        image_path = ROOT / image_path_raw
        if not image_path.exists():
            rows.append({"record_id": record_id, "usable": False, "failure_reason": "image_file_missing"})
            continue

        current_sha256 = sha256_file(image_path)
        cached = None if args.force else load_cached_metadata(json_path)
        cache_valid = (
            cached is not None
            and cached.get("image_sha256") == current_sha256
            and cached.get("model_version") == FACE_MODEL_VERSION
            and (not cached.get("usable") or npy_path.exists())
        )

        if cache_valid:
            rows.append(cached)
            if cached.get("usable"):
                n_usable += 1
            n_cached += 1
            continue

        if app is None:
            print(f"loading InsightFace ({FACE_MODEL_NAME})...")
            app = get_app()

        result = process_image(app, image_path, min_face_size=args.min_face_size,
                                min_det_score=args.min_det_score, min_blur_score=args.min_blur_score)
        meta = {
            "record_id": record_id,
            "model_name": FACE_MODEL_NAME,
            "model_version": FACE_MODEL_VERSION,
            "embedding_dimension": int(result["embedding"].shape[0]) if result.get("usable") else None,
            "image_sha256": current_sha256,
            "detector_score": result.get("detector_score"),
            "bbox": result.get("bbox"),
            "face_width": result.get("face_w"),
            "face_height": result.get("face_h"),
            "blur_score": result.get("blur_score"),
            "detected_sex": result.get("detected_sex", ""),
            "usable": result["usable"],
            "failure_reason": result.get("failure_reason", ""),
            "quality_band": result.get("quality_band", "unusable"),
            "quality_reasons": result.get("quality_reasons", []),
            "landmarks_available": result.get("landmarks_available"),
            "processed_at": datetime.now(UTC).isoformat(),
        }
        if result["usable"]:
            np.save(npy_path, result["embedding"])
            n_usable += 1
        elif npy_path.exists():
            npy_path.unlink()  # stale embedding from a prior run whose image/model has since changed
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(meta)

        if i % 100 == 0 or i == total:
            print(f"[{args.record_type}] {i}/{total} processed, {n_usable} usable so far "
                  f"({n_cached} reused from cache)")

    idx_path = MANIFEST_DIR / f"{args.record_type}_embeddings_index.csv"
    with idx_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDNAMES})

    print(f"[{args.record_type}] {n_usable}/{total} usable faces ({n_cached} reused from cache), "
          f"model={FACE_MODEL_VERSION}")
    print(f"wrote embeddings index -> {idx_path}")


if __name__ == "__main__":
    main()
