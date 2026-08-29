"""
P0 vision step: run face detection + quality gate + ArcFace embedding
(InsightFace buffalo_l) over one manifest's images (data/manifests/am_persons.csv
or pm_persons.csv), per docs/project_requirement.md sections 8-10.

Writes, per record:
    data/embeddings/{am,pm}/{record_id}.npy   -- L2-normalized 512-d embedding
                                                  of the highest-confidence face
And one summary:
    data/manifests/{am,pm}_embeddings_index.csv

A record with no usable face (none detected, too small, low detector
confidence) gets usable=False and no .npy file -- it's excluded from FAISS
but still shows up in the index CSV with a failure_reason, per the plan's
"inconclusive is a valid result" rule (section 9).

Usage:
    uv run python scripts/build_embeddings.py --record-type am
    uv run python scripts/build_embeddings.py --record-type pm
    uv run python scripts/build_embeddings.py --record-type am --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
EMBED_DIR = ROOT / "data" / "embeddings"

MODEL_NAME = "buffalo_l"
MODEL_VERSION = "insightface-buffalo_l-arcface"

MIN_FACE_SIZE = 40         # px, either dimension below this -> unusable
MIN_DET_SCORE = 0.5


def get_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def blur_score(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def process_image(app, image_path: Path) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"usable": False, "failure_reason": "unreadable_image"}

    faces = app.get(img)
    if not faces:
        return {"usable": False, "failure_reason": "no_face_detected"}

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

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return {"usable": False, "failure_reason": "face_too_small",
                "detector_score": float(face.det_score), "face_w": w, "face_h": h,
                "detected_sex": detected_sex}
    if face.det_score < MIN_DET_SCORE:
        return {"usable": False, "failure_reason": "low_detector_confidence",
                "detector_score": float(face.det_score), "face_w": w, "face_h": h,
                "detected_sex": detected_sex}

    crop_gray = cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
    b_score = blur_score(crop_gray)

    embedding = face.normed_embedding.astype(np.float32)  # already L2-normalized

    return {
        "usable": True,
        "failure_reason": "",
        "detector_score": float(face.det_score),
        "face_w": w,
        "face_h": h,
        "blur_score": b_score,
        "detected_sex": detected_sex,
        "embedding": embedding,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-type", choices=["am", "pm"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if .npy already exists")
    args = ap.parse_args()

    manifest_path = MANIFEST_DIR / f"{args.record_type}_persons.csv"
    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit)

    out_dir = EMBED_DIR / args.record_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading InsightFace ({MODEL_NAME})...")
    app = get_app()

    rows = []
    n_usable = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        record_id = row.record_id
        image_path_raw = getattr(row, "image_path", None)
        npy_path = out_dir / f"{record_id}.npy"

        if not image_path_raw or (isinstance(image_path_raw, float)):
            rows.append({"record_id": record_id, "usable": False,
                         "failure_reason": "no_image_path"})
            continue

        image_path = ROOT / image_path_raw
        if not args.force and npy_path.exists():
            rows.append({"record_id": record_id, "usable": True, "failure_reason": "",
                         "detector_score": "", "blur_score": "", "detected_sex": "", "cached": True})
            n_usable += 1
            continue

        if not image_path.exists():
            rows.append({"record_id": record_id, "usable": False,
                         "failure_reason": "image_file_missing"})
            continue

        result = process_image(app, image_path)
        if result["usable"]:
            np.save(npy_path, result["embedding"])
            n_usable += 1
        rows.append({
            "record_id": record_id,
            "usable": result["usable"],
            "failure_reason": result.get("failure_reason", ""),
            "detector_score": result.get("detector_score", ""),
            "blur_score": result.get("blur_score", ""),
            "detected_sex": result.get("detected_sex", ""),
        })

        if i % 100 == 0:
            print(f"[{args.record_type}] {i}/{len(df)} processed, {n_usable} usable so far")

    idx_path = MANIFEST_DIR / f"{args.record_type}_embeddings_index.csv"
    with idx_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "usable", "failure_reason",
                                                 "detector_score", "blur_score", "detected_sex"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print(f"[{args.record_type}] {n_usable}/{len(df)} usable faces, model={MODEL_VERSION}")
    print(f"wrote embeddings index -> {idx_path}")


if __name__ == "__main__":
    main()
