"""P0.5 local review API + static UI server, per docs/project_requirement.md
section 15/16 and docs/P0_5_IMPLEMENTATION.md Phases 6/9/10/17/18 -- serves
candidate shortlists (face+metadata, or metadata-only per Phase 9 when the face
pipeline hasn't found a usable face) for human review, computed on demand via
dvi.retrieval instead of only from a precomputed CSV. Local-only tool over
already-scraped data/, never returns an "identity"/"confidence" claim.

Run:
    uv run main.py
    # or, for autoreload during development:
    uv run uvicorn main:app --reload --port 8001

Then open http://localhost:8001/ for the review UI.
"""
from __future__ import annotations

import csv
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dvi import retrieval
from dvi.models import FACE_MODEL_VERSION, RANKING_MODEL_VERSION

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST_DIR = DATA / "manifests"
IMAGES_ROOT = DATA / "raw" / "images"
WEB_DIR = ROOT / "web"

# Local prototype default: permissive CORS. Set APP_ENV=production (and
# ALLOWED_ORIGINS="https://a.example,https://b.example") before exposing this
# beyond localhost -- see docs/P0_5_IMPLEMENTATION.md Phase 17.
APP_ENV = os.environ.get("APP_ENV", "local")

app = FastAPI(title="DVI Candidate Search API (local prototype)")

if APP_ENV == "local":
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
else:
    _origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                        allow_headers=["*"])

RANKING_DISCLAIMER = (
    "⚠ Investigative Candidate — Not an Identification. Ranking is generated from "
    "facial and metadata similarity. Do not communicate identity to families based on "
    "this result. Identification requires authorized forensic confirmation "
    "(fingerprint, dental, or DNA). Ranking score is UNCALIBRATED — not an identity "
    "probability."
)

# ---- Transient upload search (photo + optional name/sex -> face+gender matching) ----
# Uses the same InsightFace buffalo_l model as scripts/build_embeddings.py, but
# on an ephemeral uploaded image that is never persisted to data/. If the upload
# contains no usable face, the endpoint falls back to metadata-only (sex) ranking
# so a body with no metadata still participates (neutral 0.5 scores). Name is
# optional and currently informational (returned, not scored); future work could
# add fuzzy name similarity as a metadata signal.

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_SEX = {"", "male", "female", "other", "unspecified"}

_face_app = None  # lazy singleton -- ~280MB model, don't load at import time


def _get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    from insightface.app import FaceAnalysis

    from dvi.models import FACE_MODEL_NAME

    app = FaceAnalysis(name=FACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    _face_app = app
    return _face_app


def _process_upload_bytes(image_bytes: bytes) -> dict:
    """Detects the best face in an uploaded image and returns a dict with
    embedding (np.ndarray or None), detected_sex, has_face, quality_band,
    failure_reason, detector_score. Mirrors scripts/build_embeddings.process_image
    quality logic but works from bytes with no file persistence."""
    import cv2
    import numpy as np

    from dvi.quality import assess_quality

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "has_face": False,
            "embedding": None,
            "detected_sex": "",
            "quality_band": "unusable",
            "failure_reason": "unreadable_image",
            "detector_score": None,
        }

    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        return {
            "has_face": False,
            "embedding": None,
            "detected_sex": "",
            "quality_band": "unusable",
            "failure_reason": "no_face_detected",
            "detector_score": None,
        }

    face = max(faces, key=lambda f: float(getattr(f, "det_score", 0)))
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    w, h = x2 - x1, y2 - y1

    detected_sex = ""
    gender = getattr(face, "gender", None)
    if gender is not None:
        try:
            detected_sex = "male" if int(gender) == 1 else "female"
        except Exception:
            detected_sex = ""
    landmarks_available = getattr(face, "kps", None) is not None
    quality = assess_quality(
        detector_score=float(face.det_score),
        face_w=w,
        face_h=h,
        blur_score=None,
        landmarks_available=landmarks_available,
    )
    if not quality.usable:
        return {
            "has_face": False,
            "embedding": None,
            "detected_sex": detected_sex,
            "quality_band": quality.quality_band,
            "failure_reason": quality.quality_reasons[0] if quality.quality_reasons else "unusable",
            "detector_score": float(face.det_score),
        }

    # Re-compute blur on crop for band info (not a hard gate by default)
    try:
        crop_gray = cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
        b_score = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var()) if crop_gray.size else None
    except Exception:
        b_score = None

    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
        if embedding is not None:
            import numpy as _np

            n = _np.linalg.norm(embedding)
            if n > 0:
                embedding = embedding / n
    if embedding is not None:
        import numpy as _np

        embedding = _np.asarray(embedding, dtype=_np.float32)

    return {
        "has_face": True if embedding is not None else False,
        "embedding": embedding,
        "detected_sex": detected_sex,
        "quality_band": quality.quality_band,
        "failure_reason": "",
        "detector_score": float(face.det_score),
        "blur_score": b_score,
    }


def image_url(image_path) -> str | None:
    if not image_path or (isinstance(image_path, float)):
        return None
    rel = str(image_path)
    prefix = "data/raw/images/"
    rel = rel.removeprefix(prefix)
    return f"/images/{rel}"


def _list_records(record_type: str, limit: int, offset: int) -> dict:
    records = retrieval.load_records(record_type)
    ids = list(records.keys())
    total = len(ids)
    rows = []
    for rid in ids[offset: offset + limit]:
        r = dict(records[rid])
        r["image_url"] = image_url(r.get("image_path"))
        usable = r.get("usable")
        r["has_face"] = bool(usable) if usable != "" else False
        rows.append(r)
    return {"total": total, "items": rows}


@app.get("/api/health")
def health():
    return {"status": "ok", "app_env": APP_ENV}


@app.get("/api/stats")
def stats():
    am = retrieval.load_records("am")
    pm = retrieval.load_records("pm")
    am_usable = sum(1 for r in am.values() if r.get("usable") is True)
    pm_usable = sum(1 for r in pm.values() if r.get("usable") is True)
    _, am_id_map = retrieval.load_index("am")
    _, pm_id_map = retrieval.load_index("pm")

    review_count = 0
    reviews_path = MANIFEST_DIR / "reviews.csv"
    if reviews_path.exists():
        with reviews_path.open(encoding="utf-8") as f:
            review_count = sum(1 for _ in csv.DictReader(f))

    return {
        "am_records": len(am),
        "pm_records": len(pm),
        "am_usable_faces": am_usable,
        "pm_usable_faces": pm_usable,
        "am_index_size": len(am_id_map),
        "pm_index_size": len(pm_id_map),
        "review_count": review_count,
        "face_model_version": FACE_MODEL_VERSION,
        "ranking_model_version": RANKING_MODEL_VERSION,
    }


@app.get("/api/pm")
def list_pm(limit: int = 500, offset: int = 0):
    return _list_records("pm", limit, offset)


@app.get("/api/am")
def list_am(limit: int = 500, offset: int = 0):
    return _list_records("am", limit, offset)


def _provenance(c: dict, cand_rec: dict, query_rec: dict) -> dict:
    """Human-readable provenance for why a candidate was ranked."""
    reasons = []
    if c.get("face_score") is not None and c["face_score"] >= 0.7:
        reasons.append("High facial similarity")
    elif c.get("face_score") is not None and c["face_score"] >= 0.5:
        reasons.append("Moderate facial similarity")
    if c.get("location_score", 0) >= 0.8:
        reasons.append("Location compatibility")
    if c.get("date_score", 0) >= 0.7:
        reasons.append("Temporal compatibility")
    if c.get("age_score", 0) >= 0.7:
        reasons.append("Age compatibility")
    if not reasons:
        reasons.append("Combined face + metadata ranking")
    missing = []
    if not cand_rec.get("height_cm"):
        missing.append("height")
    if not cand_rec.get("clothing"):
        missing.append("clothing description")
    if not cand_rec.get("distinguishing_marks"):
        missing.append("distinguishing marks")
    # Forensic signals always missing from this system
    missing.extend(["fingerprint", "dental", "DNA"])
    return {"reasons": reasons, "missing_evidence": missing}


def _candidate_row(other: dict) -> dict:
    return {
        "record_id": other.get("record_id", ""),
        "name": other.get("name", ""),
        "sex": other.get("sex", ""),
        "age_min": other.get("age_min", ""),
        "age_max": other.get("age_max", ""),
        "height_cm": other.get("height_cm", ""),
        "location": other.get("location", ""),
        "event_date": other.get("event_date", ""),
        "event_date_normalized": other.get("event_date_normalized", ""),
        "calendar_type": other.get("calendar_type", ""),
        "clothing": other.get("clothing", ""),
        "distinguishing_marks": other.get("distinguishing_marks", ""),
        "image_url": image_url(other.get("image_path")),
    }


def _get_candidates(query_type: str, query_id: str, top_k: int) -> dict:
    candidate_type = "pm" if query_type == "am" else "am"
    try:
        result = retrieval.search_query(query_type, query_id, candidate_type, top_k=top_k)
    except KeyError:
        raise HTTPException(404, f"{query_type} record '{query_id}' not found")

    candidate_records = retrieval.load_records(candidate_type)
    query_records = retrieval.load_records(query_type)
    query_rec = query_records.get(query_id, {})
    candidates = []
    for c in result["candidates"]:
        cand_id = c["candidate_record_id"]
        rec = candidate_records.get(cand_id)
        if rec is None:
            continue
        prov = _provenance(c, rec, query_rec)
        candidates.append({
            "rank": c["rank"],
            f"{candidate_type}_record_id": cand_id,
            **_candidate_row(rec),
            "face_score": c["face_score"],
            "metadata_score": c["metadata_score"],
            "final_score": c["final_score"],
            "sex_score": c.get("sex_score"),
            "age_score": c.get("age_score"),
            "height_score": c.get("height_score"),
            "date_score": c.get("date_score"),
            "location_score": c.get("location_score"),
            "am_metadata_vision_conflict": c["am_metadata_vision_conflict"],
            "pm_metadata_vision_conflict": c["pm_metadata_vision_conflict"],
            "pair_metadata_conflict": c["pair_metadata_conflict"],
            "pair_vision_conflict": c["pair_vision_conflict"],
            "provenance": prov,
        })

    return {
        f"{query_type}_record_id": query_id,
        "source": result["candidate_source"],
        "has_face": result["has_face"],
        "candidates": candidates,
        "disclaimer": RANKING_DISCLAIMER,
    }


@app.get("/api/pm/{pm_id}/candidates")
def get_pm_candidates(pm_id: str, top_k: int = 20):
    return _get_candidates("pm", pm_id, top_k)


@app.get("/api/am/{am_id}/candidates")
def get_am_candidates(am_id: str, top_k: int = 20):
    return _get_candidates("am", am_id, top_k)


@app.post("/api/search/upload")
async def search_upload(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    sex: str = Form(default=""),
    target: str = Form(default="pm"),
    top_k: int = Form(default=20),
):
    """Upload a face photo (+ optional name/sex) and search the opposite gallery.

    - file: image/jpeg, image/png, or image/webp (max 10 MB). Face is detected
      on the fly using the same buffalo_l ArcFace pipeline as the batch
      embedding job; the file is never written to data/.
    - name: optional display name (informational, not scored).
    - sex: optional provided sex (male/female/other/unspecified). Blended with
      vision-estimated sex via dvi.scoring.sex_score; if the PM body has no
      metadata at all, this still provides a soft gender filter (face+sex only).
    - target: gallery to search ("pm" = unidentified bodies (default, family
      searching for a missing person), "am" = missing persons).
    - top_k: 1..100, default 20.
    Returns the same candidate shape as the other candidate endpoints, plus a
    `query` block describing what was extracted from the upload.
    """
    target = (target or "pm").lower().strip()
    if target not in ("pm", "am"):
        raise HTTPException(400, "target must be 'pm' or 'am'")
    candidate_type = target
    query_type = "am" if candidate_type == "pm" else "pm"

    # Normalize sex: allow empty/unspecified to mean "unknown"
    sex_norm = (sex or "").strip().lower()
    if sex_norm == "unspecified":
        sex_norm = ""
    if sex_norm not in ALLOWED_SEX:
        raise HTTPException(400, "sex must be male|female|other|unspecified (or empty)")

    try:
        top_k = int(top_k)
    except Exception:
        raise HTTPException(400, "top_k must be an integer")
    top_k = max(1, min(100, top_k))

    if file is None or not file.filename:
        raise HTTPException(400, "file is required")

    # Basic mime/extension allowlist -- cv2 will still be the final validator
    fname_lower = (file.filename or "").lower()
    allowed_ext = (".jpg", ".jpeg", ".png", ".webp")
    if not any(fname_lower.endswith(ext) for ext in allowed_ext):
        # Also check content_type as fallback, but don't hard-reject if cv2 can decode
        ctype = (file.content_type or "").lower()
        if ctype not in ("image/jpeg", "image/png", "image/webp", "image/jpg", ""):
            raise HTTPException(400, "file must be a JPEG, PNG, or WebP image")

    data = await file.read()
    if not data:
        raise HTTPException(400, "uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"file too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")

    # Extract face embedding + detected sex from the upload (ephemeral)
    info = _process_upload_bytes(data)
    has_face = bool(info.get("has_face") and info.get("embedding") is not None)
    embedding = info.get("embedding") if has_face else None
    detected_sex = info.get("detected_sex") or ""

    query_rec: dict = {
        "name": (name or "").strip(),
        "sex": sex_norm,
        "detected_sex": detected_sex,
        # other scoring fields absent -> neutral 0.5 in dvi.scoring
    }

    result = retrieval.search_transient(
        query_rec, embedding, candidate_type=candidate_type, query_type=query_type, top_k=top_k
    )

    candidate_records = retrieval.load_records(candidate_type)
    candidates = []
    for c in result["candidates"]:
        cand_id = c["candidate_record_id"]
        rec = candidate_records.get(cand_id)
        if rec is None:
            continue
        prov = _provenance(c, rec, query_rec)
        candidates.append({
            "rank": c["rank"],
            f"{candidate_type}_record_id": cand_id,
            **_candidate_row(rec),
            "face_score": c["face_score"],
            "metadata_score": c["metadata_score"],
            "final_score": c["final_score"],
            "sex_score": c.get("sex_score"),
            "age_score": c.get("age_score"),
            "height_score": c.get("height_score"),
            "date_score": c.get("date_score"),
            "location_score": c.get("location_score"),
            "am_metadata_vision_conflict": c["am_metadata_vision_conflict"],
            "pm_metadata_vision_conflict": c["pm_metadata_vision_conflict"],
            "pair_metadata_conflict": c["pair_metadata_conflict"],
            "pair_vision_conflict": c["pair_vision_conflict"],
            "provenance": prov,
        })

    return {
        "query": {
            "provided_name": (name or "").strip() or None,
            "provided_sex": sex_norm or None,
            "detected_sex": detected_sex or None,
            "has_face": has_face,
            "quality_band": info.get("quality_band"),
            "failure_reason": info.get("failure_reason") or None,
            "detector_score": info.get("detector_score"),
        },
        "candidate_type": candidate_type,
        "query_type": query_type,
        "source": result["candidate_source"],
        "has_face": has_face,
        "candidates": candidates,
        "disclaimer": RANKING_DISCLAIMER,
    }


DECISION_CANONICAL = {
    "potential": "potential",
    "candidate_for_further_examination": "potential",
    "candidate for further examination": "potential",
    "rejected": "rejected",
    "not_likely_candidate": "rejected",
    "not a likely candidate": "rejected",
    "inconclusive": "inconclusive",
    "insufficient_evidence": "inconclusive",
    "insufficient evidence": "inconclusive",
}

FORENSIC_STATUSES = {
    "not_examined", "fingerprint_requested", "dental_requested", "dna_requested",
    "excluded", "confirmed_externally", ""
}


class ReviewPayload(BaseModel):
    pm_record_id: str
    am_record_id: str
    decision: str  # canonical or new workflow labels
    forensic_status: str = ""  # optional forensic workflow status
    notes: str = ""


REVIEW_FIELDNAMES = ["review_id", "pm_record_id", "am_record_id", "decision", "forensic_status", "notes",
                       "reviewed_at", "ranking_model_version", "face_model_version"]


@app.post("/api/review")
def submit_review(payload: ReviewPayload):
    dec_norm = payload.decision.strip().lower().replace(" ", "_").replace("-", "_")
    canonical = DECISION_CANONICAL.get(dec_norm)
    # also try raw lower
    if canonical is None:
        canonical = DECISION_CANONICAL.get(payload.decision.strip().lower())
    if canonical is None:
        raise HTTPException(400, "decision must be one of: Candidate for further examination, Not a likely candidate, Insufficient evidence (or legacy: potential|rejected|inconclusive)")
    forensic_norm = (payload.forensic_status or "").strip().lower().replace(" ", "_").replace("-", "_")
    if forensic_norm not in FORENSIC_STATUSES:
        raise HTTPException(400, f"forensic_status must be one of: {', '.join(sorted(s for s in FORENSIC_STATUSES if s))} or empty")
    reviews_path = MANIFEST_DIR / "reviews.csv"
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not reviews_path.exists()
    with reviews_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        if write_header:
            writer.writeheader()
        dec_norm = payload.decision.strip().lower().replace(" ", "_").replace("-", "_")
        canonical = DECISION_CANONICAL.get(dec_norm, DECISION_CANONICAL.get(payload.decision.strip().lower(), payload.decision))
        forensic_norm = (payload.forensic_status or "").strip().lower().replace(" ", "_").replace("-", "_")
        writer.writerow({
            "review_id": str(uuid.uuid4()),
            "pm_record_id": payload.pm_record_id,
            "am_record_id": payload.am_record_id,
            "decision": canonical,
            "forensic_status": forensic_norm,
            "notes": payload.notes,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "ranking_model_version": RANKING_MODEL_VERSION,
            "face_model_version": FACE_MODEL_VERSION,
        })
    return {"ok": True}


# Order matters: specific routes/mounts above, static catch-alls last.
IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_ROOT)), name="images")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
