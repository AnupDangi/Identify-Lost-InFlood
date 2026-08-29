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
from fastapi import FastAPI, HTTPException
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
    "Ranking score is not an identity probability. This is a shortlist for human "
    "review only -- final identification requires fingerprint, dental, or DNA evidence."
)


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
    candidates = []
    for c in result["candidates"]:
        cand_id = c["candidate_record_id"]
        rec = candidate_records.get(cand_id)
        if rec is None:
            continue
        candidates.append({
            "rank": c["rank"],
            f"{candidate_type}_record_id": cand_id,
            **_candidate_row(rec),
            "face_score": c["face_score"],
            "metadata_score": c["metadata_score"],
            "final_score": c["final_score"],
            "am_metadata_vision_conflict": c["am_metadata_vision_conflict"],
            "pm_metadata_vision_conflict": c["pm_metadata_vision_conflict"],
            "pair_metadata_conflict": c["pair_metadata_conflict"],
            "pair_vision_conflict": c["pair_vision_conflict"],
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


class ReviewPayload(BaseModel):
    pm_record_id: str
    am_record_id: str
    decision: str  # potential | rejected | inconclusive
    notes: str = ""


REVIEW_FIELDNAMES = ["review_id", "pm_record_id", "am_record_id", "decision", "notes",
                      "reviewed_at", "ranking_model_version", "face_model_version"]


@app.post("/api/review")
def submit_review(payload: ReviewPayload):
    if payload.decision not in {"potential", "rejected", "inconclusive"}:
        raise HTTPException(400, "decision must be potential|rejected|inconclusive")
    reviews_path = MANIFEST_DIR / "reviews.csv"
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not reviews_path.exists()
    with reviews_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "review_id": str(uuid.uuid4()),
            "pm_record_id": payload.pm_record_id,
            "am_record_id": payload.am_record_id,
            "decision": payload.decision,
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
