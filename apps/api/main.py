"""
P0 local review API + static UI server, per docs/project_requirement.md
section 15/16 -- serves candidate shortlists (face+metadata, or metadata-only
if the face pipeline hasn't run yet) for human review. Local-only tool over
already-scraped data/, never returns an "identity"/"confidence" claim.

Run:
    uv run uvicorn apps.api.main:app --reload --port 8001

Then open http://localhost:8001/ for the review UI.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
MANIFEST_DIR = DATA / "manifests"
IMAGES_ROOT = DATA / "raw" / "images"
WEB_DIR = ROOT / "web"

app = FastAPI(title="DVI Candidate Review API (local prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def image_url(image_path) -> str | None:
    if not image_path or (isinstance(image_path, float)):
        return None
    rel = str(image_path)
    prefix = "data/raw/images/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return f"/images/{rel}"


def load_am() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_DIR / "am_persons.csv")
    return df.set_index("record_id", drop=False)


def load_pm() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_DIR / "pm_persons.csv")
    return df.set_index("record_id", drop=False)


def load_face_index(record_type: str) -> dict:
    """record_id -> {"has_face": bool, "detected_sex": str}, from
    scripts/build_embeddings.py's output. Missing entirely (pipeline not run
    yet) means has_face is unknown -- treated as True so the Prediction
    button still shows (falls back to metadata-only scoring) rather than
    hiding it for every record."""
    path = MANIFEST_DIR / f"{record_type}_embeddings_index.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[r["record_id"]] = {
            "has_face": bool(r["usable"]),
            "detected_sex": (r.get("detected_sex") or "") if pd.notna(r.get("detected_sex")) else "",
        }
    return out


def load_candidates() -> tuple[pd.DataFrame, str]:
    face_path = MANIFEST_DIR / "candidate_matches_face.csv"
    meta_path = MANIFEST_DIR / "candidate_matches.csv"
    if face_path.exists():
        return pd.read_csv(face_path), "face+metadata"
    if meta_path.exists():
        return pd.read_csv(meta_path), "metadata_only"
    return pd.DataFrame(), "none"


def _list_records(df: pd.DataFrame, record_type: str, limit: int, offset: int) -> dict:
    df = df.fillna("")
    face_idx = load_face_index(record_type)
    total = len(df)
    rows = df.iloc[offset: offset + limit].to_dict(orient="records")
    for r in rows:
        r["image_url"] = image_url(r.get("image_path"))
        face_info = face_idx.get(r["record_id"])
        r["has_face"] = face_info["has_face"] if face_info is not None else True
        r["detected_sex"] = face_info["detected_sex"] if face_info is not None else ""
    return {"total": total, "items": rows}


@app.get("/api/pm")
def list_pm(limit: int = 500, offset: int = 0):
    result = _list_records(load_pm(), "pm", limit, offset)
    _, source = load_candidates()
    result["candidate_source"] = source
    return result


@app.get("/api/am")
def list_am(limit: int = 500, offset: int = 0):
    result = _list_records(load_am(), "am", limit, offset)
    _, source = load_candidates()
    result["candidate_source"] = source
    return result


def _candidate_row(other: pd.Series) -> dict:
    other = other.fillna("")
    return {
        "record_id": other["record_id"],
        "name": other.get("name", ""),
        "sex": other["sex"],
        "age_min": other["age_min"],
        "age_max": other["age_max"],
        "height_cm": other.get("height_cm", ""),
        "location": other["location"],
        "event_date": other["event_date"],
        "clothing": other.get("clothing", ""),
        "distinguishing_marks": other.get("distinguishing_marks", ""),
        "image_url": image_url(other.get("image_path")),
    }


@app.get("/api/pm/{pm_id}/candidates")
def get_pm_candidates(pm_id: str, top_k: int = 20):
    pm_face = load_face_index("pm").get(pm_id)
    if pm_face is not None and not pm_face["has_face"]:
        raise HTTPException(422, "no usable face detected for this record -- prediction unavailable")

    am = load_am()
    cands, source = load_candidates()
    if cands.empty:
        raise HTTPException(
            404,
            "no candidate matches file found -- run scripts/compare_am_pm.py "
            "(metadata-only) or scripts/search_candidates.py (face+metadata) first",
        )
    sub = cands[cands["pm_record_id"] == pm_id].sort_values("rank").head(top_k)
    out = []
    for _, row in sub.iterrows():
        am_id = row["am_record_id"]
        if am_id not in am.index:
            continue
        out.append({
            "rank": int(row["rank"]),
            "am_record_id": am_id,
            **_candidate_row(am.loc[am_id]),
            "face_score": row.get("face_score", None),
            "metadata_score": row.get("metadata_score", None),
            "final_score": row["final_score"],
            "sex_conflict": bool(row.get("sex_conflict", False)),
        })
    return {"pm_record_id": pm_id, "source": source, "candidates": out}


@app.get("/api/am/{am_id}/candidates")
def get_am_candidates(am_id: str, top_k: int = 20):
    """Mirror of get_pm_candidates for the 'Missing' dashboard tab: which
    unidentified bodies rank this missing person as a candidate. Reuses the
    same pm->am shortlist file (face similarity is symmetric) rather than
    running a second FAISS search -- an AM record only appears here if some
    PM record's own top-K included it, which is the correct behavior for a
    shortlist tool (if no PM ranked it highly, there's nothing to surface)."""
    am_face = load_face_index("am").get(am_id)
    if am_face is not None and not am_face["has_face"]:
        raise HTTPException(422, "no usable face detected for this record -- prediction unavailable")

    pm = load_pm()
    cands, source = load_candidates()
    if cands.empty:
        raise HTTPException(
            404,
            "no candidate matches file found -- run scripts/compare_am_pm.py "
            "(metadata-only) or scripts/search_candidates.py (face+metadata) first",
        )
    sub = cands[cands["am_record_id"] == am_id].sort_values("final_score", ascending=False).head(top_k)
    out = []
    for rank, (_, row) in enumerate(sub.iterrows(), 1):
        pm_id = row["pm_record_id"]
        if pm_id not in pm.index:
            continue
        out.append({
            "rank": rank,
            "pm_record_id": pm_id,
            **_candidate_row(pm.loc[pm_id]),
            "face_score": row.get("face_score", None),
            "metadata_score": row.get("metadata_score", None),
            "final_score": row["final_score"],
            "sex_conflict": bool(row.get("sex_conflict", False)),
        })
    return {"am_record_id": am_id, "source": source, "candidates": out}


class ReviewPayload(BaseModel):
    pm_record_id: str
    am_record_id: str
    decision: str  # potential | rejected | inconclusive
    notes: str = ""


@app.post("/api/review")
def submit_review(payload: ReviewPayload):
    if payload.decision not in {"potential", "rejected", "inconclusive"}:
        raise HTTPException(400, "decision must be potential|rejected|inconclusive")
    reviews_path = MANIFEST_DIR / "reviews.csv"
    write_header = not reviews_path.exists()
    with reviews_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pm_record_id", "am_record_id", "decision", "notes", "reviewed_at"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "pm_record_id": payload.pm_record_id,
            "am_record_id": payload.am_record_id,
            "decision": payload.decision,
            "notes": payload.notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        })
    return {"ok": True}


# Order matters: specific routes/mounts above, static catch-alls last.
app.mount("/images", StaticFiles(directory=str(IMAGES_ROOT)), name="images")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
