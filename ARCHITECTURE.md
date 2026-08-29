# Architecture

## Goal and constraint

Retrieve a **Top-K shortlist** of candidate missing-person (AM) matches for each
unidentified-body (PM) record, and vice versa, for a human investigator to review.
The system never emits an "identity" or a "confidence" claim — see
[`docs/project_requirement.md`](docs/project_requirement.md) §15. Final identification
is a forensic decision (fingerprint/dental/DNA), not this tool's job.

## Data flow

```
Nepal Police UDB (missing/dead-bodies-lists)
        |
        v
scripts/scrape_udb.py  ----------------------------\
  (or an ad-hoc list scrape, normalized via          |
   scripts/import_pm_from_list_scrape.py)            |
        |                                            |
        v                                            v
data/manifests/am_persons.csv          data/manifests/pm_persons.csv
data/raw/images/am/{id}.jpg            data/raw/images/pm/...
        |                                            |
        v                                            v
scripts/build_embeddings.py  (InsightFace buffalo_l: detect, quality-gate, embed, gender)
        |                                            |
        v                                            v
data/embeddings/am/{id}.npy            data/embeddings/pm/{id}.npy
data/manifests/am_embeddings_index.csv data/manifests/pm_embeddings_index.csv
        |
        v
scripts/build_index.py --record-type am
        |
        v
data/index/am.index (FAISS IndexFlatIP)  +  data/index/am_id_map.json
        |
        v
scripts/search_candidates.py
  for each usable PM embedding:
    FAISS top-100 AM neighbors (cosine similarity)
      -> blend with metadata score (scripts/compare_am_pm.py)
      -> data/manifests/candidate_matches_face.csv  (pm_record_id, am_record_id,
         rank, face_score, metadata_score, final_score, sex_conflict, ...)
        |
        v
apps/api/main.py  (FastAPI)  --serves-->  web/index.html  (review UI)
        |
        v
data/manifests/reviews.csv  (Potential / Rejected / Inconclusive decisions)
```

If `search_candidates.py` hasn't been run yet (no face pipeline available),
`scripts/compare_am_pm.py` produces `candidate_matches.csv` (metadata-only, full
cross-product) as a fallback — the API prefers the face+metadata file when both exist.

## Why these pieces, and not others

- **InsightFace `buffalo_l`** bundles SCRFD detection, landmarks, ArcFace recognition,
  and a gender/age model in one forward pass — P0 needs exactly this, not five separate
  libraries (per the design doc's "don't integrate everything on day one" guidance).
- **FAISS `IndexFlatIP`**, not IVF/HNSW: exact search is fine up to ~100K vectors, and AM
  is ~9K. Premature approximate-search tuning would trade correctness for speed nobody
  needs yet.
- **One-directional FAISS index (AM only)**: the natural disaster workflow is "found a
  body, who could this be" (PM→AM). The API's `Missing` tab (AM→PM) reuses the same
  `candidate_matches_face.csv`, pivoted by `am_record_id`, instead of a second FAISS
  index — cosine similarity is symmetric, so this is exactly the same score, just a
  different sort key. The tradeoff: an AM record only shows candidates if some PM
  record's own top-100 FAISS search happened to include it. That's the correct behavior
  for a shortlist tool (nothing ranked it, so there's nothing to surface), not a bug.

## Quality gate (why PM has fewer usable faces than AM)

`build_embeddings.py` marks a face unusable (and excludes it from the FAISS index/search
entirely, rather than forcing a low-quality match) when:

- no face detected at all,
- the detected face is smaller than 40px in either dimension,
- detector confidence is below 0.5.

Post-mortem/disaster photographs are inherently harder to enroll — occlusion, injury,
poor lighting, camera angle. `has_face: false` is a valid, expected outcome per the design
doc's "inconclusive is a valid result" principle, not a pipeline failure. The review UI
hides the Prediction button entirely for those records rather than showing a meaningless
result, and the API returns `422` if the endpoint is called directly for one.

## Scoring model

```
face_score      = cosine_similarity(am_embedding, pm_embedding) rescaled from [-1,1] to [0,1]
metadata_score  = 0.30 * sex_score + 0.25 * age_score + 0.10 * height_score
                + 0.15 * date_score + 0.20 * location_score
final_score     = face_weight * face_score + (1 - face_weight) * metadata_score   (face_weight = 0.6 default)
```

All sub-scores are **soft compatibility bands**, not hard filters (design doc §14): a
20-year age gap ranks lower than a 2-year gap, it doesn't disqualify the pair, because
post-mortem age estimation is frequently wrong. The weights above are hand-picked
starting points for experimentation, explicitly **not** claimed to be statistically
calibrated — the design doc's recommended next step is logistic regression against
labeled review outcomes once `reviews.csv` has enough data to fit one.

`sex_score` blends two independent signals when both are available: the scraped
metadata's stated sex, and each side's InsightFace-detected sex from the photo itself.
Each contributes equally; a pair whose metadata agrees but whose photos read as different
sexes scores lower than a metadata-only match would (`sex_conflict: true` surfaces this
to the reviewer), and the reverse also holds. This is a cross-check against data-entry
error, not a new hard filter — same philosophy as age/height.

## API surface

| Endpoint | Purpose |
|---|---|
| `GET /api/pm`, `GET /api/am` | List records with `has_face`/`detected_sex` merged in |
| `GET /api/pm/{id}/candidates` | Top-K AM candidates for a PM record (422 if no face) |
| `GET /api/am/{id}/candidates` | Top-K PM candidates for an AM record (422 if no face) |
| `POST /api/review` | Log a Potential/Rejected/Inconclusive decision |
| `GET /images/...` | Serves `data/raw/images/` directly (local-only tool) |

## Known limitations (P0)

- PM records scraped via the list-page fallback (`import_pm_from_list_scrape.py`) lack
  height, clothing, and distinguishing-marks fields — those only exist on each record's
  detail page, which `scrape_udb.py --record-type pm` would need to fetch separately.
- Some PM `event_date` values are still in Bikram Sambat rather than converted to
  Gregorian (visible as implausible years like 2082 in `pm_persons.csv`) — `date_score`
  treats these as low-confidence rather than crashing, but they're not corrected.
- `final_score` weights are heuristic, not fit to outcome data yet (see Scoring model
  above).
- No authentication on the review API/UI — it's designed to run locally against
  `data/`, which itself must never leave local/authorized infrastructure.

## Planned next (P1, per design doc §22-23)

AdaFace ensemble, DINOv2/SigLIP body-appearance matching for low face-quality cases,
OCR for case-ID/facility-tag association, and calibrated (logistic regression) scoring
once enough reviewed pairs exist.
