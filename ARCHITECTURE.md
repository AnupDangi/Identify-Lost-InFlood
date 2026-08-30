# Architecture

## Goal and constraint

Retrieve a **Top-K shortlist** of candidate missing-person (AM) matches for each unidentified-body (PM) record, and vice versa, for a human investigator to review. The system never emits an "identity" or a "confidence" claim — see [`docs/project_requirement.md`](docs/project_requirement.md) §15 and [`FOR_INVESTIGATORS.md`](FOR_INVESTIGATORS.md). Final identification is a forensic decision (fingerprint/dental/DNA), not this tool's job.

> **System boundary:** AI stops at candidate retrieval → human investigation. It does NOT cross into identity determination. Every candidate screen displays: *⚠ Investigative Candidate — Not an Identification. Ranking is generated from facial and metadata similarity. Do not communicate identity to families based on this result. Identification requires authorized forensic confirmation.*

## Data flow

```
PROTOTYPE

Public Nepal Police UDB (missing/dead-bodies-lists)
        |
        v
research/data_collection/scrape_udb.py  --\
  (research ingestion adapter; HTTPS       |
   bypass for incomplete cert chain,       |
   resumable; shim at scripts/scrape_     |
   _udb.py for backwards compatibility)    |
   (or an ad-hoc list scrape, normalized  |
    via scripts/import_pm_from_list_      |
    _scrape.py)                           |
        |                                 |
        v                                 v
data/manifests/am_persons.csv          data/manifests/pm_persons.csv
data/raw/images/am/{id}.jpg            data/raw/images/pm/...
        |                                 |
        v                                 v
scripts/build_embeddings.py  (InsightFace buffalo_l: detect, quality-gate, embed, gender)
        |                                 |
        v                                 v
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
scripts/search_candidates.py  (via dvi/retrieval.py)
  for each usable PM embedding:
    FAISS top-100 AM neighbors (cosine similarity)
      -> blend with metadata score (dvi/scoring.py)
      -> data/manifests/candidates.csv (query_type, query_record_id, candidate_type,
         rank, face_score, metadata_score, final_score, sex_score, age_score, ...)
      provenance per candidate (reason for ranking + missing evidence)
        |
        v
main.py  (FastAPI)  --serves-->  web/index.html  (review UI)
        |
        v
data/manifests/reviews.csv  (Candidate for further examination / Not a likely candidate / Insufficient evidence + forensic_status)
```

```
AUTHORIZED PILOT (proposed, not implemented)

Nepal Police AM/PM records
        ↓
Approved API / export / internal database
        ↓
Same retrieval loop (embed → index → search → re-rank → review)
```

> The scraper exists only to reproduce the independent research prototype. It is not the proposed institutional ingestion mechanism — see README.md Data handling and FOR_INVESTIGATORS.md §8.

If `search_candidates.py` hasn't been run yet (no face pipeline available), `scripts/compare_am_pm.py` produces `candidate_matches.csv` (metadata-only, full cross-product) as a fallback — the API prefers the face+metadata file when both exist.

## Why these pieces, and not others

- **InsightFace `buffalo_l`** bundles SCRFD detection, landmarks, ArcFace recognition, and a gender/age model in one forward pass — P0 needs exactly this, not five separate libraries (per the design doc's "don't integrate everything on day one" guidance).
- **FAISS `IndexFlatIP`**, not IVF/HNSW: exact search is fine up to ~100K vectors, and AM is ~9K. Premature approximate-search tuning would trade correctness for speed nobody needs yet.
- **One-directional FAISS index (AM only)**: the natural disaster workflow is "found a body, who could this be" (PM→AM). The API's `Missing` tab (AM→PM) reuses the same `candidates.csv`, searched bidirectionally via `dvi/retrieval.py`, rather than a single pivoted file — cosine similarity is symmetric, but a Top-K neighborhood is not. The API computes candidates on demand via `dvi/retrieval` so it doesn't only serve whatever the last batch run produced.

## Quality gate (why PM has fewer usable faces than AM)

`build_embeddings.py` marks a face unusable (and excludes it from the FAISS index/search entirely, rather than forcing a low-quality match) when:

- no face detected at all,
- the detected face is smaller than 40px in either dimension,
- detector confidence is below 0.5.

Post-mortem/disaster photographs are inherently harder to enroll — occlusion, injury, poor lighting, camera angle. `has_face: false` is a valid, expected outcome per the design doc's "inconclusive is a valid result" principle, not a pipeline failure. The review UI hides the **Find Candidate Matches** button entirely for those records rather than showing a meaningless result. The API falls back to metadata-only ranking rather than returning 422 for no-face queries (see `dvi/retrieval.py` Phases 7-9).

## Scoring model

```
face_score      = cosine_similarity(am_embedding, pm_embedding) rescaled from [-1,1] to [0,1]
metadata_score  = 0.30 * sex_score + 0.25 * age_score + 0.10 * height_score
                + 0.15 * date_score + 0.20 * location_score
final_score     = face_weight * face_score + (1 - face_weight) * metadata_score   (face_weight = 0.6 default)
```

Displayed as: `Face similarity score: 0.78 / Metadata compatibility: 0.63 / Composite ranking score: 0.72 — UNCALIBRATED — NOT IDENTITY PROBABILITY`.

All sub-scores are **soft compatibility bands**, not hard filters (design doc §14): a 20-year age gap ranks lower than a 2-year gap, it doesn't disqualify the pair, because post-mortem age estimation is frequently wrong. The weights above are hand-picked starting points for experimentation, explicitly **not** claimed to be statistically calibrated — see `dvi/scoring.py` header. The design doc's recommended next step is logistic regression against labeled review outcomes once `reviews.csv` has enough data to fit one.

`sex_score` blends two independent signals when both are available: the scraped metadata's stated sex, and each side's InsightFace-detected sex from the photo itself. Each contributes equally. **Recorded sex → metadata compatibility; model-predicted gender → diagnostic flag only.** InsightFace gender on PM imagery has not been validated and can be wrong due to poor-quality PM imagery. A pair whose metadata agrees but whose photos read as different sexes scores lower than a metadata-only match would, but 10.1% disagreement must not be interpreted automatically as a data-entry error — it may arise from metadata errors, model errors, image quality, or incorrect candidate pairing. See `dvi/scoring.py` and `OVERVIEW.md`.

Each candidate also carries provenance:

```
PM-00427
└── Candidate AM-00719
      ├─ Face similarity             0.74
      ├─ Recorded sex                compatible
      ├─ Age                         42 vs estimated 40–45
      ├─ Height                      unavailable
      ├─ Missing date                2025-08-03 AD
      ├─ Recovery date               2025-08-10 (BS 2082-04-26 → AD verified)
      ├─ Last known location         Rasuwa
      ├─ Recovery location           Rasuwa
      └─ Clothing description        partial compatibility

Reason for ranking: High facial similarity + temporal/location compatibility
Missing evidence: fingerprint, dental, DNA, distinguishing marks
```

## API surface

| Endpoint | Purpose |
|---|---|
| `GET /api/pm`, `GET /api/am` | List records with `has_face`/`detected_sex` merged in |
| `GET /api/pm/{id}/candidates` | Top-K AM candidates for a PM record (with provenance, component scores, investigative disclaimer) |
| `GET /api/am/{id}/candidates` | Top-K PM candidates for an AM record (with provenance) |
| `POST /api/search/upload` | Transient photo + optional name/gender search (falls back to metadata_only) |
| `POST /api/review` | Log `Candidate for further examination / Not a likely candidate / Insufficient evidence` + optional `forensic_status` (Not examined / Fingerprint/Dental/DNA requested / Excluded / Confirmed externally) |
| `GET /images/...` | Serves `data/raw/images/` directly (local-only tool) |

Review decisions are stored with `review_id, pm_record_id, am_record_id, decision (canonical: potential|rejected|inconclusive), forensic_status, notes, reviewed_at, ranking_model_version, face_model_version`.

## Known limitations (P0)

- PM records scraped via the list-page fallback (`import_pm_from_list_scrape.py`) lack height, clothing, and distinguishing-marks fields — those only exist on each record's detail page, which the research adapter would need to fetch separately.
- Some PM `event_date` values were stored in Bikram Sambat (visible as years such as 2082) — now normalized via `dvi/dates.py` with explicit columns `raw_event_date, calendar_type, event_date_normalized, conversion_status: verified`. `date_score` uses the normalized Gregorian date; unparseable dates score neutral (0.5) rather than crashing or being silently guessed.
- `final_score` / `composite ranking score` weights are heuristic, not fit to outcome data yet (see Scoring model above) — **UNCALIBRATED, not a probability**.
- No authentication on the review API/UI — it's designed to run locally against `data/`, which itself must never leave local/authorized infrastructure.
- No ground-truth evaluation yet; Recall@K is TBD (see `reports/evaluation/latest.md`).

## What is NOT being built next (frozen)

```
❌ AdaFace ensemble
❌ DINOv2 body matching
❌ multimodal LLM
❌ DNA algorithm
❌ better FAISS infrastructure
❌ cloud deployment
❌ dashboards
❌ notification system
❌ hospital integration
```

Instead (pre-institutional checklist): privacy cleanup, terminology cleanup, synthetic demo, proper disclaimers, score calibration wording, provenance/explanation, evaluation protocol, investigator brief, contact Nepal Police.

## Evaluation

Ground-truth identities: TBD. Number of confirmed pairs: TBD. Recall@1/5/10/20 and false candidate rate: TBD. Subgroups (good-quality face, blurred, occluded, injured, extreme pose, decomposition): TBD. No accuracy claim until evaluated against independently confirmed identities — see `OVERVIEW.md` Evaluation and `FOR_INVESTIGATORS.md` §7.

Research question: *Can an AI-assisted retrieval system reduce the number of AM records per PM case without excluding the correct candidate from the shortlist?* Metric: **Recall@20** (9,290 → Top-20).
