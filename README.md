# AI-Assisted Disaster Victim Reconciliation — Nepal

A research prototype for reducing the manual search space during Disaster Victim Identification (DVI).

The system compares:

- **AM (Ante-Mortem):** missing-person records
- **PM (Post-Mortem):** unidentified-person records

and produces a **ranked shortlist of possible AM↔PM candidates for human investigators**.

> **This system DOES NOT identify a person.** Facial similarity and metadata are investigative leads only. Final identification must follow authorized forensic procedures such as fingerprints, dental examination and/or DNA analysis.

Built as an independent research prototype using publicly accessible Nepal Police UDB records.

- Current prototype: 9,290 AM records
- 1,960 PM records
- 6,108 usable AM facial representations
- 857 usable PM facial representations

No real victim data, photographs, embeddings, or biometric templates are distributed through this repository.

---

## Research question

> Can an AI-assisted retrieval system reduce the number of AM records that an investigator must manually examine for each unidentified PM case **without excluding the correct candidate from the shortlist?**

The primary metric is **Recall@K** (does the true match survive into the Top-20?). The system aims to narrow `9,290 → Top-20` candidates per PM record. No accuracy claim is made until evaluated against independently confirmed identities — see [Evaluation](#evaluation).

```
DISASTER

Missing people                 Unidentified victims
     │                                │
     │ AM                             │ PM
     └────────────┐       ┌───────────┘
                  ↓       ↓
                RETRIEVAL
                    │
                    ↓
            Thousands → Top-K
                    │
                    ↓
             HUMAN INVESTIGATOR
                    │
            ┌───────┼────────┐
            ↓       ↓        ↓
      fingerprint dental    DNA
            └───────┼────────┘
                    ↓
          FORENSIC IDENTIFICATION

AI stops at: candidate retrieval → human investigation
It does NOT cross into: identity determination.
```

---

## Problem → How it helps → Limitations

**Problem:** After a disaster, two lists grow independently — missing persons reported by families and unidentified bodies recovered by authorities. Manual matching across thousands of records is intractable.

**How it helps:** For each PM record, the system ranks AM records by face similarity (InsightFace/ArcFace) + metadata compatibility (sex, age, height, date, location) and presents Top-20 candidates with full provenance for human review.

**Limitations:**

- Face detection fails on ~34% of AM and ~56% of PM photos (expected for post-mortem imagery); those records fall back to metadata-only ranking.
- Composite ranking scores are **heuristic and uncalibrated** — they are ordering signals, not probabilities.
- No ground-truth evaluation has been completed; `reports/evaluation/latest.md` reports TBD until confirmed identities are available.
- Some PM dates remain in Bikram Sambat and location/height/clothing fields may be missing depending on import path — see [Known limitations](ARCHITECTURE.md#known-limitations-p0).
- No authentication; runs locally against `data/` only.

See [`FOR_INVESTIGATORS.md`](FOR_INVESTIGATORS.md) for a one-page brief aimed at forensic officers.

---

## Data sources

- **Missing persons:** https://udb.nepalpolice.gov.np/missing
- **Unidentified bodies:** https://udb.nepalpolice.gov.np/dead-bodies-lists

Both are public listings on the Nepal Police UDB. The prototype was built via a research ingestion adapter over these public listings (see **Data handling** below). An authorized pilot would use an approved API/export/internal database instead — see [Authorized pilot vs. prototype](ARCHITECTURE.md).

> **This tool never returns an "identity" or a "confidence %" claim.** Final identification must go through fingerprint, dental, or DNA evidence — see section 15 of the project requirements.

---

## Data handling

This project scrapes and processes real photographs and personal data of missing people and deceased individuals. `data/` (all scraped images, manifests, embeddings, and the FAISS index) is **gitignored and must never be committed, published, or moved to a public bucket**. Only run the scraper under an authorized basis (institutional/police authorization) — see the privacy discussion in `docs/project_requirement.md`.

`research/data_collection/scrape_udb.py` (symlinked from `scripts/scrape_udb.py` for backwards compatibility) exists **only to reproduce the independent research prototype**. It is not the proposed institutional ingestion mechanism — see [`docs/PRIVACY_CLEANUP.md`](docs/PRIVACY_CLEANUP.md).

---

## Demo (synthetic)

The repository ships synthetic demonstration data under `demo/` — no real AM/PM photos are included or referenced.

```
demo/
├── README.md
├── am/
│   ├── AM-DEMO-001.json
│   └── ...
├── pm/
│   ├── PM-DEMO-001.json
│   └── ...
└── screenshots/   (synthetic placeholders only)
```

Example synthetic candidate:

```
Unidentified Record PM-DEMO-04

           Candidate search

#1 AM-DEMO-17
Face similarity:     0.81
Location:            Strong
Date:                Compatible
Age:                 Compatible

→ Candidate for further examination
   UNCALIBRATED — NOT IDENTITY PROBABILITY
```

To see the UI yourself, run the pipeline locally and open `http://localhost:8001/`. Any screenshot you generate must use synthetic/placeholder records only — paths under `demo/images/`, `demo/private/`, `demo/real/`, `demo/raw/` are gitignored.

---

## Setup

```bash
uv sync
```

Requires Python 3.13 (pinned via `.python-version`). First run of the face pipeline downloads the InsightFace `buffalo_l` model bundle (~280MB) to `~/.insightface/models/`.

---

## Pipeline

Run these in order from the repo root. Each step is resumable — re-running skips work already done unless you pass `--force`.

```bash
# 1. Research ingestion adapter over public UDB listings (paginated, resumable)
#    In an authorized pilot this would be replaced by an approved API/export
uv run python3 research/data_collection/scrape_udb.py --record-type both --concurrency 6
# (legacy path still works: scripts/scrape_udb.py)

# 1b. If you have an ad-hoc list-page scrape instead of scrape_udb.py's output
#     (e.g. dead_bodies_dataset.csv), normalize it into the same schema:
uv run python3 scripts/import_pm_from_list_scrape.py

# 1c. If AM's `location` column comes back blank (a label-parsing bug fixed
#     in scrape_udb.py, backfillable from the raw JSON without re-scraping):
uv run python3 scripts/backfill_am_location.py

# 2. Face detection + quality gate + ArcFace embeddings (+ gender cross-check)
uv run python3 scripts/build_embeddings.py --record-type am
uv run python3 scripts/build_embeddings.py --record-type pm

# 3. Build the FAISS index over AM (missing-person) embeddings
uv run python3 scripts/build_index.py --record-type am

# 4. PM -> AM candidate search: face similarity + metadata re-ranking
uv run python3 scripts/search_candidates.py --top-k 20

# (Optional) metadata-only comparison, useful before the face pipeline has run
uv run python3 scripts/compare_am_pm.py --top-k 20

# 5. Review API + UI
uv run main.py
# open http://localhost:8001/
```

---

## Review UI

`http://localhost:8001/` — two tabs, **Dead Bodies** and **Missing**. Each record shows a **Find Candidate Matches** button only if a usable face was detected in its photo (`has_face`); records where face detection failed entirely show "No face detected" instead, since face-based search is meaningless without one. Clicking the button shows the top candidates from the *other* dataset (dead bodies ↔ missing persons are symmetric), each with a face/metadata/composite score breakdown, an investigative-candidate disclaimer, and `Candidate for further examination / Not a likely candidate / Insufficient evidence` buttons that log to `data/manifests/reviews.csv`. Unidentified-body photos are blurred by default ("Reveal image") since they can be graphic.

Every candidate screen displays:

> ### ⚠ Investigative Candidate — Not an Identification
>
> Ranking is generated from facial and metadata similarity. Do not communicate identity to families based on this result. Identification requires authorized forensic confirmation.

Each candidate also includes **provenance** (why it was ranked — face similarity, temporal/location compatibility, missing evidence) and a **Forensic status** workflow (fingerprint/dental/DNA) recorded externally.

---

## Evaluation

Ground-truth identities: **TBD**
Number of confirmed pairs: **TBD**

| Metric | Result |
|---|---:|
| Recall@1 | TBD |
| Recall@5 | TBD |
| Recall@10 | TBD |
| Recall@20 | TBD |
| False candidate rate | TBD |

Subgroups (good-quality face, blurred, occluded, injured, extreme pose, decomposition): **TBD**

> **No accuracy claim is made until evaluated against independently confirmed identities.**

See `reports/evaluation/latest.md` for the latest run (currently synthetic/placeholder only).

---

## Repository layout

```
research/
  data_collection/
    scrape_udb.py                Research ingestion adapter (HTTPS-bypass
                                 for the site's incomplete cert chain, resumable)
scripts/
  import_pm_from_list_scrape.py  Normalizes an ad-hoc list-page scrape into the
                                 AM/PM manifest schema
  backfill_am_location.py        Re-parses already-saved raw JSON to fix AM's
                                 `location` column (label-parsing bug, no re-scrape)
  build_embeddings.py            InsightFace (buffalo_l) face detection, quality
                                 gate, ArcFace embeddings, gender cross-check
  build_index.py                 FAISS IndexFlatIP over usable embeddings
  search_candidates.py           Face + metadata candidate shortlist (PM -> AM)
  compare_am_pm.py               Metadata-only fallback shortlist (no face pipeline
                                 required) + the sex/age/height/date/location
                                 scoring functions reused by search_candidates.py
main.py                          FastAPI review API + static file server (entry point)
web/index.html                   Review UI (plain HTML/JS, no build step)
docs/project_requirement.md      Original design document
ARCHITECTURE.md                  System architecture, data flow, scoring model
FOR_INVESTIGATORS.md             One-page brief for forensic officers
demo/                            Synthetic demonstration dataset (no real photos)
data/                            Gitignored: scraped images, manifests, embeddings, index
```

---

## Status

- AM (missing persons): 9,290 records scraped nationwide, 6,108 usable face embeddings (65.7%).
- PM (unidentified bodies): 1,960 records scraped, 376 tagged to the Rasuwa disaster specifically, 857 usable face embeddings (43.7%).
- 17,140 candidate rows produced (top-20 AM matches per usable PM record); composite ranking scores are uncalibrated — 63/857 queries produced a rank-1 candidate with ranking score above 0.70, but this threshold has no probabilistic interpretation and does not measure identification accuracy. Ground-truth evaluation is still required.
- Face detection succeeds on a minority of PM photos (post-mortem/disaster images are harder to enroll) — this is expected and reported per-record via `has_face`/`failure_reason` rather than forced through matching, per the "inconclusive is a valid result" principle in the design doc.
- No review decisions logged yet (`data/manifests/reviews.csv` doesn't exist) — the UI is built and tested end-to-end (see `tests/`), but no real investigative review session has been logged.

Full breakdown, what these numbers mean, and where this could go next: see [`OVERVIEW.md`](OVERVIEW.md).
