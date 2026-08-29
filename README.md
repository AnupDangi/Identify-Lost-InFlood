# Find-Lost-InFlood

AI-assisted AM↔PM reconciliation prototype for disaster victim identification (DVI):
matches ante-mortem (missing person) records against post-mortem (unidentified body)
records scraped from the Nepal Police Unidentified/Missing Person Database (UDB), using
face embeddings + metadata compatibility to produce a **shortlist for human review** —
never an automated identification. See [`docs/project_requirement.md`](docs/project_requirement.md)
for the full design rationale, [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the pieces
fit together, and [`OVERVIEW.md`](OVERVIEW.md) for a plain-language walkthrough of what
it does, current results, and where this could go as an institutional/government tool.

## Data sources

- **Missing persons**: https://udb.nepalpolice.gov.np/missing
- **Unidentified bodies**: https://udb.nepalpolice.gov.np/dead-bodies-lists

Both are public listings on the Nepal Police Unidentified/Missing Person Database (UDB).
`scripts/scrape_udb.py` scrapes them directly (paginated, resumable); records/photos are
normalized into `data/manifests/{am,pm}_persons.csv` per the schema in
`docs/project_requirement.md`. See **Data handling** below before scraping at any scale
beyond a small research sample.

> **This tool never returns an "identity" or a "confidence %" claim.** Final identification
> must go through fingerprint, dental, or DNA evidence — see section 15 of the project
> requirements.

## Data handling

This project scrapes and processes real photographs and personal data of missing people
and deceased individuals. `data/` (all scraped images, manifests, embeddings, and the
FAISS index) is **gitignored and must never be committed, published, or moved to a public
bucket**. Only run the scraper under an authorized basis (institutional/police
authorization) — see the privacy discussion in `docs/project_requirement.md`.

## Setup

```bash
uv sync
```

Requires Python 3.13 (pinned via `.python-version`). First run of the face pipeline
downloads the InsightFace `buffalo_l` model bundle (~280MB) to `~/.insightface/models/`.

## Pipeline

Run these in order from the repo root. Each step is resumable — re-running skips work
already done unless you pass `--force`.

```bash
# 1. Scrape Nepal Police UDB listings (missing persons and/or dead bodies)
uv run python3 scripts/scrape_udb.py --record-type both --concurrency 6

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

## Review UI

`http://localhost:8001/` — two tabs, **Dead Bodies** and **Missing**. Each record shows a
**Prediction** button only if a usable face was detected in its photo (`has_face`); records
where face detection failed entirely show "No face detected" instead, since face-based
prediction is meaningless without one. Clicking Prediction shows the top candidates from
the *other* dataset (dead bodies ↔ missing persons are symmetric), each with a
face/metadata/final score breakdown, a sex-mismatch warning when the photo's detected sex
disagrees with the record's stated sex, and Potential / Reject / Inconclusive buttons that
log to `data/manifests/reviews.csv`. Unidentified-body photos are blurred by default
("Reveal image") since they can be graphic.

### Demo

![Dead Bodies tab, candidate view](demo/images/image1.png)
![Dead Bodies tab, another record's candidates](demo/images/image2.png)

> `demo/images/` is gitignored, same reasoning as `data/` — these screenshots show real,
> identifiable people (a revealed graphic body photo, named missing persons), and this
> repo has a public GitHub remote. The images above render locally (they exist on disk)
> but **will show as broken links on GitHub** — that's intentional, not a bug. If you want
> a screenshot that's safe to publish, swap in a cropped/blurred or synthetic one at the
> same path.

## Repository layout

```
scripts/
  scrape_udb.py                  Parallel scraper for both UDB listings (HTTPS-bypass
                                  for the site's incomplete cert chain, resumable)
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
data/                            Gitignored: scraped images, manifests, embeddings, index
```

## Status

- AM (missing persons): 9,290 records scraped nationwide, 6,108 usable face embeddings
  (65.7%).
- PM (unidentified bodies): 1,960 records scraped, 376 tagged to the Rasuwa disaster
  specifically, 857 usable face embeddings (43.7%).
- 17,140 candidate rows produced (top-20 AM matches per usable PM record); 10.1% flagged
  `sex_conflict` (metadata sex agrees but the photo's detected sex doesn't, or vice versa).
- Face detection succeeds on a minority of PM photos (post-mortem/disaster images are
  harder to enroll) — this is expected and reported per-record via `has_face`/
  `failure_reason` rather than forced through matching, per the "inconclusive is a valid
  result" principle in the design doc.

Full breakdown, what these numbers mean, and where this could go next: see
[`OVERVIEW.md`](OVERVIEW.md).
