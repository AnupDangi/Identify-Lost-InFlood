# Overview

A plain-language summary of what this system does, what it currently produces on real data, and how it could grow into an institutional tool. For technical design see [`ARCHITECTURE.md`](ARCHITECTURE.md); for the original design rationale see [`docs/project_requirement.md`](docs/project_requirement.md).

> **This system DOES NOT identify a person.** Facial similarity and metadata are investigative leads only. Final identification must follow authorized forensic procedures such as fingerprints, dental examination and/or DNA analysis.

## Research question

> Can an AI-assisted retrieval system reduce the number of AM records that an investigator must manually examine for each unidentified PM case **without excluding the correct candidate from the shortlist?**

The primary metric is **Recall@20** (did the correct person survive into the Top-20?). If confirmed person X is somewhere among 9,290 missing people, did the system narrow `9,290 → 20` without losing the true match? No accuracy claim is made until evaluated against independently confirmed identities.

## What problem this solves

After a disaster (this prototype is built around the Rasuwa flood), two lists exist and grow independently:

- **Missing persons** reported by families — a name, a photo, where they were last seen.
- **Unidentified bodies** recovered by authorities — often no name, a photo, where and when they were found.

Matching these two lists today is manual: an investigator or a grieving family member scrolls through hundreds of photos by eye. At the scale this system scraped — **9,290 missing-person records, 1,960 unidentified-body records** — that's not tractable by hand. This tool narrows "compare everyone to everyone" down to "here are the 20 missing persons most worth a human's attention for this one unidentified body," combining a face-recognition model with the metadata (age, sex, location, dates) already in the records.

**It never decides identity.** It produces a ranked shortlist for a person to review, the same way the design doc's INTERPOL reference describes DVI facial-recognition systems working: candidate lists for manual review, with fingerprint/dental/DNA as the actual identifiers.

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

## How it works, in order

1. **Research ingestion** — `research/data_collection/scrape_udb.py` ingests both public Nepal Police UDB listings into a common schema — name, sex, age, height, date, location, clothing, distinguishing marks, photo. In an authorized pilot this adapter would be replaced by an approved API/export/internal database (see [Where this could go](#where-this-could-go-an-institutional--government-tool)).
2. **Detect and embed faces** (`scripts/build_embeddings.py`) — InsightFace locates the face in each photo, rejects ones too small/blurry/low-confidence to trust (see *Data quality* below), and produces a 512-dimension ArcFace vector for the rest. It also reads the same model's gender estimate as a diagnostic flag only (see *Sex signal* below).
3. **Index** (`scripts/build_index.py`) — the missing-persons' vectors go into a FAISS similarity index.
4. **Search and rank** (`scripts/search_candidates.py`) — for each unidentified body, find its 100 nearest missing-person faces by cosine similarity, then re-rank the top 20 by blending that face similarity with how well the metadata lines up (sex, age, height, how many days between "went missing" and "found," and whether the recorded locations overlap). All sub-scores use heuristic, uncalibrated weights — see `dvi/scoring.py`.
5. **Human review** (`main.py` + `web/index.html`) — an investigator browses either list, clicks **Find Candidate Matches**, and marks each candidate as `Candidate for further examination / Not a likely candidate / Insufficient evidence`. Every decision is logged with a timestamp for audit. A separate **Forensic status** (fingerprint/dental/DNA requested, excluded, or identification confirmed externally) records the external forensic determination — the software itself never exposes a "Confirm Identity" button.

## What it actually produced, on this data

| | Missing persons (AM) | Unidentified bodies (PM) |
|---|---:|---:|
| Records scraped | 9,290 | 1,960 |
| Usable face detected | 6,108 (65.7%) | 857 (43.7%) |
| Tagged to the Rasuwa disaster specifically | — | 376 |

- **17,140 candidate pairs** generated (top-20 missing-person matches for each of the 857 unidentified bodies with a usable face).
- **10.1% of candidate pairs produced disagreement between recorded sex metadata and model-estimated sex.** This may arise from metadata errors, model errors, image quality, or incorrect candidate pairing and must not be interpreted automatically as a data-entry error. Recorded sex is used for metadata compatibility; model-predicted gender is a diagnostic flag only until validated against this type of imagery.
- **63/857 queries produced a rank-1 candidate with an uncalibrated composite ranking score above 0.70.** This threshold has no probabilistic interpretation and does not measure identification accuracy. Ground-truth evaluation is still required. Scores are presented as `Face similarity score: 0.78 / Metadata compatibility: 0.63 / Composite ranking score: 0.72 — UNCALIBRATED — NOT IDENTITY PROBABILITY`.
- Missing-person reports cluster geographically: कोशी (3,372), बागमती (1,620), गण्डकी (1,220), लुम्बिनी (1,139) provinces account for most of the AM dataset — useful for understanding where this dataset is actually representative versus sparse.
- No review decisions logged yet (`data/manifests/reviews.csv` doesn't exist) — the UI is built and tested end-to-end, but nobody has run a real review session through it yet.

### Why so many photos fail face detection

Post-mortem and disaster photography is inherently harder to enroll than a studio ID photo: injury, occlusion, poor lighting, extreme camera angles, decomposition. A record with `has_face: false` isn't a bug — the design doc calls this out explicitly ("inconclusive is a valid result," §9) — it means face-based matching genuinely can't help for that one record, and the UI reflects that honestly (no Find Candidate Matches button) rather than forcing a low-quality match through the pipeline. This is also why PM's usable-face rate (43.7%) is roughly two-thirds of AM's (65.7%): AM photos are typically the last normal photo a family had of someone; PM photos are of a body.

### A caught-and-fixed data bug worth knowing about

AM's `location` field came back completely blank on the first pass — not missing data, a label-parsing bug (a stray space before a colon in the source HTML broke every lookup). The underlying data was there in the raw scrape the whole time; `scripts/backfill_am_location.py` recovered it for 8,705 of 9,290 records without re-scraping, and the ingestion adapter is fixed for future runs. Mentioned here because "the system says it works" is a different claim from "the system's output has been checked for silent gaps" — this project tries to do the latter, and that habit is worth keeping as it grows.

### Date normalization

Some PM `event_date` values were stored in Bikram Sambat (visible as years such as 2082) and must not be silently compared as Gregorian. The pipeline now persists:

```
source_date: 2082-04-17 BS
normalized_date: 2025-08-02 AD
calendar: BS
conversion_status: verified
```

See `dvi/dates.py`. Never silently guess; unparseable dates score neutral (0.5).

## Evaluation

> **No accuracy claim is made until evaluated against independently confirmed identities.**

| Metric | Result |
|---|---|
| Ground-truth identities | TBD |
| Number of confirmed pairs | TBD |
| Recall@1 | TBD |
| Recall@5 | TBD |
| Recall@10 | TBD |
| Recall@20 | TBD |
| False candidate rate | TBD |

Subgroups (all TBD until ground truth is available):

| Condition | Queries | R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| good-quality face | TBD | TBD | TBD | TBD | TBD |
| blurred | TBD | TBD | TBD | TBD | TBD |
| occluded | TBD | TBD | TBD | TBD | TBD |
| injured | TBD | TBD | TBD | TBD | TBD |
| extreme pose | TBD | TBD | TBD | TBD | TBD |
| decomposition | TBD | TBD | TBD | TBD | TBD |

**9,290 records ≠ model validation. 17,140 candidate pairs ≠ successful identification.** Until Recall@K is measured on held-out, identity-separated confirmed pairs, the system's ability to reduce workload *without excluding the true match* is unproven.

## Where this could go: an institutional / government tool

The design doc frames this correctly: a real deployment is **not** "scrape the public website into a personal account" — it's a two-mode path:

```
PROTOTYPE

Public Nepal Police UDB
        ↓
Research ingestion adapter
(research/data_collection/scrape_udb.py)


AUTHORIZED PILOT

Nepal Police AM/PM records
        ↓
Approved API / export / internal database
        ↓
Reconciliation system (same retrieval loop)
```

- **Mode A (this prototype)**: proves the AM↔PM retrieval loop actually surfaces useful candidates before investing in anything heavier. The scraper exists only to reproduce the independent research prototype and is not the proposed institutional ingestion mechanism.
- **Mode B (authorized pilot)**: Nepal Police (or another authorized body) provides a proper export/API, or the pipeline runs inside their own infrastructure — private storage, encryption, access control, audit trails, explicit retention policy. Nepal's Privacy Act treats biometric data as personal information requiring legal authority to collect and process; that's the gate to walk through before this touches anyone beyond the researcher building it.

If that authorization exists, this same pipeline is the plausible shape of a real service:

- **Detect**: the current shortlist mechanism, extended only after validation (AdaFace/body-appearance matching for the ~35-56% of photos where face detection alone comes up empty, OCR for case-ID tags — all frozen until Recall@K is proven).
- **Inform**: once a candidate is marked "Candidate for further examination" and confirmed by an investigator (never automatically), a notification path to the family that reported the missing person — this is a workflow/integration question, not a modeling one, and deliberately isn't built yet because it shouldn't exist before the matching quality and review process are trustworthy.
- **Update**: `reviews.csv` is already the seed of an audit trail; a real deployment would turn confirmed matches into an update to the official missing-person record itself, closing the loop instead of leaving two disconnected lists.

None of that should be built before the review workflow above has real investigator usage and the design doc's evaluation plan (Recall@1/5/10/20 against an identity-separated benchmark, stress-tested against blur/occlusion/pose) has actually run — the fastest way to lose trust in a tool like this is to ship confident-sounding numbers that haven't been checked against ground truth. This prototype's honest current state — a working shortlist mechanism, tested end-to-end, with zero real reviews logged yet — is the correct place to be before that conversation.

## What is NOT being built next

Frozen until evaluation and privacy/investigator review are complete:

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

Instead:

```
✅ privacy cleanup (git history)
✅ terminology cleanup (scores, Find Candidate Matches)
✅ synthetic demo
✅ proper disclaimers
✅ score calibration wording
✅ provenance/explanation
✅ evaluation protocol
✅ investigator brief
✅ contact Nepal Police
```
