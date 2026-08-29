# Overview

A plain-language summary of what this system does, what it currently produces on real
data, and how it could grow into an institutional tool. For technical design see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for the original design rationale see
[`docs/project_requirement.md`](docs/project_requirement.md).

## What problem this solves

After a disaster (this prototype is built around the Rasuwa flood), two lists exist and
grow independently:

- **Missing persons** reported by families — a name, a photo, where they were last seen.
- **Unidentified bodies** recovered by authorities — often no name, a photo, where and
  when they were found.

Matching these two lists today is manual: an investigator or a grieving family member
scrolls through hundreds of photos by eye. At the scale this system scraped — **9,290
missing-person records, 1,960 unidentified-body records** — that's not tractable by hand.
This tool narrows "compare everyone to everyone" down to "here are the 20 missing persons
most worth a human's attention for this one unidentified body," combining a face-recognition
model with the metadata (age, sex, location, dates) already in the records.

**It never decides identity.** It produces a ranked shortlist for a person to review, the
same way the design doc's INTERPOL reference describes DVI facial-recognition systems
working: candidate lists for manual review, with fingerprint/dental/DNA as the actual
identifiers.

## How it works, in order

1. **Scrape** both public Nepal Police UDB listings (`scripts/scrape_udb.py`) into a
   common schema — name, sex, age, height, date, location, clothing, distinguishing
   marks, photo.
2. **Detect and embed faces** (`scripts/build_embeddings.py`) — InsightFace locates the
   face in each photo, rejects ones too small/blurry/low-confidence to trust (see
   *Data quality* below), and produces a 512-dimension ArcFace vector for the rest. It
   also reads the same model's gender estimate as an independent cross-check against the
   record's stated sex.
3. **Index** (`scripts/build_index.py`) — the missing-persons' vectors go into a FAISS
   similarity index.
4. **Search and rank** (`scripts/search_candidates.py`) — for each unidentified body,
   find its 100 nearest missing-person faces by cosine similarity, then re-rank the top
   20 by blending that face similarity with how well the metadata lines up (sex, age,
   height, how many days between "went missing" and "found," and whether the recorded
   locations overlap).
5. **Human review** (`main.py` + `web/index.html`) — an investigator browses
   either list, opens a record's candidates, and marks each Potential / Rejected /
   Inconclusive. Every decision is logged with a timestamp for audit.

## What it actually produced, on this data

| | Missing persons (AM) | Unidentified bodies (PM) |
|---|---:|---:|
| Records scraped | 9,290 | 1,960 |
| Usable face detected | 6,108 (65.7%) | 857 (43.7%) |
| Tagged to the Rasuwa disaster specifically | — | 376 |

- **17,140 candidate pairs** generated (top-20 missing-person matches for each of the 857
  unidentified bodies with a usable face).
- **10.1% flagged `sex_conflict`** — the record's stated sex and the photo's detected sex
  disagree. That's a real, useful signal: it catches data-entry mistakes and mislabeled
  photos, not just bad matches.
- Rank-1 (best) candidate scores above 70% for 63 of 857 queries — a meaningful minority
  where the system is confidently narrowing the field, alongside a majority where it's
  offering "these are plausible, look closely" rather than "this is almost certainly them."
- Missing-person reports cluster geographically: कोशी (3,372), बागमती (1,620), गण्डकी
  (1,220), लुम्बिनी (1,139) provinces account for most of the AM dataset — useful for
  understanding where this dataset is actually representative versus sparse.
- No review decisions logged yet (`data/manifests/reviews.csv` doesn't exist) — the UI is
  built and tested end-to-end, but nobody has run a real review session through it yet.

### Why so many photos fail face detection

Post-mortem and disaster photography is inherently harder to enroll than a studio ID
photo: injury, occlusion, poor lighting, extreme camera angles, decomposition. A record
with `has_face: false` isn't a bug — the design doc calls this out explicitly ("inconclusive
is a valid result," §9) — it means face-based matching genuinely can't help for that one
record, and the UI reflects that honestly (no Prediction button) rather than forcing a
low-quality match through the pipeline. This is also why PM's usable-face rate (43.7%) is
roughly two-thirds of AM's (65.7%): AM photos are typically the last normal photo a family
had of someone; PM photos are of a body.

### A caught-and-fixed data bug worth knowing about

AM's `location` field came back completely blank on the first pass — not missing data,
a label-parsing bug (a stray space before a colon in the source HTML broke every lookup).
The underlying data was there in the raw scrape the whole time; `scripts/backfill_am_location.py`
recovered it for 8,705 of 9,290 records without re-scraping, and `scripts/scrape_udb.py`
is fixed for future runs. Mentioned here because "the system says it works" is a different
claim from "the system's output has been checked for silent gaps" — this project tries to
do the latter, and that habit is worth keeping as it grows.

## Where this could go: an institutional / government tool

The design doc frames this correctly: a real deployment is **not** "scrape the public
website into a personal AWS account" — it's a two-mode path:

- **Mode A (this prototype)**: proves the AM↔PM retrieval loop actually surfaces useful
  candidates before investing in anything heavier.
- **Mode B (authorized pilot)**: Nepal Police (or another authorized body) provides a
  proper export/API, or the pipeline runs inside their own infrastructure — private
  storage, encryption, access control, audit trails, explicit retention policy. Nepal's
  Privacy Act treats biometric data as personal information requiring legal authority to
  collect and process; that's the gate to walk through before this touches anyone beyond
  the researcher building it.

If that authorization exists, this same pipeline is the plausible shape of a real service:

- **Detect**: the current shortlist mechanism, extended with AdaFace/body-appearance
  matching (DINOv2/SigLIP) for the ~35-56% of photos where face detection alone comes up
  empty, and OCR for the case-ID tags visible on some PM photos.
- **Inform**: once a Potential match is confirmed by an investigator (never
  automatically), a notification path to the family that reported the missing person —
  this is a workflow/integration question, not a modeling one, and deliberately isn't
  built yet because it shouldn't exist before the matching quality and review process are
  trustworthy.
- **Update**: `reviews.csv` is already the seed of an audit trail; a real deployment
  would turn confirmed matches into an update to the official missing-person record
  itself, closing the loop instead of leaving two disconnected lists.

None of that should be built before the review workflow above has real investigator usage
and the design doc's evaluation plan (Recall@1/5/10/20 against an identity-separated
benchmark, stress-tested against blur/occlusion/pose) has actually run — the fastest way
to lose trust in a tool like this is to ship confident-sounding numbers that haven't been
checked against ground truth. This prototype's honest current state — a working shortlist
mechanism, tested end-to-end, with zero real reviews logged yet — is the correct place to
be before that conversation.
