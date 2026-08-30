# For Investigators — AI-Assisted Disaster Victim Reconciliation (One-Page Brief)

> This system **DOES NOT identify a person.** It produces a ranked shortlist of candidate AM↔PM pairs for human forensic review. Final identification must follow authorized forensic procedures (fingerprint, dental, DNA).

---

## 1. Problem

After a disaster, two lists grow independently:

- **AM (Ante-Mortem):** missing-person reports (name, photo, last-seen location/date)
- **PM (Post-Mortem):** unidentified-person records (photo, recovery location/date, facility)

Manual reconciliation across thousands of records (here: 9,290 AM, 1,960 PM) is intractable by eye.

## 2. What the prototype does

- Ingests AM/PM records into a common schema
- Detects faces (InsightFace/ArcFace) and filters by quality; embeds usable faces (512-D)
- Indexes AM faces (FAISS) and, for each PM record, retrieves Top-100 by face similarity, then re-ranks Top-20 by blending face similarity with metadata compatibility (sex, age, height, date, location)
- Presents candidates with full provenance:

```
PM-00427
└── Candidate AM-00719
      ├─ Face similarity             0.74
      ├─ Recorded sex                compatible
      ├─ Age                         42 vs estimated 40–45
      ├─ Height                      unavailable
      ├─ Missing date                2025-08-03 (AD)
      ├─ Recovery date               2025-08-10 (BS 2082-04-26 → AD verified)
      ├─ Last known location         Rasuwa (synthetic)
      ├─ Recovery location           Rasuwa (synthetic)
      └─ Clothing description        partial compatibility

Reason for ranking: High facial similarity + temporal/location compatibility
Missing evidence: fingerprint, dental, DNA, distinguishing marks
Composite ranking score: 0.72 — UNCALIBRATED — NOT IDENTITY PROBABILITY
```

Scores are heuristic, uncalibrated ordering signals — not probabilities.

## 3. What it does NOT do

- Does not determine identity
- Does not output "Match confidence: 78%" — scores are shown as `Face similarity score: 0.78 / Metadata compatibility: 0.63 / Composite ranking score: 0.72 — UNCALIBRATED`
- Does not communicate results to families
- Does not replace fingerprint/dental/DNA

Every candidate screen displays:

> ### ⚠ Investigative Candidate — Not an Identification
> Ranking is generated from facial and metadata similarity. Do not communicate identity to families based on this result. Identification requires authorized forensic confirmation.

## 4. How a candidate is generated

PM photo → face quality gate → ArcFace embedding → FAISS over AM gallery → metadata re-ranking (heuristic weights: sex 0.30, age 0.25, height 0.10, date 0.15, location 0.20; face 0.60 vs metadata 0.40) → Top-20. A record with no usable face falls back to metadata-only ranking.

## 5. Current dataset (prototype)

- 9,290 AM records, 6,108 usable faces (65.7%)
- 1,960 PM records, 857 usable faces (43.7%), 376 tagged to Rasuwa disaster
- 17,140 candidate pairs (Top-20 per usable PM)
- No real review decisions logged yet; no ground-truth evaluation.

Dates: some records remain in Bikram Sambat — normalized to Gregorian with `raw_event_date / calendar_type / event_date_normalized / conversion_status`. Never silently guessed.

## 6. Current limitations

- Face detection fails on ~35-56% of photos (post-mortem imagery harder) — inconclusive is a valid result, not forced.
- Composite scores are uncalibrated; 63/857 rank-1 scores above 0.70 has no probabilistic meaning and is not an accuracy claim.
- 10.1% sex-disagreement signal may be metadata error, model error, image quality, or wrong pairing — must not be auto-interpreted as data-entry error. Model-predicted gender is diagnostic flag only.
- No authentication; runs locally against `data/` only. `data/` is gitignored and never distributed.
- Requires ground-truth confirmation to measure Recall@K.

## 7. Evaluation protocol (TBD until confirmed identities)

| Metric | Status |
|---|---|
| Ground-truth identities | TBD |
| Confirmed pairs | TBD |
| Recall@1 / @5 / @10 / @20 | TBD |
| False candidate rate | TBD |
| Subgroups (blur/occlusion/injury/pose/decomposition) | TBD |

> No accuracy claim until evaluated against independently confirmed identities.

Research question: *Can the system reduce the number of AM records per PM case without excluding the true candidate from the shortlist?* Metric: **Recall@20**.

## 8. Proposed pilot vs. prototype

```
PROTOTYPE:        Public Nepal Police UDB → research ingestion adapter
AUTHORIZED PILOT: Nepal Police AM/PM records → approved API/export/internal database → same retrieval loop
```

The scraper (`research/data_collection/scrape_udb.py`, shim at `scripts/scrape_udb.py`) exists only to reproduce the research prototype and is not the proposed institutional ingestion mechanism.

Future model additions (AdaFace, DINOv2/SigLIP body matching, OCR, calibrated scoring) are **frozen** until evaluation and privacy review are complete.

## 9. Review workflow

- **Investigative decision:** `Candidate for further examination / Not a likely candidate / Insufficient evidence`
- **Forensic status (external, separate):** `Not examined / Fingerprint comparison requested / Dental comparison requested / DNA comparison requested / Excluded through forensic evidence / Identification confirmed externally`

The software never exposes a "Confirm Identity" button — it records the external forensic determination.

## 10. Data / privacy requirements (authorized pilot)

- Private storage, SSE-KMS, TLS, IAM roles, no public bucket
- Role-based access, audit trail (who viewed/searched/decided, when, model version), short-lived signed image URLs
- Explicit retention policy; Nepal Privacy Act compliance for biometric data

## 11. Questions for investigators

1. How are AM and PM cases reconciled today?
2. Approximately how many records does an investigator review per case?
3. Which AM/PM attributes are actually available internally (height/clothing/marks/facility)?
4. At what stage are fingerprints, dental evidence and DNA used?
5. What constitutes sufficient evidence to request DNA comparison?
6. Would Top-5/10/20 candidate retrieval reduce workload?
7. What false-negative rate would make the system unacceptable?
8. What provenance detail would be most useful on a candidate screen?
9. Who should have access, and what audit is required?
10. What is the preferred authorized data ingestion mechanism (API/export/internal DB)?

## 12. Contact

Research prototype — contact repository owner for pilot discussion. No real victim data is distributed via this repository; demonstration uses synthetic records under `demo/`.

---

*See `OVERVIEW.md` for plain-language detail, `ARCHITECTURE.md` for technical design, `docs/project_requirement.md` for original rationale, and `reports/evaluation/latest.md` for evaluation status.*
