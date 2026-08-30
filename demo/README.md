# Demo — Synthetic Data Only

This directory contains **entirely synthetic** demonstration records. No real victim data, photographs, embeddings, or biometric templates are included.

All names, IDs, photos (if any), and metadata below are fictional and generated for illustration only.

```
demo/
├── README.md
├── am/
│   ├── AM-DEMO-001.json
│   ├── AM-DEMO-002.json
│   └── ...
├── pm/
│   ├── PM-DEMO-001.json
│   └── ...
└── screenshots/   # synthetic placeholders only (no real AM/PM photos)
```

## How to use

Load these records into the local UI to see candidate ranking without scraping real data:

```bash
# Copy synthetic manifests into data/ (optional helper script)
python demo/load_demo.py --target data/
```

Or manually inspect:

```bash
cat demo/am/AM-DEMO-001.json
cat demo/pm/PM-DEMO-001.json
```

## Screenshot policy

Only screenshots generated against these synthetic records may be committed. Screenshots containing real AM/PM photos must never be committed — paths `demo/images/`, `demo/private/`, `demo/real/`, `demo/raw/` are gitignored. See `docs/PRIVACY_CLEANUP.md`.

## Example synthetic candidate

```
Unidentified Record PM-DEMO-04
  recovery date: 2025-08-10 (AD)
  location: Rasuwa (synthetic)
  photo: placeholder

           Candidate search → Top-3

#1 AM-DEMO-017
  Face similarity score:       0.81
  Metadata compatibility:      0.68
  Composite ranking score:     0.76  UNCALIBRATED — NOT IDENTITY PROBABILITY
  Provenance: High facial similarity + temporal/location compatibility
  Missing evidence: fingerprint, dental, DNA

→ Candidate for further examination
```

> ⚠ Investigative Candidate — Not an Identification. Do not communicate identity to families based on this result.
