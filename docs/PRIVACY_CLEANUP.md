# Privacy cleanup: committed demo screenshots

## What was found

`demo/images/image1.png` and `demo/images/image2.png` were committed to git in
commit `2a86f1e` ("feat:add demo images as well"). Per the README at the time,
these were screenshots of the review UI's **Dead Bodies** tab -- which, per
`ARCHITECTURE.md` and `docs/project_requirement.md`, renders real photographs
scraped from the Nepal Police UDB `dead-bodies-lists` endpoint (PM / unidentified
body records), along with the record's location, dates, and other metadata fields.
That makes the screenshots real, identifiable personal data of deceased individuals,
not synthetic/redacted examples.

These files were **not opened or inspected as part of this cleanup** -- the
classification above is based on the caption text ("Dead Bodies tab, candidate view")
and the surrounding documentation, which is sufficient to treat them as sensitive
without needing to view the contents.

## What was changed

1. `git rm --cached demo/images/image1.png demo/images/image2.png` -- untracked the
   files going forward.
2. `.gitignore` now ignores `demo/images/`, `demo/private/`, `demo/raw/`, and
   `demo/real/`, so a future screenshot dropped into any of those paths will not be
   committed by accident.
3. `README.md`'s Demo section no longer references the removed images and now points to synthetic data under `demo/` only.
4. **History purge (completed 2026-08-30):** `git filter-repo --path demo/images/image1.png --path demo/images/image2.png --invert-paths --force` was run locally, rewriting history so the blobs no longer exist in any commit.

**Verification:**

```bash
git log --all --oneline -- demo/images/image1.png demo/images/image2.png
# -> (no output, as intended)
git rev-list --objects --all | grep demo/images
# -> (no output, as intended)
```

The purge removed `origin` (filter-repo safety behavior); it was re-added as `https://github.com/AnupDangi/Find-Lost-InFlood.git`. To publish the purge, force-push the rewritten history:

```bash
git push origin --force --all
git push origin --force --tags
```

Coordinate with anyone else who has the repo cloned — they will need to re-clone or hard-reset. If the repo was forked/mirrored outside your control, treat the images as permanently public regardless of this cleanup.

## Research ingestion vs authorized pilot

`research/data_collection/scrape_udb.py` (shim at `scripts/scrape_udb.py`) exists only to reproduce the research prototype. It is not the proposed institutional ingestion mechanism (which would be an approved API/export/internal database). See `ARCHITECTURE.md` and `README.md`.

## Going forward

- Never commit a screenshot, log excerpt, or test fixture that renders real AM/PM
  photos, names, case IDs, phone numbers, or other fields sourced from `data/`.
- `data/` itself was already gitignored before this session and remains the only
  place real scraped records should live.
- If a demo screenshot is needed for documentation, generate it against synthetic
  placeholder records (fake name, placeholder image, made-up IDs) so a screenshot of
  the actual UI carries no real personal data.
