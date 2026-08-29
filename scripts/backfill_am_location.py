"""One-off backfill for the AM `location` column and, per
docs/P0_5_IMPLEMENTATION.md Phase 11, the structured `province`/`district`/
`municipality`/`ward` columns dvi/location.py's location_score() uses.

`location` was blank for the entire dataset due to a label-parsing bug fixed in
scrape_udb.py's extract_sections() (trailing space before the colon in labels
like "प्रदेश :" wasn't stripped, so find_first("प्रदेश") never matched
"प्रदेश "). The actual location data was captured correctly in
data/raw/json/am/{id}.json at scrape time -- this just re-parses those
already-downloaded files with the fixed key matching, no network requests.

PM has no equivalent: it was imported from a list-page scrape
(scripts/import_pm_from_list_scrape.py) with no per-record detail-page JSON, so
there is nothing to backfill structured fields from -- PM keeps only its
free-text `location` ("found place"). dvi/location.py's location_score()
already accounts for that asymmetry.

Usage:
    uv run python scripts/backfill_am_location.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
RAW_JSON_DIR = ROOT / "data" / "raw" / "json" / "am"


def find_first_stripped(sections: dict, *labels: str) -> str:
    for section in sections.values():
        stripped = {k.strip(): v for k, v in section.items()}
        for label in labels:
            v = stripped.get(label, "")
            if v:
                return v
    return ""


def main():
    manifest_path = MANIFEST_DIR / "am_persons.csv"
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    for col in ("province", "district", "municipality", "ward"):
        if col not in fieldnames:
            fieldnames.append(col)

    updated = 0
    still_blank = 0
    for row in rows:
        record_id = row["record_id"]
        numeric_id = record_id.removeprefix("AM")
        json_path = RAW_JSON_DIR / f"{numeric_id}.json"
        if not json_path.exists():
            continue
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        sections = parsed.get("sections", {})
        province = find_first_stripped(sections, "प्रदेश")
        district = find_first_stripped(sections, "जिल्ला")
        municipality = find_first_stripped(sections, "गाउँपालिका / नगरपालिका")
        ward = find_first_stripped(sections, "वार्ड नम्बर")
        location = " ".join(p for p in (province, district, municipality) if p).strip()

        row["province"] = province
        row["district"] = district
        row["municipality"] = municipality
        row["ward"] = ward
        if location:
            row["location"] = location
            updated += 1
        else:
            still_blank += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"backfilled location for {updated} record(s), {still_blank} still blank -> {manifest_path}")
    print("also wrote structured province/district/municipality/ward columns (AM only -- PM has no "
          "detail-page JSON to backfill structured fields from, see module docstring).")


if __name__ == "__main__":
    main()
