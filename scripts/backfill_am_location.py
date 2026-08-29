"""
One-off backfill for the AM `location` column, which was blank for the
entire dataset due to a label-parsing bug fixed in scrape_udb.py's
extract_sections() (trailing space before the colon in labels like
"प्रदेश :" wasn't stripped, so find_first("प्रदेश") never matched
"प्रदेश "). The actual location data was captured correctly in
data/raw/json/am/{id}.json at scrape time -- this just re-parses those
already-downloaded files with the fixed key matching, no network requests.

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

    updated = 0
    still_blank = 0
    for row in rows:
        record_id = row["record_id"]
        numeric_id = record_id[2:] if record_id.startswith("AM") else record_id
        json_path = RAW_JSON_DIR / f"{numeric_id}.json"
        if not json_path.exists():
            continue
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        sections = parsed.get("sections", {})
        province = find_first_stripped(sections, "प्रदेश")
        district = find_first_stripped(sections, "जिल्ला")
        municipality = find_first_stripped(sections, "गाउँपालिका / नगरपालिका")
        location = " ".join(p for p in (province, district, municipality) if p).strip()
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


if __name__ == "__main__":
    main()
