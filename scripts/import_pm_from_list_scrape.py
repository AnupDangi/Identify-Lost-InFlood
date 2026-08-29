"""
Import the PM (unidentified body) dataset from the ad-hoc list-page scrape
already sitting in data/raw/images/pm/ (dead_bodies_dataset.csv +
dead_bodies_images/) into the normalized manifest schema used by
scripts/compare_am_pm.py and scripts/scrape_udb.py's am_persons.csv.

Why this exists: the "structured" CSV that script produced
(dead_bodies_structured_dataset.csv) has every metadata column blank -- its
`' | '.join(...)` + `':-' in part` splitting logic put each label
("नाम:-") and its value ("नखुलेको") in separate pipe-delimited tokens, so
`part.split(':-', 1)` never found a value on the same token. The raw
`Details` column in dead_bodies_dataset.csv still has the full pipe-delimited
token stream, so we parse THAT properly here instead of re-scraping.

Caveat: this data came from the LIST pages only (not each record's detail
page), so height/clothing/distinguishing-marks/fingerprint are not available
-- only what dis played in the list row: name (usually "नखुलेको" /
unidentified), sex, age / estimated age, found location, found date/time,
current facility, and the disaster tag badge (e.g. "रसुवा विपद्को शव").
There is also no real case ID captured (the scraper only saved a
page/index-based filename), so record_id here is a placeholder
"PM-LIST-p{page}-i{index}", not the site's actual case number. Re-run
scripts/scrape_udb.py --record-type pm if you need detail-page fields or a
real source_ref back to the case page.

Usage:
    uv run python scripts/import_pm_from_list_scrape.py
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PM_DIR = ROOT / "data" / "raw" / "images" / "pm"
RAW_CSV = PM_DIR / "dead_bodies_dataset.csv"
IMAGES_DIR = PM_DIR / "dead_bodies_images"
OUT_MANIFEST = ROOT / "data" / "manifests" / "pm_persons.csv"

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def to_latin_digits(s: str) -> str:
    return s.translate(DEVANAGARI_DIGITS)


def all_numbers(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", to_latin_digits(s))]


def parse_details(details: str) -> dict[str, str]:
    """Walk the ' | '-joined token stream and group tokens under the most
    recent 'label:-' token, per the row markup this came from (see
    docs/project_requirement.md's scraping notes)."""
    tokens = [t.strip() for t in str(details).split("|")]
    fields: dict[str, list[str]] = {}
    preamble: list[str] = []
    current_label: Optional[str] = None
    for tok in tokens:
        if not tok:
            continue
        if tok.endswith(":-"):
            current_label = tok[:-2].strip()
            fields.setdefault(current_label, [])
        elif current_label is not None:
            fields[current_label].append(tok)
        else:
            preamble.append(tok)
    joined = {k: re.sub(r"\s+", " ", " ".join(v)).strip(", ") for k, v in fields.items()}
    joined["_preamble"] = " | ".join(preamble)  # serial number + disaster tag badge
    return joined


def extract_disaster_tag(preamble: str) -> str:
    # preamble looks like "१ | रसुवा विपद्को शव" -- drop the leading serial number
    parts = [p.strip() for p in preamble.split("|") if p.strip()]
    for p in parts:
        if not re.fullmatch(r"[०-९0-9]+", p):
            return p
    return ""


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"not found: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    print(f"loaded {len(df)} raw rows from {RAW_CSV}")

    records = []
    missing_images = 0
    for _, row in df.iterrows():
        details = parse_details(row["Details"])

        name = details.get("नाम", "").strip()
        if name in ("नखुलेको", ""):
            name = ""  # "unidentified" is not a name

        sex_raw = details.get("लिङ्ग", "").strip().lower()
        sex = {"पुरुष": "male", "महिला": "female", "अन्य": "other",
               "male": "male", "female": "female"}.get(sex_raw, sex_raw)

        ages = all_numbers(details.get("उमेर", ""))
        est_ages = all_numbers(details.get("अन्दाजी उमेर", ""))
        all_ages = ages + est_ages
        age_min = min(all_ages) if all_ages else None
        age_max = max(all_ages) if all_ages else None

        location = details.get("भेटिएको ठाउँ", "")
        event_date_raw = details.get("भेटिएको मिति/समय", "")
        m = re.search(r"\d{4}-\d{2}-\d{2}", event_date_raw)
        event_date = m.group(0) if m else ""

        facility = details.get("हाल शव राखेको स्थान", "")
        disaster_tag = extract_disaster_tag(details.get("_preamble", ""))

        img_name = row["Image_Filename"]
        img_src = IMAGES_DIR / img_name
        page = row["Page"]
        # index comes from the filename itself (dead_bodies_p{page}_i{index}.jpg)
        idx_m = re.search(r"_i(\d+)\.jpg$", str(img_name))
        idx = idx_m.group(1) if idx_m else "0"
        record_id = f"PM-LIST-p{page}-i{idx}"

        image_path = ""
        sha256 = ""
        if img_src.exists():
            raw_bytes = img_src.read_bytes()
            if raw_bytes:
                sha256 = hashlib.sha256(raw_bytes).hexdigest()
                image_path = str(img_src.relative_to(ROOT))
            else:
                missing_images += 1
        else:
            missing_images += 1

        records.append({
            "record_id": record_id,
            "record_type": "PM",
            "name": name,
            "sex": sex,
            "age_min": age_min,
            "age_max": age_max,
            "height_cm": None,  # not available from list-page scrape
            "event_date": event_date,
            "location": location,
            "clothing": "",  # not available from list-page scrape
            "distinguishing_marks": "",  # not available from list-page scrape
            "image_path": image_path,
            "image_sha256": sha256,
            "source_ref": "",  # no case id captured by the list-page scrape
            "scraped_at": "",
            "current_facility": facility,
            "disaster_tag": disaster_tag,
        })

    fieldnames = ["record_id", "record_type", "name", "sex", "age_min", "age_max",
                  "height_cm", "event_date", "location", "clothing",
                  "distinguishing_marks", "image_path", "image_sha256", "source_ref",
                  "scraped_at", "current_facility", "disaster_tag"]
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    n_with_date = sum(1 for r in records if r["event_date"])
    n_with_age = sum(1 for r in records if r["age_min"] is not None)
    n_with_loc = sum(1 for r in records if r["location"])
    print(f"wrote {len(records)} PM record(s) -> {OUT_MANIFEST}")
    print(f"  with event_date: {n_with_date}, with age: {n_with_age}, with location: {n_with_loc}")
    print(f"  missing/unreadable image files: {missing_images}")


if __name__ == "__main__":
    main()
