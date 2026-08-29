"""One-off backfill: adds raw_event_date/calendar_type/event_date_normalized
columns to am_persons.csv and pm_persons.csv from the existing `event_date`
column, per docs/P0_5_IMPLEMENTATION.md Phase 5.

No network requests -- pure re-derivation from data already on disk. Safe to
re-run (idempotent: recomputes from `event_date` each time, doesn't accumulate).

dvi.scoring.date_score() works correctly even without running this (it computes
normalization on the fly via dvi.dates.get_normalized() if these columns are
absent) -- this script exists to PERSIST the normalized/raw/calendar_type split
on the manifest itself, per Phase 5's "preserve both" requirement, so the UI and
any other consumer can see calendar_type without recomputing it.

Usage:
    uv run python scripts/normalize_dates.py
    uv run python scripts/normalize_dates.py --record-type pm
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvi.dates import normalize_date

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"


def backfill(record_type: str) -> None:
    manifest_path = MANIFEST_DIR / f"{record_type}_persons.csv"
    if not manifest_path.exists():
        print(f"[{record_type}] no manifest at {manifest_path}, skipping")
        return

    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[{record_type}] manifest is empty, skipping")
        return

    n_bs, n_ad, n_unknown = 0, 0, 0
    for row in rows:
        normalized = normalize_date(row.get("event_date", ""))
        row["raw_event_date"] = normalized["raw_event_date"]
        row["calendar_type"] = normalized["calendar_type"]
        row["event_date_normalized"] = normalized["event_date_normalized"]
        if normalized["calendar_type"] == "BS":
            n_bs += 1
        elif normalized["calendar_type"] == "AD":
            n_ad += 1
        else:
            n_unknown += 1

    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{record_type}] {len(rows)} record(s): {n_bs} BS, {n_ad} AD, {n_unknown} unknown/unparseable "
          f"-> {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-type", choices=["am", "pm", "both"], default="both")
    args = ap.parse_args()
    for rt in (["am", "pm"] if args.record_type == "both" else [args.record_type]):
        backfill(rt)


if __name__ == "__main__":
    main()
