"""
Research ingestion adapter for the public Nepal Police UDB listings.

This adapter exists ONLY to reproduce the independent research prototype.
It is NOT the proposed institutional ingestion mechanism.

    PROTOTYPE:            Public Nepal Police UDB → this adapter
    AUTHORIZED PILOT:     Nepal Police AM/PM records → approved API / export / internal database

See README.md (Data handling) and ARCHITECTURE.md. Do not present the
prototype scraper as the future system when showing this to Nepal Police.

Authorized scrape of the Nepal Police Unidentified/Missing Person Database
(UDB) at udb.nepalpolice.gov.np, for AM (missing person) <-> PM (unidentified
body) reconciliation as described in docs/project_requirement.md.

This script is only meant to be run under an authorized basis (institutional /
police authorization) -- see README.md. It writes to data/, which is
gitignored and must never be committed, published, or moved to a public
bucket.

By default it walks EVERY page of both lists (no date filter) -- pass
--date-from/--date-to to scope it to a window instead. Detail pages and
photos are fetched concurrently (bounded worker pool, per-worker pacing) and
already-downloaded records are skipped on re-run, so an interrupted full
scrape can just be re-launched.

Usage:
    # full scrape, both lists, moderate concurrency
    uv run python scripts/scrape_udb.py --record-type both --concurrency 6

    # scoped to a date window
    uv run python scripts/scrape_udb.py --date-from 2026-08-20 --date-to 2026-08-29

    # smoke test
    uv run python scripts/scrape_udb.py --record-type am --limit 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvi.dates import normalize_date

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://udb.nepalpolice.gov.np"
USER_AGENT = "Mozilla/5.0 (compatible; DVI-Research-Scraper/0.1; authorized research use)"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RAW_DIR = DATA_DIR / "raw"

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

write_lock = threading.Lock()
progress_lock = threading.Lock()


def to_latin_digits(s: str) -> str:
    return s.translate(DEVANAGARI_DIGITS)


def clean(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def all_numbers(s: str) -> list[float]:
    s = to_latin_digits(s)
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]


def height_to_cm(raw: str) -> float | None:
    raw = clean(raw)
    if not raw:
        return None
    nums = all_numbers(raw)
    if not nums:
        return None
    if "फिट" in raw or re.search(r"\bft\b", raw, re.IGNORECASE):
        feet = nums[0]
        inches = nums[1] if len(nums) > 1 else 0.0
        return round(feet * 30.48 + inches * 2.54, 1)
    if "cm" in raw.lower() or "से" in raw:
        return nums[0]
    if nums[0] < 9:  # bare small number -> assume feet
        return round(nums[0] * 30.48, 1)
    return nums[0]


@dataclass
class ListItem:
    record_id: str
    record_type: str  # AM | PM
    detail_url: str
    photo_url: str


def make_session(pool_size: int) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "ne,en;q=0.8",
    })
    s.verify = False
    
    # ADDED: Low-level retries for socket drops (SSLEOFError, ConnectionReset)
    retries = Retry(
        total=5, 
        backoff_factor=0.5, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch(session: requests.Session, url: str, params: dict | None = None,
          retries: int = 3, delay: float = 0.5, timeout: int = 30) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            print(f"  ! HTTP {resp.status_code} for {url} (attempt {attempt})", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"  ! error fetching {url}: {exc} (attempt {attempt})", file=sys.stderr)
        time.sleep(delay * attempt)
    return None


def list_ids(session: requests.Session, record_type: str, date_from: str | None,
             date_to: str | None, delay: float, concurrency: int,
             limit: int | None = None) -> list[ListItem]:
    if record_type == "AM":
        list_url, detail_tpl, photo_tpl, link_marker = (
            f"{BASE}/missing", f"{BASE}/missing/{{id}}", f"{BASE}/missing/photo/{{id}}", "/missing/",
        )
    else:
        list_url, detail_tpl, photo_tpl, link_marker = (
            f"{BASE}/dead-bodies-lists", f"{BASE}/dead-bodies/{{id}}", f"{BASE}/deadbody/photo/{{id}}",
            "/dead-bodies/",
        )

    base_params = {}
    if date_from:
        base_params["date_from"] = date_from
    if date_to:
        base_params["date_to"] = date_to

    resp = fetch(session, list_url, params={**base_params, "page": 1}, delay=delay)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    total_pages = 1
    m = re.search(r"out of\s+(\d+)\s+Pages", soup.get_text())
    if m:
        total_pages = int(m.group(1))
    scope = f"{date_from or 'start'}..{date_to or 'now'}" if (date_from or date_to) else "full unfiltered list"
    print(f"[{record_type}] {total_pages} page(s) ({scope})")

    items: dict[str, ListItem] = {}
    items_lock = threading.Lock()

    def parse_page(page_soup: BeautifulSoup):
        found = []
        for a in page_soup.select(f'a[href*="{link_marker}"]'):
            href = a.get("href", "")
            mm = re.search(rf"{re.escape(link_marker)}(\d+)", href)
            if not mm:
                continue
            rid = mm.group(1)
            found.append(rid)
        with items_lock:
            for rid in found:
                if rid not in items:
                    items[rid] = ListItem(
                        record_id=rid,
                        record_type=record_type,
                        detail_url=detail_tpl.format(id=rid),
                        photo_url=photo_tpl.format(id=rid),
                    )

    parse_page(soup)

    def fetch_page(page: int):
        # ADDED: try/except wrapper so a bad page doesn't break f.result()
        try:
            params = {**base_params, "page": page}
            r = fetch(session, list_url, params=params, delay=delay)
            time.sleep(delay)
            if r is not None:
                parse_page(BeautifulSoup(r.text, "html.parser"))
        except Exception as _:  # noqa: BLE001
            print(f"  ! unhandled error on list page {page}: {_}", file=sys.stderr)

    pages = list(range(2, total_pages + 1))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fetch_page, p) for p in pages]
        done = 1
        for f in as_completed(futures):
            f.result()  # Safely returns None on caught exception now
            done += 1
            if done % 25 == 0 or done == len(pages) + 1:
                print(f"[{record_type}] list pages: {done}/{total_pages}")
            if limit and len(items) >= limit:
                break

    result = list(items.values())
    if limit:
        result = result[:limit]
    return result


def extract_sections(soup: BeautifulSoup) -> dict[str, dict[str, str]]:
    """Map each `.card-title` section to its {label: value} pairs, read from
    the repeating `<strong>label</strong><span>value</span>` rows Laravel
    renders for this template."""
    sections: dict[str, dict[str, str]] = {}
    for title_el in soup.select(".card-title"):
        section_name = clean(title_el.get_text())
        card = title_el.find_parent("div", class_="card-body") or title_el.parent
        if card is None:
            continue
        fields: dict[str, str] = {}
        for strong in card.find_all("strong"):
            # Some labels in the source markup have a space before the colon
            # ("प्रदेश :"), which rstrip(":") alone leaves behind as a
            # trailing space -- silently breaking every find_first() lookup
            # for that label (this is why AM's `location` column ended up
            # blank for the entire dataset; see ARCHITECTURE.md).
            label = clean(strong.get_text()).rstrip(":").strip()
            if not label:
                continue
            span = strong.find_next_sibling("span")
            if span is not None:
                value = clean(span.get_text())
            else:
                value = clean(strong.parent.get_text())[len(label):].lstrip(": ").strip()
            fields[label] = value
        sections[section_name] = fields
    return sections


def parse_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    name = clean(heading.get_text()) if heading else ""
    sections = extract_sections(soup)
    return {"name": name, "sections": sections}


def find_first(sections: dict[str, dict[str, str]], *labels: str) -> str:
    for section in sections.values():
        for label in labels:
            if section.get(label):
                return section[label]
    return ""


def normalize(record_type: str, item_id: str, parsed: dict, image_path: str,
              image_sha256: str, source_ref: str) -> dict:
    sections = parsed["sections"]
    name = parsed["name"] if record_type == "AM" else ""

    sex_raw = find_first(sections, "लिङ्ग").lower()
    
    # ADDED: "अन्य": "other" mapping for unidentified body parts
    sex_mapping = {
        "male": "male", 
        "female": "female", 
        "पुरुष": "male", 
        "महिला": "female", 
        "अन्य": "other"
    }
    sex = sex_mapping.get(sex_raw, sex_raw)

    age_raw = find_first(sections, "उमेर /अन्दाजी उमेर", "उमेर")
    ages = all_numbers(age_raw)
    age_min = ages[0] if ages else None
    age_max = ages[-1] if ages else None

    height_cm = height_to_cm(find_first(sections, "उचाई"))

    if record_type == "AM":
        event_date = find_first(sections, "हराएको मिति")
    else:
        event_date = find_first(sections, "भेटिएको मिति")
    province = find_first(sections, "प्रदेश")
    district = find_first(sections, "जिल्ला")
    municipality = find_first(sections, "गाउँपालिका / नगरपालिका")
    ward = find_first(sections, "वार्ड नम्बर")
    location = province + " " + district + " " + municipality

    # Phase 5: persist raw/calendar_type/normalized alongside the raw string --
    # UDB dates have no calendar marker and mix BS/AD, see dvi/dates.py.
    date_norm = normalize_date(clean(event_date))

    return {
        "record_id": f"{record_type}{item_id}",
        "record_type": record_type,
        "name": name,
        "sex": sex,
        "age_min": age_min,
        "age_max": age_max,
        "height_cm": height_cm,
        "event_date": clean(event_date),
        "raw_event_date": date_norm["raw_event_date"],
        "calendar_type": date_norm["calendar_type"],
        "event_date_normalized": date_norm["event_date_normalized"],
        "location": clean(location),
        "province": clean(province),
        "district": clean(district),
        "municipality": clean(municipality),
        "ward": clean(ward),
        "clothing": find_first(sections, "लगाएको लुगा"),
        "distinguishing_marks": find_first(sections, "विशेष चिन्ह", "विशिष्ट लक्षणहरु"),
        "image_path": image_path,
        "image_sha256": image_sha256,
        "source_ref": source_ref,
    }


def download_image(session: requests.Session, url: str, dest: Path, delay: float) -> tuple[str, str]:
    """Download, strip EXIF, save as JPEG. Returns (relative_path, sha256)."""
    resp = fetch(session, url, delay=delay)
    if resp is None or not resp.content:
        return "", ""
    raw = resp.content
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        data = list(img.getdata())
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(data)
        dest.parent.mkdir(parents=True, exist_ok=True)
        clean_img.save(dest, format="JPEG", quality=92)
    except Exception as _:  # noqa: BLE001
        print(f"  ! could not process image {url}: {_}", file=sys.stderr)
        return "", ""
    return str(dest.relative_to(ROOT)), sha256


def process_one(session: requests.Session, item: ListItem, sub: str, delay: float,
                 force: bool) -> dict | None:
    raw_json_path = RAW_DIR / "json" / sub / f"{item.record_id}.json"
    img_dest = RAW_DIR / "images" / sub / f"{item.record_id}.jpg"

    if not force and raw_json_path.exists() and img_dest.exists():
        parsed = json.loads(raw_json_path.read_text(encoding="utf-8"))
        sha256 = hashlib.sha256(img_dest.read_bytes()).hexdigest()
        record = normalize(item.record_type, item.record_id, parsed,
                            str(img_dest.relative_to(ROOT)), sha256, source_ref=item.detail_url)
        record["scraped_at"] = datetime.now(UTC).isoformat()
        return record

    resp = fetch(session, item.detail_url, delay=delay)
    time.sleep(delay)
    if resp is None:
        return None
    parsed = parse_detail(resp.text)

    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    raw_json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    image_path, sha256 = download_image(session, item.photo_url, img_dest, delay)
    time.sleep(delay)

    record = normalize(item.record_type, item.record_id, parsed, image_path, sha256,
                        source_ref=item.detail_url)
    record["scraped_at"] = datetime.now(UTC).isoformat()
    return record


FIELDNAMES = ["record_id", "record_type", "name", "sex", "age_min", "age_max",
              "height_cm", "event_date", "raw_event_date", "calendar_type",
              "event_date_normalized", "location", "province", "district",
              "municipality", "ward", "clothing", "distinguishing_marks",
              "image_path", "image_sha256", "source_ref", "scraped_at"]


def scrape(record_type: str, date_from: str | None, date_to: str | None, delay: float,
           concurrency: int, limit: int | None, force: bool, out_path: Path):
    session = make_session(pool_size=concurrency)
    items = list_ids(session, record_type, date_from, date_to, delay, concurrency, limit)
    print(f"[{record_type}] {len(items)} record(s) to fetch")

    sub = "am" if record_type == "AM" else "pm"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    f = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        f.flush()

    done = 0
    written = 0

    def worker(item: ListItem):
        # ADDED: catch-all error handling so one bad record doesn't crash the script
        try:
            return item, process_one(session, item, sub, delay, force)
        except Exception as _:  # noqa: BLE001
            print(f"  ! unhandled error processing {item.record_id}: {_}", file=sys.stderr)
            return item, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, item) for item in items]
        for fut in as_completed(futures):
            _ , record = fut.result()
            with progress_lock:
                done += 1
                if done % 20 == 0 or done == len(items):
                    print(f"[{record_type}] {done}/{len(items)} processed")
            if record:
                with write_lock:
                    writer.writerow(record)
                    f.flush()
                    written += 1

    f.close()
    print(f"[{record_type}] wrote {written} record(s) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date-from", default=None, help="YYYY-MM-DD (Gregorian); omit for no filter")
    ap.add_argument("--date-to", default=None, help="YYYY-MM-DD (Gregorian); omit for no filter")
    ap.add_argument("--record-type", choices=["am", "pm", "both"], default="both")
    ap.add_argument("--delay", type=float, default=0.4, help="per-worker pacing between requests")
    ap.add_argument("--concurrency", type=int, default=6, help="parallel workers (be respectful of the source server)")
    ap.add_argument("--limit", type=int, default=None, help="cap records per type (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if already downloaded")
    args = ap.parse_args()

    to_run = []
    if args.record_type in ("am", "both"):
        to_run.append("AM")
    if args.record_type in ("pm", "both"):
        to_run.append("PM")

    for rt in to_run:
        out = MANIFEST_DIR / ("am_persons.csv" if rt == "AM" else "pm_persons.csv")
        scrape(rt, args.date_from, args.date_to, args.delay, args.concurrency,
               args.limit, args.force, out)


if __name__ == "__main__":
    main()