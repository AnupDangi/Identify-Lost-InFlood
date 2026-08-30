"""Legacy shim -- use research/data_collection/scrape_udb.py directly.

This file exists only for backwards compatibility with older docs/scripts.
The scraper exists only to reproduce the independent research prototype.
It is NOT the proposed institutional ingestion mechanism -- see
ARCHITECTURE.md and README.md (Authorized pilot vs prototype).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data_collection"))
from scrape_udb import *  # noqa: F401,F403
import scrape_udb as _m
if __name__ == "__main__":
    _m.main()
