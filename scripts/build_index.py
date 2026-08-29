"""
Build a FAISS IndexFlatIP over one side's (usually AM) usable face embeddings,
per docs/project_requirement.md section 11.

Usage:
    uv run python scripts/build_index.py --record-type am
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
EMBED_DIR = ROOT / "data" / "embeddings"
INDEX_DIR = ROOT / "data" / "index"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-type", choices=["am", "pm"], required=True)
    args = ap.parse_args()

    idx_csv = MANIFEST_DIR / f"{args.record_type}_embeddings_index.csv"
    df = pd.read_csv(idx_csv)
    usable = df[df["usable"] == True]

    vectors = []
    id_map = []
    embed_dir = EMBED_DIR / args.record_type
    for record_id in usable["record_id"]:
        npy_path = embed_dir / f"{record_id}.npy"
        if not npy_path.exists():
            continue
        vectors.append(np.load(npy_path))
        id_map.append(record_id)

    if not vectors:
        raise SystemExit(f"no embeddings found under {embed_dir} -- run build_embeddings.py first")

    matrix = np.vstack(vectors).astype(np.float32)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / f"{args.record_type}.index"))
    (INDEX_DIR / f"{args.record_type}_id_map.json").write_text(json.dumps(id_map, ensure_ascii=False))

    print(f"[{args.record_type}] indexed {len(id_map)} embedding(s), dim={dim}")
    print(f"wrote -> {INDEX_DIR / f'{args.record_type}.index'}")


if __name__ == "__main__":
    main()
