#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack per-slide H5 tile features into a single memory-mapped .npy file."""

import csv
import json
import os
import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Allow imports from the project root (one level above scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_paths, get_env

DEF_FEATURE_DIM = 1024
DEF_WSI_COL = "WSI File Names"
DEF_LABEL_COL = "ISUP_grade_group"


def to_wsi_base(wsi_name: str) -> str:
    """Extract the base filename (no extension) from a WSI name string."""
    name = str(wsi_name).split(";")[0].strip()
    base = os.path.splitext(os.path.basename(name))[0]
    return base


def pass1_count_total_tiles(meta_df: pd.DataFrame, scale_dir: str, wsi_col: str,
                            label_col: str, feature_dim: int):
    """Count total tiles under a given scale and return a record list.

    Each record is a tuple of (slide_id, label, h5_path, n_tiles).
    """
    records = []
    total = 0
    for _, row in meta_df.iterrows():
        slide_id = to_wsi_base(row[wsi_col])
        label = int(row[label_col])
        h5_path = os.path.join(scale_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                dset = f["features"]
                n, d = dset.shape
                if d != feature_dim or n <= 0:
                    continue
                total += n
                records.append((slide_id, label, h5_path, n))
        except Exception as e:
            print(f"[WARN] skip {h5_path}: {e}")
            continue
    return total, records


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def pass2_write_memmap_and_index(records, out_dir: str, scale_name: str, total_tiles: int,
                                 feature_dim: int):
    """Write all tiles sequentially into a memmap and output a slide index CSV."""
    scale_out = os.path.join(out_dir, scale_name)
    ensure_dir(scale_out)

    memmap_path = os.path.join(scale_out, "features.npy")
    idx_csv_path = os.path.join(scale_out, "index.csv")

    mmap = np.memmap(memmap_path, dtype=np.float32, mode="w+",
                     shape=(total_tiles, feature_dim))

    idx_rows = []
    cursor = 0
    for slide_id, label, h5_path, n in records:
        try:
            with h5py.File(h5_path, "r") as f:
                feats = f["features"][:]  # load all tiles at once
                if feats.shape[1] != feature_dim:
                    raise ValueError(f"Feature dim mismatch reading {h5_path}")
                k = feats.shape[0]
                mmap[cursor: cursor + k, :] = feats.astype(np.float32, copy=False)
                idx_rows.append({
                    "slide_id": slide_id,
                    "y": int(label),
                    "start": int(cursor),
                    "length": int(k),
                    "n_tiles_read": int(k),
                    "h5_path": h5_path,
                })
                cursor += k
        except Exception as e:
            print(f"[WARN] failed read/write {h5_path}: {e}")
            continue

    mmap.flush()

    with open(idx_csv_path, "w", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=["slide_id", "y", "start", "length", "n_tiles_read", "h5_path"])
        writer.writeheader()
        writer.writerows(idx_rows)

    return {
        "scale": scale_name,
        "memmap_path": memmap_path,
        "index_csv": idx_csv_path,
        "n_slides": len(idx_rows),
        "n_tiles_total": int(cursor),
        "feature_dim": feature_dim,
    }


def main():
    ap = argparse.ArgumentParser("Pack WSI tile features into memmap matrices (per scale).")
    ap.add_argument("--env", type=str, default=None,
                    help="Environment name from paths.yaml (default: from BASELINE_ENV or 'nscc')")
    ap.add_argument("--csv", type=str, default=None,
                    help="Metadata CSV override (default: aggc_meta from paths.yaml)")
    ap.add_argument("--out_dir", type=str, default="packed",
                    help="Output dir for packed files")
    ap.add_argument("--feature_dim", type=int, default=DEF_FEATURE_DIM,
                    help="Tile feature dimension (default: 1024)")
    ap.add_argument("--wsi_col", type=str, default=DEF_WSI_COL,
                    help="CSV column for WSI filenames")
    ap.add_argument("--label_col", type=str, default=DEF_LABEL_COL,
                    help="CSV column for labels")
    args = ap.parse_args()

    # Load environment-specific paths from config.
    env = get_env(args.env)
    paths = load_paths(env)

    # Resolve metadata CSV path.
    csv_path = args.csv if args.csv else str(paths["metadata"]["aggc_meta"])
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    meta_df = pd.read_csv(csv_path)

    for col in [args.wsi_col, args.label_col]:
        if col not in meta_df.columns:
            raise ValueError(f"CSV is missing required column: {col}")

    # Build scale-dir mapping from config raw_features section.
    scale_dirs = {
        name: str(path)
        for name, path in paths["raw_features"].items()
    }

    ensure_dir(args.out_dir)
    summary = {"feature_dim": args.feature_dim, "scales": []}

    for scale_name, scale_dir in scale_dirs.items():
        print(f"\n=== Scale: {scale_name} ===")
        print(f"src: {scale_dir}")
        if not os.path.isdir(scale_dir):
            print(f"[WARN] scale dir not found, skip: {scale_dir}")
            continue

        total_tiles, records = pass1_count_total_tiles(
            meta_df, scale_dir, args.wsi_col, args.label_col, args.feature_dim
        )
        if total_tiles == 0 or len(records) == 0:
            print(f"[WARN] no usable slides for scale {scale_name}, skip.")
            continue

        print(f"[PASS1] slides usable: {len(records)}, total tiles: {total_tiles}")
        info = pass2_write_memmap_and_index(
            records, args.out_dir, scale_name, total_tiles, args.feature_dim
        )
        print(f"[PASS2] wrote memmap: {info['memmap_path']}")
        print(f"[PASS2] wrote index : {info['index_csv']}")
        summary["scales"].append(info)

    sum_path = os.path.join(args.out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nAll done. Summary -> {sum_path}")


if __name__ == "__main__":
    main()
