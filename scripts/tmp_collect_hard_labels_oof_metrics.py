"""Collect tile-level in-domain metrics from hard_labels experiment outputs.

This script scans experiment folders for ``aggc_oof_tile_proba.npy``, compares
OOF tile probabilities against AGGC tile ground truth, and writes a summary CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect per-experiment OOF tile metrics into one CSV"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing hard_labels experiment folders",
    )
    parser.add_argument(
        "--tile-npz",
        type=Path,
        required=True,
        help=(
            "Path to AGGC tile npz with one-hot targets, e.g. "
            "Processed_Features/AGGC/20x_512/AGGC_tile_targets_masks_names.npz"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for experiment metric summary",
    )
    parser.add_argument(
        "--proba-filename",
        type=str,
        default="aggc_oof_tile_proba.npy",
        help="Probability filename to search for inside each experiment folder",
    )
    return parser.parse_args()


def normalize_probs(raw_probs: np.ndarray) -> np.ndarray:
    probs = raw_probs.astype(np.float64)
    probs = np.clip(probs, 0.0, None)

    row_sums = probs.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze(axis=1) > 0
    probs[nonzero] = probs[nonzero] / row_sums[nonzero]

    if (~nonzero).any():
        probs[~nonzero] = 1.0 / probs.shape[1]

    return probs


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, exp_name: str):
    y_pred = probs.argmax(axis=1)

    row = {
        "experiment": exp_name,
        "n_tiles": int(len(y_true)),
        "n_classes": int(probs.shape[1]),
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    try:
        row["log_loss"] = log_loss(y_true, probs, labels=np.arange(probs.shape[1]))
    except ValueError:
        row["log_loss"] = np.nan

    try:
        row["auc_macro_ovr"] = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro"
        )
        row["auc_weighted_ovr"] = roc_auc_score(
            y_true, probs, multi_class="ovr", average="weighted"
        )
        auc_each = roc_auc_score(y_true, probs, multi_class="ovr", average=None)
        for class_idx, class_auc in enumerate(auc_each):
            row[f"auc_class_{class_idx}"] = class_auc
    except ValueError:
        row["auc_macro_ovr"] = np.nan
        row["auc_weighted_ovr"] = np.nan

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(probs.shape[1]), zero_division=0
    )
    for class_idx in range(probs.shape[1]):
        row[f"precision_class_{class_idx}"] = prec[class_idx]
        row[f"recall_class_{class_idx}"] = rec[class_idx]
        row[f"f1_class_{class_idx}"] = f1[class_idx]
        row[f"support_class_{class_idx}"] = int(support[class_idx])

    return row


def main():
    args = parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {args.results_dir}")
    if not args.tile_npz.exists():
        raise FileNotFoundError(f"tile npz not found: {args.tile_npz}")

    tile_npz = np.load(args.tile_npz)
    if "targets" not in tile_npz:
        raise KeyError(f"'targets' not found in npz keys: {list(tile_npz.keys())}")

    targets = tile_npz["targets"]
    if targets.ndim != 2:
        raise ValueError(f"targets must be 2D one-hot array, got shape {targets.shape}")
    y_true = targets.argmax(axis=1)

    proba_paths = sorted(args.results_dir.rglob(args.proba_filename))
    if not proba_paths:
        raise FileNotFoundError(
            f"No '{args.proba_filename}' found under {args.results_dir}"
        )

    rows = []
    for proba_path in proba_paths:
        exp_name = str(proba_path.parent.relative_to(args.results_dir))
        probs = np.load(proba_path)

        if probs.ndim != 2:
            print(f"[skip] {exp_name}: proba not 2D (shape={probs.shape})")
            continue
        if probs.shape[0] != y_true.shape[0]:
            print(
                f"[skip] {exp_name}: n_tiles mismatch "
                f"(proba={probs.shape[0]}, gt={y_true.shape[0]})"
            )
            continue

        probs = normalize_probs(probs)
        rows.append(compute_metrics(y_true, probs, exp_name))

    if not rows:
        raise RuntimeError("No valid experiment files were processed.")

    df = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Processed experiments: {len(df)}")
    print(f"Saved metrics CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
