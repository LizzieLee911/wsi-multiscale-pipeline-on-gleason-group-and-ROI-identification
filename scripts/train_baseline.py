"""Main CLI for tile-level baseline classifier training.

Usage
-----
    python scripts/train_baseline.py --model LR_SGD --env nscc
    python scripts/train_baseline.py --model XGB --combine-2048 --l2 --save
    python scripts/train_baseline.py --model MLP --no-l2 --env local
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import get_env, load_params, load_paths
from src.data.features import prepare_features
from src.data.splits import compute_class_weights
from src.evaluation.grading import agg_from_tiles
from src.evaluation.metrics import evaluate_slide_predictions
from src.models.train import cross_validate, predict_external
from src.utils.io import build_save_path, save_experiment_artifacts

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",)


def parse_args():
    p = argparse.ArgumentParser(description="Tile-level baseline classifier")
    p.add_argument("--model", type=str, default="LR_SGD",
                   choices=["LR_SGD", "XGB", "MLP"])
    p.add_argument("--env", type=str, default=None,
                   help="Environment name (nscc, local)")
    p.add_argument("--l2", action="store_true", dest="l2", default=True)
    p.add_argument("--no-l2", action="store_false", dest="l2")
    p.add_argument("--combine-2048", action="store_true", dest="combine_2048",
                   default=False)
    p.add_argument("--no-combine-2048", action="store_false",
                   dest="combine_2048")
    p.add_argument("--save", action="store_true", dest="save", default=False)
    p.add_argument("--no-save", action="store_false", dest="save")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agg-input", type=str, default="proba",
                   choices=["proba", "labels"],
                   help="Slide aggregation input: use tile probabilities or labels")
    p.add_argument("--config-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else None
    env = get_env(args.env)
    paths = load_paths(env, config_dir)
    params = load_params(config_dir)

    print(f"Environment: {env}")
    print(f"Model: {args.model} | L2: {args.l2} | 2048: {args.combine_2048}")

    # --- Prepare features ---
    norm_mode = params["training"].get("norm_mode", "multiscale")
    aggc_feats, tcga_feats = prepare_features(
        paths["features"],
        combine_2048=args.combine_2048,
        l2_norm=args.l2,
        norm_mode=norm_mode,
    )

    # --- Load metadata and targets ---
    df_aggc_idx = pd.read_csv(paths["features"]["aggc_1024"]["index_csv"])
    tile_npz = np.load(paths["features"]["aggc_1024"]["tile_npz"])
    Y = tile_npz["targets"]
    y_major = Y.argmax(axis=1)

    df_tcga_idx = pd.read_csv(paths["features"]["tcga_1024"]["index_csv"])

    truth_tcga_df = pd.read_csv(paths["metadata"]["tcga_truth"])
    filter_col = params["evaluation"].get("tcga_filter_col")
    filter_val = params["evaluation"].get("tcga_filter_val")
    if filter_col:
        truth_tcga_df = truth_tcga_df[truth_tcga_df[filter_col] == filter_val]

    truth_aggc_df = pd.read_csv(paths["metadata"]["aggc_meta"])

    # --- Train ---
    model_params = params["models"][args.model]
    split_level = params["training"].get("split_level", "slide")
    n_splits = params["training"].get("n_splits", 5)

    cv_result = cross_validate(
        aggc_feats, y_major, df_aggc_idx,
        args.model, model_params,
        n_splits=n_splits, seed=args.seed,
        split_level=split_level,
    )

    tile_m = cv_result["tile_metrics"]
    print(f"\n  OOF accuracy       = {tile_m['acc']:.4f}")
    print(f"  OOF balanced acc   = {tile_m['bacc']:.4f}")
    print(f"  OOF macro AUC      = {tile_m['auc_macro']:.4f}")
    print(f"  OOF macro F1       = {tile_m['f1_macro']:.4f}")
    print(f"  OOF weighted F1    = {tile_m['f1_weighted']:.4f}")

    # --- Predict TCGA ---
    tcga_proba_mean, tcga_pred_labels = predict_external(
        cv_result["models"], tcga_feats
    )

    # --- Slide-level aggregation and evaluation ---
    out_tcga = agg_from_tiles(
        df_tcga_idx,
        tcga_proba_mean if args.agg_input == "proba" else tcga_pred_labels,
        agg_input=args.agg_input,
    )
    out_aggc = agg_from_tiles(
        df_aggc_idx,
        (
            cv_result["oof_proba"]
            if args.agg_input == "proba"
            else cv_result["oof_pred_labels"]
        ),
        agg_input=args.agg_input,
    )

    print("\n--- TCGA slide-level ---")
    results_tcga = evaluate_slide_predictions(out_tcga, truth_tcga_df)

    print("\n--- AGGC slide-level (OOF) ---")
    results_aggc = evaluate_slide_predictions(out_aggc, truth_aggc_df)

    # --- Save ---
    if args.save:
        save_path = build_save_path(
            paths["output"]["base_save_dir"],
            args.model, args.l2, args.combine_2048,
        )
        save_experiment_artifacts(
            save_path,
            cv_result["models"],
            cv_result["oof_proba"],
            cv_result["oof_pred_labels"],
            tcga_proba_mean,
            out_tcga,
            out_aggc,
            results_tcga,
            results_aggc,
        )
        print(f"\nArtifacts saved to: {save_path}")


if __name__ == "__main__":
    main()
