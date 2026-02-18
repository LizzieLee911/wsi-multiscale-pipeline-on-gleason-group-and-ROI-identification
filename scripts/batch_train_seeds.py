"""Multi-seed experiment runner for statistical comparison.

Usage
-----
    python scripts/batch_train_seeds.py --env nscc
    python scripts/batch_train_seeds.py --env nscc --model LR_SGD --combine-2048
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_env, load_params, load_paths
from src.data.features import prepare_features
from src.evaluation.grading import agg_from_tiles
from src.evaluation.metrics import evaluate_slide_predictions
from src.models.train import train_one_seed
from src.utils.io import save_json_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed baseline experiment")
    p.add_argument("--env", type=str, default=None)
    p.add_argument("--model", type=str, default="LR_SGD",
                   choices=["LR_SGD", "XGB", "MLP"])
    p.add_argument("--combine-2048", action="store_true", dest="combine_2048",
                   default=False)
    p.add_argument("--no-combine-2048", action="store_false",
                   dest="combine_2048")
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--no-save", action="store_false", dest="save")
    p.add_argument("--agg-input", type=str, default="labels",
                   choices=["proba", "labels"],
                   help="Slide aggregation input: use tile probabilities or hard labels")
    p.add_argument("--config-dir", type=str, default=None)
    return p.parse_args()


def run_experiment(aggc_feats, tcga_feats, labels, df_aggc_idx, df_tcga_idx,
                   truth_tcga_df, truth_aggc_df, model_name, model_params,
                   seeds, n_splits, split_level, agg_input="proba",
                   save_dir=None):
    """Run over all *seeds* and return a list of per-seed result dicts."""
    results = []
    for seed in seeds:
        print(f"\n{'='*40} Seed {seed} {'='*40}")
        r = train_one_seed(
            seed, aggc_feats, tcga_feats, labels,
            df_aggc_idx, df_tcga_idx, truth_tcga_df, truth_aggc_df,
            model_name, model_params, n_splits, split_level,
        )

        out_tcga = agg_from_tiles(
            df_tcga_idx,
            r["tcga_proba_mean"] if agg_input == "proba" else r["tcga_pred_labels"],
            agg_input=agg_input,
        )
        out_aggc = agg_from_tiles(
            df_aggc_idx,
            r["oof_proba"] if agg_input == "proba" else r["oof_pred_labels"],
            agg_input=agg_input,
        )
        results_tcga = evaluate_slide_predictions(out_tcga, truth_tcga_df, verbose=False)
        results_aggc = evaluate_slide_predictions(out_aggc, truth_aggc_df, verbose=False)
        r["results_tcga"] = results_tcga
        r["results_aggc"] = results_aggc
        r["tcga_balanced_acc"] = results_tcga["balanced_acc_valid"]
        r["aggc_balanced_acc"] = results_aggc["balanced_acc_valid"]

        results.append(r)

        if save_dir:
            seed_dir = Path(save_dir)
            seed_dir.mkdir(parents=True, exist_ok=True)

            np.save(seed_dir / f"aggc_oof_tile_labels_seed{seed}.npy",
                    r["oof_pred_labels"])
            np.save(seed_dir / f"aggc_oof_tile_proba_seed{seed}.npy",
                    r["oof_proba"])
            np.save(seed_dir / f"tcga_proba_mean_seed{seed}.npy",
                    r["tcga_proba_mean"])
            joblib.dump(r["models"], seed_dir / f"models_seed{seed}.pkl")
            save_json_metrics(r["results_aggc"],
                              seed_dir / f"metrics_aggc_seed{seed}.json")
            save_json_metrics(r["results_tcga"],
                              seed_dir / f"metrics_tcga_seed{seed}.json")

    if save_dir:
        summary = [{
            "seed": r["seed"],
            "tcga_balanced_acc": r["tcga_balanced_acc"],
            "aggc_balanced_acc": r["aggc_balanced_acc"],
        } for r in results]
        pd.DataFrame(summary).to_csv(
            Path(save_dir) / "summary_metrics.csv", index=False
        )

    return results


def main():
    args = parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else None
    env = get_env(args.env)
    paths = load_paths(env, config_dir)
    params = load_params(config_dir)

    seeds = params["batch_training"]["seeds"]
    model_name = args.model
    model_params = params["models"][model_name]
    n_splits = params["training"]["n_splits"]
    split_level = params["training"].get("split_level", "slide")
    norm_mode = params["training"].get("norm_mode", "multiscale")

    # --- Load data ---
    df_aggc_idx = pd.read_csv(paths["features"]["aggc_1024"]["index_csv"])
    tile_npz = np.load(paths["features"]["aggc_1024"]["tile_npz"])
    labels = tile_npz["targets"]

    df_tcga_idx = pd.read_csv(paths["features"]["tcga_1024"]["index_csv"])

    truth_tcga_df = pd.read_csv(paths["metadata"]["tcga_truth"])
    filter_col = params["evaluation"].get("tcga_filter_col")
    filter_val = params["evaluation"].get("tcga_filter_val")
    if filter_col:
        truth_tcga_df = truth_tcga_df[truth_tcga_df[filter_col] == filter_val]
    truth_aggc_df = pd.read_csv(paths["metadata"]["aggc_meta"])

    # --- Run default features ---
    aggc_default, tcga_default = prepare_features(
        paths["features"], combine_2048=False, l2_norm=True,
        norm_mode=norm_mode,
    )
    save_dir_default = None
    if args.save:
        save_dir_default = (
            paths["output"]["base_save_dir"]
            / f"{model_name}_l2_tileonly_seeds" / "default"
        )
    results_default = run_experiment(
        aggc_default, tcga_default, labels,
        df_aggc_idx, df_tcga_idx, truth_tcga_df, truth_aggc_df,
        model_name, model_params, seeds, n_splits, split_level,
        args.agg_input,
        save_dir_default,
    )

    # --- Run combined 2048 features ---
    aggc_combine, tcga_combine = prepare_features(
        paths["features"], combine_2048=True, l2_norm=True,
        norm_mode=norm_mode,
    )
    save_dir_combine = None
    if args.save:
        save_dir_combine = (
            paths["output"]["base_save_dir"]
            / f"{model_name}_l2_with2048_seeds" / "combine_2048"
        )
    results_combine = run_experiment(
        aggc_combine, tcga_combine, labels,
        df_aggc_idx, df_tcga_idx, truth_tcga_df, truth_aggc_df,
        model_name, model_params, seeds, n_splits, split_level,
        args.agg_input,
        save_dir_combine,
    )

    # --- Statistical comparison ---
    tcga_default_acc = [r["tcga_balanced_acc"] for r in results_default]
    tcga_combine_acc = [r["tcga_balanced_acc"] for r in results_combine]
    aggc_default_acc = [r["aggc_balanced_acc"] for r in results_default]
    aggc_combine_acc = [r["aggc_balanced_acc"] for r in results_combine]

    tcga_t = stats.ttest_ind(tcga_combine_acc, tcga_default_acc, equal_var=False)
    aggc_t = stats.ttest_ind(aggc_combine_acc, aggc_default_acc, equal_var=False)

    print("\n" + "=" * 60)
    print("TCGA balanced acc (default):", np.mean(tcga_default_acc))
    print("TCGA balanced acc (combine):", np.mean(tcga_combine_acc))
    print("TCGA t-test:", tcga_t)
    print()
    print("AGGC balanced acc (default):", np.mean(aggc_default_acc))
    print("AGGC balanced acc (combine):", np.mean(aggc_combine_acc))
    print("AGGC t-test:", aggc_t)


if __name__ == "__main__":
    main()
