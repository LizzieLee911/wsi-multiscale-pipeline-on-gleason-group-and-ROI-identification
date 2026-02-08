# %%
import csv
import argparse
import zipfile
import pandas as pd
from pathlib import Path
import numpy as np, scipy.sparse as sp
import sys
from shapely.geometry import box, shape, Polygon,mapping

from datetime import datetime
import joblib

import os, json, h5py
from collections import Counter
import matplotlib.pyplot as plt
import geopandas as gpd
import pyvips
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,classification_report,confusion_matrix,balanced_accuracy_score
import warnings
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import SGDClassifier
from scipy import stats
warnings.filterwarnings("ignore")

import torch
if torch.cuda.is_available(): print("CUDA is available! PyTorch can see the GPU.")
else: print("CUDA is not available. PyTorch will use the CPU.")

os.chdir('/scratch/users/ntu/lizh0106/nscc_work')
print(os.getcwd())

SAVE = True

# %%
def load_memmap_features(path, feat_dim=1024, dtype=np.float32, copy=True):
    """infer memmp shape"""

    path = os.fspath(path)
    itemsize = np.dtype(dtype).itemsize
    filesize = os.path.getsize(path)

    total_elems = filesize // itemsize
    assert total_elems % feat_dim == 0, "File size not divisible by feat_dim"

    n_samples = total_elems // feat_dim
    print (n_samples)
    mm = np.memmap(path,dtype=dtype,mode="r",shape=(n_samples, feat_dim))

    return mm.copy() if copy else mm

# %%
AGGC_features      = "Processed_Features/AGGC/20x_512/features.npy"
AGGC_f_index       = "Processed_Features/AGGC/20x_512/index.csv"
AGGC_tile_npz_path = "Processed_Features/AGGC/20x_512/AGGC_tile_targets_masks_names.npz"
AGGC_meta_path     = "WsiBERT/AGGC_metadata.csv"

TCGA_features = "Processed_Features/TCGA_PRAD/tcga_without_anno_arrays/20x_512/features.npy"
TCGA_f_index = "Processed_Features/TCGA_PRAD/tcga_without_anno_arrays/20x_512/index.csv"

AGGC_tnm = np.load(AGGC_tile_npz_path)

#### SAVE
save_path = "WsiBERT/models_Baseline/LR_SGD_l2_with2048_seeds"
os.makedirs(save_path, exist_ok=True)


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms


def prepare_features(combine_2048):
    tcga_feats = load_memmap_features(TCGA_features)
    aggc_feats = load_memmap_features(AGGC_features)

    if combine_2048:
        tcga_features_2048 = "Processed_Features/TCGA_PRAD/tcga_without_anno_arrays_c_2048/2048/features.npy"
        aggc_features_2048 = "Processed_Features/AGGC/AGGC_CENTER/AGGC_20x512/features.npy"

        tcga_feats_2048 = load_memmap_features(tcga_features_2048)
        aggc_feats_2048 = load_memmap_features(aggc_features_2048)

        assert tcga_feats.shape[0] == tcga_feats_2048.shape[0]
        assert aggc_feats.shape[0] == aggc_feats_2048.shape[0]

        tcga_feats = np.concatenate([tcga_feats, tcga_feats_2048], axis=1)
        aggc_feats = np.concatenate([aggc_feats, aggc_feats_2048], axis=1)

    tcga_feats = l2_normalize_rows(tcga_feats)
    aggc_feats = l2_normalize_rows(aggc_feats)
    return aggc_feats, tcga_feats

# %%
# %% [markdown]
# primary >50%, secondary ≥5% need to aggregate to have the overall gleason score

# %%

pattern_map = {0:3, 1:4, 2:5}   # G3/G4/G5

# ISUP 规则: Gleason → ISUP group(0-4)
def gleason_to_isup(g1, g2):
    score = g1 + g2
    if score <= 6:
        return 0     # ISUP 1
    elif score == 7:
        if g1 == 3 and g2 == 4:
            return 1  # ISUP 2
        elif (g1 == 4 and g2 == 3):
            return 2  # ISUP 3
    elif score == 8:
        return 3      # ISUP 4
    else:  # 9 or 10
        return 4      # ISUP 5
def agg_from_tiles(df_tcga_index_tiles,tcga_pred_labels):
    '''
    df_tcga_index_tiles: df with assigned rows
    tcga_pred_labels: tile labels from 0-3
    '''
    out_rows = []
    for i, row in df_tcga_index_tiles.iterrows():
        slide_id = row["slide_id"]
        start, end = row["start"], row["start"] + row["length"]

        tiles_preds = tcga_pred_labels[start:end]

        # G3/G4/G5
        counts = np.bincount(tiles_preds, minlength=4)
        tumor_counts = counts[:3]  ##Only tumor, no others
        total_tumor = tumor_counts.sum()

        if total_tumor == 0:
            # if no tumore tiles
            out_rows.append({
                "slide_id": slide_id,
                "p3": 0.0, "p4": 0.0, "p5": 0.0,
                "primary_pattern": None,
                "secondary_pattern": None,
                "gleason": None,
                "ISUP_grade_group": None
            })
            continue

        p3, p4, p5 = tumor_counts / total_tumor
        fractions = np.array([p3, p4, p5])

        # 排序，最多和第二多
        order = np.argsort(-fractions)
        p1_idx, p2_idx = order[0], order[1]
        p1, p2 = fractions[p1_idx], fractions[p2_idx]

        # 医学规则
        if p1 >= 0.95 or p2 < 0.05:
            g1 = g2 = pattern_map[p1_idx]    # 3+3, 4+4, 5+5
        else:
            g1 = pattern_map[p1_idx]
            g2 = pattern_map[p2_idx]

        gleason_str = f"{g1}+{g2}"
        isup = gleason_to_isup(g1, g2)

        out_rows.append({
            "slide_id": slide_id,
            "p3": float(p3),
            "p4": float(p4),
            "p5": float(p5),
            "primary_pattern": g1,
            "secondary_pattern": g2,
            "gleason": gleason_str,
            "ISUP_grade_group": isup
        })

    out_df = pd.DataFrame(out_rows)
    return out_df

# %%
def evaluate_slide_predictions(out_df,truth_df,pred_col="ISUP_grade_group",true_col="ISUP_grade_group",
    treat_nan_as_class=5,verbose=True):

    pred_raw = out_df[pred_col].values
    true_raw = truth_df[true_col].values

    assert len(pred_raw) == len(true_raw), "Prediction and truth length mismatch."

    # ---------------------------
    # 2) Handle NaNs
    # ---------------------------
    nan_mask = np.isnan(pred_raw)

    pred_all = pred_raw.copy()
    pred_all[nan_mask] = treat_nan_as_class
    pred_all = pred_all.astype(int)
    true_all = true_raw.astype(int)

    # ---------------------------
    # 3) Metrics including NaN as a class
    # ---------------------------
    acc_all = accuracy_score(true_all, pred_all)
    labels_all = sorted(list(set(true_all)) + [treat_nan_as_class])
    cm_all = confusion_matrix(true_all, pred_all, labels=labels_all)

    # ---------------------------
    # 4) Metrics excluding NaN
    # ---------------------------
    mask_valid = ~nan_mask
    pred_valid = pred_all[mask_valid]
    true_valid = true_all[mask_valid]

    acc_valid = accuracy_score(true_valid, pred_valid)
    bal_acc_valid = balanced_accuracy_score(true_valid, pred_valid)

    clf_report = classification_report(
        true_valid, pred_valid,
        labels=sorted(list(set(true_valid))),
        output_dict=False
    )

    # ---------------------------
    # 5) Optional printout
    # ---------------------------
    if verbose:
        print("=== Slide-level Evaluation ===")
        print(f"Slides with NaN predictions: {nan_mask.sum()} / {len(pred_raw)}\n")

        print("--- Including NaNs (as class {}) ---".format(treat_nan_as_class))
        print("Accuracy:", acc_all)
        print("Confusion matrix:\n", cm_all, "\n")

        print("--- Excluding NaNs ---")
        print("Accuracy:", acc_valid)
        print("Balanced accuracy:", bal_acc_valid)
        print("Classification report:\n", clf_report)

    # ---------------------------
    # 6) Return results
    # ---------------------------
    return {
        "nan_count": int(nan_mask.sum()),
        "acc_all": acc_all,
        "confusion_matrix_all": cm_all,
        "acc_valid": acc_valid,
        "balanced_acc_valid": bal_acc_valid,
        "classification_report": clf_report
    }


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()   # 转成 python int/float
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

def save_json_metrics(metrics, filename):
    metrics = to_serializable(metrics)  # ← 关键一步

    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)


def train_one_seed(seed, aggc_feats, tcga_feats, labels, df_tcga_index_tiles, truth_tcga_df,
                   df_aggc_index_tiles, truth_aggc_df):
    y_major = labels.argmax(axis=1)
    n_samples = aggc_feats.shape[0]
    classes_all = np.unique(y_major)
    n_classes = len(classes_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    models = []
    oof_proba = np.zeros((n_samples, n_classes), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(aggc_feats, y_major), 1):
        X_tr, X_val = aggc_feats[train_idx], aggc_feats[val_idx]
        y_tr_major = y_major[train_idx]

        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=50,
            tol=1e-3,
            learning_rate="optimal",
            early_stopping=True,
            n_iter_no_change=5,
            class_weight="balanced",
            random_state=seed
        )
        model.fit(X_tr, y_tr_major)

        models.append(model)

        pred_val_proba_fold = model.predict_proba(X_val)
        temp = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        idx_map = np.searchsorted(classes_all, model.classes_)
        temp[:, idx_map] = pred_val_proba_fold
        oof_proba[val_idx] = temp

        print(f"Seed {seed} fold {fold} done. max n_iter = {np.max(model.n_iter_)}")

    oof_pred_labels = oof_proba.argmax(axis=1)

    tcga_proba_list = []
    for fold_idx, model in enumerate(models, 1):
        pred_tcga_proba = model.predict_proba(tcga_feats)
        tcga_proba_list.append(pred_tcga_proba)
        print(f"Seed {seed} TCGA prediction from fold {fold_idx} model done. Shape: {pred_tcga_proba.shape}")

    tcga_proba_stack = np.stack(tcga_proba_list, axis=0)
    tcga_proba_mean = tcga_proba_stack.mean(axis=0)
    tcga_pred_labels = tcga_proba_mean.argmax(axis=1)

    out_tcga = agg_from_tiles(df_tcga_index_tiles, tcga_pred_labels)
    out_aggc = agg_from_tiles(df_aggc_index_tiles, oof_pred_labels)

    results_tcga = evaluate_slide_predictions(out_tcga, truth_tcga_df, verbose=False)
    results_aggc = evaluate_slide_predictions(out_aggc, truth_aggc_df, verbose=False)

    return {
        "seed": seed,
        "models": models,
        "oof_proba": oof_proba,
        "oof_pred_labels": oof_pred_labels,
        "tcga_proba_mean": tcga_proba_mean,
        "tcga_pred_labels": tcga_pred_labels,
        "tcga_balanced_acc": results_tcga["balanced_acc_valid"],
        "aggc_balanced_acc": results_aggc["balanced_acc_valid"],
        "results_tcga": results_tcga,
        "results_aggc": results_aggc,
    }


def run_experiment(combine_2048, seeds):
    aggc_feats, tcga_feats = prepare_features(combine_2048)

    labels = AGGC_tnm["targets"]
    df_tcga_index_tiles = pd.read_csv(TCGA_f_index)
    df_aggc_index_tiles = pd.read_csv(AGGC_f_index)

    truth_tcga_df = pd.read_csv("Processed_Features/TCGA_PRAD/final_data_2025NOV.csv")
    truth_tcga_df = truth_tcga_df[truth_tcga_df["have_valid_geojson"] == 0]
    truth_aggc_df = pd.read_csv(AGGC_meta_path)

    results = []
    for seed in seeds:
        seed_result = train_one_seed(
            seed,
            aggc_feats,
            tcga_feats,
            labels,
            df_tcga_index_tiles,
            truth_tcga_df,
            df_aggc_index_tiles,
            truth_aggc_df,
        )
        results.append(seed_result)

        if SAVE:
            combine_tag = "combine_2048" if combine_2048 else "default"
            seed_dir = os.path.join(save_path, combine_tag)
            os.makedirs(seed_dir, exist_ok=True)

            np.save(os.path.join(seed_dir, f"aggc_oof_tile_labels_seed{seed}.npy"),
                    seed_result["oof_pred_labels"])
            np.save(os.path.join(seed_dir, f"aggc_oof_tile_proba_seed{seed}.npy"),
                    seed_result["oof_proba"])
            np.save(os.path.join(seed_dir, f"tcga_proba_mean_seed{seed}.npy"),
                    seed_result["tcga_proba_mean"])
            joblib.dump(seed_result["models"],
                        os.path.join(seed_dir, f"models_seed{seed}.pkl"))

            save_json_metrics(seed_result["results_aggc"],
                              os.path.join(seed_dir, f"metrics_aggc_seed{seed}.json"))
            save_json_metrics(seed_result["results_tcga"],
                              os.path.join(seed_dir, f"metrics_tcga_seed{seed}.json"))

    if SAVE:
        combine_tag = "combine_2048" if combine_2048 else "default"
        seed_dir = os.path.join(save_path, combine_tag)
        os.makedirs(seed_dir, exist_ok=True)

        summary_rows = [{
            "seed": r["seed"],
            "tcga_balanced_acc": r["tcga_balanced_acc"],
            "aggc_balanced_acc": r["aggc_balanced_acc"],
        } for r in results]
        pd.DataFrame(summary_rows).to_csv(os.path.join(seed_dir, "summary_metrics.csv"), index=False)

    return results


seeds = list(range(30))

results_default = run_experiment(False, seeds)
results_combine = run_experiment(True, seeds)

tcga_bal_acc_default = [r["tcga_balanced_acc"] for r in results_default]
tcga_bal_acc_combine = [r["tcga_balanced_acc"] for r in results_combine]

aggc_bal_acc_default = [r["aggc_balanced_acc"] for r in results_default]
aggc_bal_acc_combine = [r["aggc_balanced_acc"] for r in results_combine]

tcga_ttest = stats.ttest_ind(tcga_bal_acc_combine, tcga_bal_acc_default, equal_var=False)
aggc_ttest = stats.ttest_ind(aggc_bal_acc_combine, aggc_bal_acc_default, equal_var=False)

print("TCGA balanced accuracy (default):", tcga_bal_acc_default)
print("TCGA balanced accuracy (combine):", tcga_bal_acc_combine)
print("TCGA t-test:", tcga_ttest)

print("AGGC balanced accuracy (default):", aggc_bal_acc_default)
print("AGGC balanced accuracy (combine):", aggc_bal_acc_combine)
print("AGGC t-test:", aggc_ttest)

# %%
# path_t = os.path.join(save_path, "metrics_aggc.json")

# with open(path_t, 'r') as f:
#     data = json.load(f)

# print(json.dumps(data))
