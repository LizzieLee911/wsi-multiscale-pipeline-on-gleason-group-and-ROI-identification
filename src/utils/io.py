"""File I/O utilities: JSON serialization, save‑path construction, artifact saving."""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def save_json_metrics(metrics, filename):
    """Serialize *metrics* (with timestamp) to a JSON file."""
    metrics = to_serializable(metrics)
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def build_save_path(base_dir, model_name, l2_norm, combine_2048):
    """Construct and create the experiment output directory.

    Returns a ``Path`` like ``<base_dir>/XGB_l2_with2048/``.
    """
    l2_str = "l2" if l2_norm else "nol2"
    comb_str = "with2048" if combine_2048 else "tileonly"

    run_name = f"{model_name}_{l2_str}_{comb_str}"
    save_path = Path(base_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def save_experiment_artifacts(
    save_path,
    models,
    oof_proba,
    oof_pred_labels,
    tcga_proba_mean,
    out_tcga,
    out_aggc,
    results_tcga,
    results_aggc,
):
    """Persist all experiment outputs to *save_path*.

    Saves:
      - ``5fold.pkl``           – list of fold models (joblib)
      - ``tcga.csv``            – slide‑level TCGA predictions
      - ``pred_aggc_oof.csv``   – slide‑level AGGC OOF predictions
      - ``tcga_proba_mean.npy`` – tile‑level TCGA mean probabilities
      - ``aggc_oof_tile_labels.npy`` / ``aggc_oof_tile_proba.npy``
      - ``metrics_tcga.json``   / ``metrics_aggc.json``
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, save_path / "5fold.pkl")

    out_tcga.to_csv(save_path / "tcga.csv", index=False)
    out_aggc.to_csv(save_path / "pred_aggc_oof.csv", index=False)

    np.save(save_path / "tcga_proba_mean.npy", tcga_proba_mean)
    np.save(save_path / "aggc_oof_tile_labels.npy", oof_pred_labels)
    np.save(save_path / "aggc_oof_tile_proba.npy", oof_proba)

    save_json_metrics(results_tcga, save_path / "metrics_tcga.json")
    save_json_metrics(results_aggc, save_path / "metrics_aggc.json")
