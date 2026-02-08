"""Tile-level and slide-level evaluation metric computation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def normalize_probs(raw):
    """Clip negatives and normalize rows to sum to 1.

    All-zero rows receive uniform probability.
    """
    raw = raw.astype(np.float64)
    raw = np.clip(raw, 0, None)

    row_sums = raw.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze() != 0

    raw[nonzero] /= row_sums[nonzero]
    if (~nonzero).any():
        raw[~nonzero] = 1.0 / raw.shape[1]

    return raw


def compute_tile_metrics(y_true, y_pred, y_proba):
    """Compute OOF tile-level classification metrics.

    Parameters
    ----------
    y_true : np.ndarray — ground truth labels.
    y_pred : np.ndarray — predicted labels.
    y_proba : np.ndarray — predicted probabilities ``(n_samples, n_classes)``.

    Returns
    -------
    dict with keys: ``acc``, ``bacc``, ``auc_macro``, ``auc_each``,
    ``f1_macro``, ``f1_weighted``.
    """
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    try:
        auc_macro = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
        auc_each = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average=None
        )
    except ValueError:
        auc_macro = float("nan")
        auc_each = None

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    return {
        "acc": acc,
        "bacc": bacc,
        "auc_macro": auc_macro,
        "auc_each": auc_each,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def evaluate_slide_predictions(
    out_df,
    truth_df,
    pred_col="ISUP_grade_group",
    true_col="ISUP_grade_group",
    treat_nan_as_class=5,
    verbose=True,
):
    """Compute slide-level evaluation metrics.

    Parameters
    ----------
    out_df : pd.DataFrame
        Predicted slide-level DataFrame (from ``agg_from_tiles``).
    truth_df : pd.DataFrame
        Ground truth DataFrame.
    pred_col, true_col : str
        Column names for predicted / true ISUP grade group.
    treat_nan_as_class : int
        Class label assigned to NaN predictions for metric computation.
    verbose : bool
        Print a summary to stdout.

    Returns
    -------
    dict with keys: ``nan_count``, ``acc_all``, ``confusion_matrix_all``,
    ``acc_valid``, ``balanced_acc_valid``, ``classification_report``.
    """
    pred_raw = out_df[pred_col].values
    true_raw = truth_df[true_col].values

    assert len(pred_raw) == len(true_raw), (
        f"Length mismatch: predictions={len(pred_raw)}, truth={len(true_raw)}"
    )

    nan_mask = np.isnan(pred_raw)

    pred_all = pred_raw.copy()
    pred_all[nan_mask] = treat_nan_as_class
    pred_all = pred_all.astype(int)
    true_all = true_raw.astype(int)

    acc_all = accuracy_score(true_all, pred_all)
    labels_all = sorted(set(true_all.tolist()) | {treat_nan_as_class})

    mask_valid = ~nan_mask
    pred_valid = pred_all[mask_valid]
    true_valid = true_all[mask_valid]

    acc_valid = accuracy_score(true_valid, pred_valid)
    cm = confusion_matrix(true_valid, pred_valid, labels=labels_all)
    bal_acc_valid = balanced_accuracy_score(true_valid, pred_valid)

    clf_report = classification_report(
        true_valid,
        pred_valid,
        labels=sorted(set(true_valid.tolist())),
        output_dict=False,
    )

    if verbose:
        print("=== Slide-level Evaluation ===")
        print(f"Slides with NaN predictions: {nan_mask.sum()} / {len(pred_raw)}\n")
        print(f"--- Including NaNs (as class {treat_nan_as_class}) ---")
        print("Accuracy:", acc_all)
        print("--- Excluding NaNs ---")
        print("Accuracy:", acc_valid)
        print("Balanced accuracy:", bal_acc_valid)
        print("Confusion matrix:\n", cm, "\n")
        print("Classification report:\n", clf_report)

    return {
        "nan_count": int(nan_mask.sum()),
        "acc_all": acc_all,
        "confusion_matrix_all": cm,
        "acc_valid": acc_valid,
        "balanced_acc_valid": bal_acc_valid,
        "classification_report": clf_report,
    }
