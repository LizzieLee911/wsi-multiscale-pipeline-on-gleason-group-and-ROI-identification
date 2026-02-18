"""Gleason grading and ISUP grade group computation from tile predictions."""

import numpy as np
import pandas as pd


PATTERN_MAP = {0: 3, 1: 4, 2: 5}


def gleason_to_isup(g1, g2):
    """Map primary + secondary Gleason pattern to ISUP grade group (0–4).

    Parameters
    ----------
    g1, g2 : int
        Primary and secondary Gleason patterns (3, 4, or 5).

    Returns
    -------
    int — ISUP grade group (0=GG1 … 4=GG5).
    """
    score = g1 + g2
    if score <= 6:
        return 0       # ISUP 1
    elif score == 7:
        if g1 == 3 and g2 == 4:
            return 1   # ISUP 2
        elif g1 == 4 and g2 == 3:
            return 2   # ISUP 3
        return 1       # fallback for 7 (shouldn't occur with valid patterns)
    elif score == 8:
        return 3       # ISUP 4
    else:
        return 4       # ISUP 5


def agg_from_tiles(df_index, tile_predictions, pattern_map=None,
                   primary_thresh=0.95, secondary_min=0.05,
                   agg_input="proba"):
    """Aggregate tile-level class predictions to slide-level Gleason/ISUP.

    Parameters
    ----------
    df_index : pd.DataFrame
        Index DataFrame with columns ``slide_id``, ``start``, ``length``.
    tile_predictions : np.ndarray
        Tile-level model outputs. Supports either:
        - class labels with shape ``(n_tiles,)`` (0=G3, 1=G4, 2=G5, 3=Other)
        - class probabilities with shape ``(n_tiles, n_classes)``.
    pattern_map : dict, optional
        Mapping from class index to Gleason pattern. Defaults to
        ``{0: 3, 1: 4, 2: 5}``.
    primary_thresh : float
        If the primary pattern fraction ≥ this, treat as single-pattern.
    secondary_min : float
        If the second pattern fraction < this, treat as single-pattern.
    agg_input : str
        Aggregation input type. ``"labels"`` uses tile counts per class;
        ``"proba"`` sums tile probabilities per class before ranking.
        For backward compatibility, if ``agg_input='proba'`` but an array of
        hard labels with shape ``(n_tiles,)`` is provided, labels-based
        aggregation is used automatically.

    Returns
    -------
    pd.DataFrame with columns ``slide_id``, ``p3``, ``p4``, ``p5``,
    ``primary_pattern``, ``secondary_pattern``, ``gleason``,
    ``ISUP_grade_group``.
    """
    if pattern_map is None:
        pattern_map = PATTERN_MAP

    if agg_input not in {"labels", "proba"}:
        raise ValueError("agg_input must be either 'labels' or 'proba'.")

    tile_predictions = np.asarray(tile_predictions)

    out_rows = []
    for _, row in df_index.iterrows():
        slide_id = row["slide_id"]
        start = row["start"]
        end = start + row["length"]

        tiles_preds = tile_predictions[start:end]

        use_labels = agg_input == "labels" or tiles_preds.ndim == 1

        if use_labels:
            counts = np.bincount(tiles_preds, minlength=4)
            tumor_scores = counts[:3].astype(float)
        else:
            if tiles_preds.ndim != 2 or tiles_preds.shape[1] < 3:
                raise ValueError(
                    "For agg_input='proba', tile_predictions must have "
                    "shape (n_tiles, n_classes>=3)."
                )
            tumor_scores = tiles_preds[:, :3].sum(axis=0)

        total_tumor = tumor_scores.sum()

        if total_tumor == 0:
            out_rows.append({
                "slide_id": slide_id,
                "p3": 0.0, "p4": 0.0, "p5": 0.0,
                "primary_pattern": None,
                "secondary_pattern": None,
                "gleason": None,
                "ISUP_grade_group": None,
            })
            continue

        p3, p4, p5 = tumor_scores / total_tumor
        fractions = np.array([p3, p4, p5])

        order = np.argsort(-fractions)
        p1_idx, p2_idx = order[0], order[1]
        p1_frac, p2_frac = fractions[p1_idx], fractions[p2_idx]

        if p1_frac >= primary_thresh or p2_frac < secondary_min:
            g1 = g2 = pattern_map[p1_idx]
        else:
            g1 = pattern_map[p1_idx]
            g2 = pattern_map[p2_idx]

        out_rows.append({
            "slide_id": slide_id,
            "p3": float(p3),
            "p4": float(p4),
            "p5": float(p5),
            "primary_pattern": g1,
            "secondary_pattern": g2,
            "gleason": f"{g1}+{g2}",
            "ISUP_grade_group": gleason_to_isup(g1, g2),
        })

    return pd.DataFrame(out_rows)
