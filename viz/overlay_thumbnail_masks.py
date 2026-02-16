#!/usr/bin/env python3
"""
Generate WSI thumbnail and overlay ROI masks from a mask folder.

Example:
  python overlay_thumbnail_masks.py \
    --image AGGCSUBSET1_NEW/Group1/Subset1_Train_1.tiff \
    --mask-dir AGGC_Annotation/Annotation_all_file/Subset1_Train_1 \
    --scale 0.02 \
    --out-dir outputs \
    --legend
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import pyvips  # type: ignore
    _HAS_PYVIPS = True
except Exception:
    pyvips = None
    _HAS_PYVIPS = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.return_wsi_thumbnail import make_wsi_thumbnail


MASK_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "g3": (255, 0, 0),       # red
    "g4": (255, 255, 0),     # yellow
    "g5": (0, 0, 255),       # blue
    "normal": (0, 200, 0),   # green
    "stroma": (160, 32, 240) # purple
}

MASK_PATTERNS: Dict[str, Iterable[str]] = {
    "g3": ("g3_mask", "g3mask", "g3"),
    "g4": ("g4_mask", "g4mask", "g4"),
    "g5": ("g5_mask", "g5mask", "g5"),
    "normal": ("normal",),
    "stroma": ("stroma",),
}


def _find_mask_files(mask_dir: Path) -> Dict[str, Path]:
    candidates = sorted([p for p in mask_dir.iterdir() if p.is_file()])
    selected: Dict[str, Path] = {}
    for key, patterns in MASK_PATTERNS.items():
        for p in candidates:
            name = p.name.lower()
            if any(pattern in name for pattern in patterns):
                selected[key] = p
                break
    return selected


def _mask_to_bool(mask_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if _HAS_PYVIPS:
        m = pyvips.Image.new_from_file(str(mask_path), access="sequential")
        scale = min(target_w / m.width, target_h / m.height)
        m = m.resize(scale, kernel="nearest")
        if m.width != target_w or m.height != target_h:
            if m.width < target_w or m.height < target_h:
                m = m.embed(0, 0, target_w, target_h, extend="black")
            else:
                m = m.crop(0, 0, target_w, target_h)
        arr = np.ndarray(
            buffer=m.write_to_memory(),
            dtype=np.uint8,
            shape=(m.height, m.width, m.bands),
        )
        mask = arr[..., 0] > 0
        return mask

    img = Image.open(mask_path)
    img = img.convert("L")
    img = img.resize((target_w, target_h), resample=Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    return arr > 0


def _overlay_masks(
    thumbnail: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float,
    edge_thickness: int = 4,
) -> np.ndarray:
    overlay = thumbnail.astype(np.float32).copy()
    for key, mask in masks.items():
        if key not in MASK_COLOR_MAP:
            continue
        color = np.array(MASK_COLOR_MAP[key], dtype=np.float32)
        mask_3 = mask[..., None]
        overlay = np.where(
            mask_3,
            (1 - alpha) * overlay + alpha * color,
            overlay,
        )
        # draw an opaque outline around mask with configurable thickness
        if edge_thickness and edge_thickness > 0:
            eroded = mask.copy()
            for _ in range(edge_thickness):
                padded = np.pad(eroded, pad_width=1, mode="constant", constant_values=False)
                center = padded[1:-1, 1:-1]
                up = padded[:-2, 1:-1]
                down = padded[2:, 1:-1]
                left = padded[1:-1, :-2]
                right = padded[1:-1, 2:]
                eroded = center & up & down & left & right
            edge = mask & (~eroded)
            edge_3 = edge[..., None]
            overlay = np.where(edge_3, color, overlay)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _save_overlay_with_legend(overlay: np.ndarray,present_keys: List[str],output_path: Path,) -> None:
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")

    patches = [
        Patch(facecolor=np.array(MASK_COLOR_MAP[k]) / 255.0, label=k.upper())
        for k in present_keys
        if k in MASK_COLOR_MAP
    ]
    if patches:
        ax.legend(handles=patches, loc="lower right", frameon=True, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def run(
    image_path: Path,
    mask_dir: Path,
    scale: float,
    out_dir: Path,
    alpha: float,
    prefer_backend: str,
    with_legend: bool,
    edge_thickness: int = 4,
) -> Tuple[Path, Path]:
    thumbnail = make_wsi_thumbnail(image_path, scale=scale, prefer=prefer_backend)
    thumb_path = out_dir / f"{image_path.stem}_thumb_s{scale:.4f}.jpg"
    _save_image(thumbnail, thumb_path)

    mask_files = _find_mask_files(mask_dir)
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    h, w = thumbnail.shape[:2]
    masks = {k: _mask_to_bool(p, (h, w)) for k, p in mask_files.items()}
    overlay = _overlay_masks(thumbnail, masks, alpha=alpha, edge_thickness=edge_thickness)

    overlay_path = out_dir / f"{image_path.stem}_thumb_s{scale:.4f}_overlay.jpg"
    if with_legend:
        present = list(masks.keys())
        _save_overlay_with_legend(overlay, present, overlay_path)
    else:
        _save_image(overlay, overlay_path)

    return thumb_path, overlay_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thumbnail and overlay ROI masks with color codes."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to WSI image")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Mask folder path")
    parser.add_argument("--scale", type=float, default=0.02, help="Thumbnail scale")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay alpha")
    parser.add_argument(
        "--edge-thickness",
        type=int,
        default=4,
        help="Outline thickness in pixels (integer, 0 disables outline)",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="auto",
        choices=["auto", "openslide", "pyvips"],
        help="Thumbnail backend preference",
    )
    parser.add_argument("--legend", action="store_true", help="Include legend in overlay")
    args = parser.parse_args()

    thumb_path, overlay_path = run(
        image_path=args.image,
        mask_dir=args.mask_dir,
        scale=args.scale,
        out_dir=args.out_dir,
        alpha=args.alpha,
        prefer_backend=args.prefer,
        with_legend=args.legend,
        edge_thickness=args.edge_thickness,
    )
    print(f"Thumbnail saved to: {thumb_path}")
    print(f"Overlay saved to: {overlay_path}")


if __name__ == "__main__":
    main()
