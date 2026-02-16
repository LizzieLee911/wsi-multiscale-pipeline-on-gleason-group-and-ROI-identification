"""
WSI thumbnail generator (SVS / TIFF) that:
- Reads ONLY the base/original level (Level 0) content.
- Resizes by a user-specified scale factor, preserving aspect ratio.
- Avoids loading the full-resolution image into RAM when possible.
- Returns an RGB numpy array for notebook visualization.
- Optionally saves as JPG.

Recommended deps (any subset works; it will fall back):
  pip install openslide-python pillow numpy
  (optional but GREAT for huge single-layer TIFF) pip install pyvips
Notes:
- SVS usually works best with OpenSlide.
- Huge single-layer TIFF without pyramid often works best with pyvips.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

# Optional deps
try:
    import openslide  # type: ignore
    _HAS_OPENSLIDE = True
except Exception:
    openslide = None
    _HAS_OPENSLIDE = False

try:
    import pyvips  # type: ignore
    _HAS_PYVIPS = True
except Exception:
    pyvips = None
    _HAS_PYVIPS = False


def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert arrays to RGB uint8 safely."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # If has alpha, drop it
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Convert dtype -> uint8
    if arr.dtype == np.uint8:
        return arr

    # Common: uint16
    if arr.dtype == np.uint16:
        arr = (arr / 257).astype(np.uint8)  # 65535/255≈257
        return arr

    # Floats or others: clip 0..255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_jpg(rgb: np.ndarray, out_path: Union[str, Path], quality: int = 92) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path, format="JPEG", quality=int(quality), subsampling=2, optimize=True)


def _thumb_with_openslide(path: Union[str, Path], scale: float) -> np.ndarray:
    if not _HAS_OPENSLIDE:
        raise RuntimeError("openslide-python not available")

    slide = openslide.OpenSlide(str(path))
    w0, h0 = slide.dimensions  # Level 0 size

    if scale <= 0 or scale > 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    w = max(1, int(round(w0 * scale)))
    h = max(1, int(round(h0 * scale)))

    # OpenSlide will pull data efficiently from Level 0 / pyramid without loading full image.
    thumb = slide.get_thumbnail((w, h))  # PIL Image (RGB)
    rgb = np.array(thumb.convert("RGB"), dtype=np.uint8)
    slide.close()
    return rgb


def _thumb_with_pyvips(path: Union[str, Path], scale: float) -> np.ndarray:
    """
    Very memory-friendly for huge TIFFs (even single-layer) and often works for SVS too.
    Uses shrink-on-load when possible.
    """
    if not _HAS_PYVIPS:
        raise RuntimeError("pyvips not available")

    if scale <= 0 or scale > 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    # shrink is integer; choose a safe integer shrink close to 1/scale
    # Example: scale=0.05 => 1/scale=20 => shrink=20
    shrink = int(round(1.0 / scale))
    shrink = max(1, shrink)

    # sequential access avoids random seeks; good for huge images
    img = pyvips.Image.new_from_file(str(path), access="sequential", shrink=shrink)

    # If shrink rounded too much, refine with exact scaling to hit requested scale
    # Effective scale after shrink:
    eff_scale = 1.0 / shrink
    refine = scale / eff_scale  # in (0, ~2)
    if abs(refine - 1.0) > 1e-3:
        img = img.resize(refine)

    # Ensure 3 bands RGB
    if img.bands == 1:
        img = img.bandjoin([img, img])  # -> 3 bands
    elif img.bands >= 3:
        img = img.extract_band(0, n=3)

    # Convert to uchar
    if img.format != "uchar":
        # If 16-bit, scale down; vips will clamp if cast directly.
        # A simple safe approach: linear scale based on max for 16-bit.
        if img.format in ("ushort", "uint"):
            img = (img / 257).cast("uchar")
        else:
            img = img.cast("uchar")

    mem = img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(img.height, img.width, img.bands)
    return arr


def make_wsi_thumbnail(
    path: Union[str, Path],
    scale: float = 0.02,
    save: bool = False,
    out_jpg: Optional[Union[str, Path]] = None,
    prefer: str = "auto",
    jpg_quality: int = 92,
) -> np.ndarray:
    """
    Generate a thumbnail from the ORIGINAL/base image (Level 0 content), resized by `scale`.

    Parameters
    ----------
    path : str | Path
        Path to .svs or .tif/.tiff.
    scale : float
        Scale factor in (0, 1]. Keeps aspect ratio automatically.
        Example: 0.02 => 2% of width/height.
    save : bool
        If True, saves JPG.
    out_jpg : str | Path | None
        Output JPG path if save=True. If None, saves next to input with suffix.
    prefer : {"auto","openslide","pyvips"}
        Backend preference.
        - "auto": try openslide (great for SVS), then pyvips (great for huge TIFF), then error.
        - "openslide": force openslide
        - "pyvips": force pyvips
    jpg_quality : int
        JPEG quality 1-100.

    Returns
    -------
    rgb : np.ndarray
        RGB uint8 array of shape (H, W, 3) for notebook visualization.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    prefer = prefer.lower().strip()
    if prefer not in {"auto", "openslide", "pyvips"}:
        raise ValueError("prefer must be one of: auto, openslide, pyvips")

    last_err = None
    rgb: Optional[np.ndarray] = None

    backends = []
    if prefer == "openslide":
        backends = ["openslide"]
    elif prefer == "pyvips":
        backends = ["pyvips"]
    else:
        # auto: OpenSlide first (especially for SVS), then pyvips
        backends = ["openslide", "pyvips"]

    for be in backends:
        try:
            if be == "openslide":
                rgb = _thumb_with_openslide(path, scale)
            elif be == "pyvips":
                rgb = _thumb_with_pyvips(path, scale)
            else:
                raise RuntimeError("Unknown backend")
            break
        except Exception as e:
            last_err = e
            rgb = None

    if rgb is None:
        msg = f"Failed to generate thumbnail for {path}.\nLast error: {last_err}"
        msg += "\nTip: install pyvips for huge single-layer TIFF: pip install pyvips"
        raise RuntimeError(msg)

    rgb = _ensure_rgb_uint8(rgb)

    if save:
        if out_jpg is None:
            out_jpg = path.with_suffix("")  # remove .svs/.tif
            out_jpg = str(out_jpg) + f"_thumb_s{scale:.4f}.jpg"
        _save_jpg(rgb, out_jpg, quality=jpg_quality)

    return rgb


# ---------------------------
# Example usage (Notebook):
# ---------------------------
# rgb = make_wsi_thumbnail("slide.svs", scale=0.02, save=False)
# from matplotlib import pyplot as plt
# plt.figure()
# plt.imshow(rgb)
# plt.axis("off")

# ---------------------------
# Example usage (Script/CLI):
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate WSI thumbnail from SVS/TIFF Level 0.")
    parser.add_argument("path", type=str, help="Path to .svs / .tif / .tiff")
    parser.add_argument("--scale", type=float, default=0.02, help="Scale factor in (0,1], default 0.02")
    parser.add_argument("--save", action="store_true", help="Save JPG")
    parser.add_argument("--out", type=str, default=None, help="Output JPG path")
    parser.add_argument("--prefer", type=str, default="auto", choices=["auto", "openslide", "pyvips"])
    parser.add_argument("--quality", type=int, default=92, help="JPG quality 1-100")
    args = parser.parse_args()

    rgb = make_wsi_thumbnail(
        args.path,
        scale=args.scale,
        save=args.save,
        out_jpg=args.out,
        prefer=args.prefer,
        jpg_quality=args.quality,
    )
    print(f"Thumbnail generated: shape={rgb.shape}, dtype={rgb.dtype}")
    if args.save:
        print(f"Saved JPG to: {args.out or (str(Path(args.path).with_suffix('')) + f'_thumb_s{args.scale:.4f}.jpg')}")
