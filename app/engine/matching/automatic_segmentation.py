"""Automatic subject-mask generation with an optional ONNX background-removal model.

The model is deliberately optional: a missing model never blocks image analysis.
GrabCut remains an explicitly labelled fallback and is never presented as semantic
segmentation. Generated masks are cached separately from immutable source images.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _mask_quality(mask: np.ndarray) -> dict:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    fraction = float(binary.mean())
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)),
        reverse=True,
    )
    largest_fraction = components[0] / max(1, int(binary.sum())) if components else 0.0
    return {
        "foreground_fraction": round(fraction, 5),
        "component_count": len(components),
        "largest_component_fraction": round(largest_fraction, 5),
        "plausible": 0.005 <= fraction <= 0.98 and bool(components),
    }


def _load_rgb(path: Path, max_dimension: int) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        original_size = oriented.size
        oriented.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        return np.asarray(oriented), original_size


@lru_cache(maxsize=1)
def _u2net_session():
    # Reuse model weights/session across every image in the same worker.
    from rembg import new_session
    return new_session("u2net", providers=["CPUExecutionProvider"])


def _rembg_mask(rgb: np.ndarray) -> np.ndarray:
    # Lazy import keeps the base CPU feature-analysis installation lightweight.
    from rembg import remove

    result = remove(Image.fromarray(rgb), session=_u2net_session(), only_mask=True)
    if isinstance(result, Image.Image):
        alpha = np.asarray(result.convert("L"), dtype=np.uint8)
    else:
        alpha = np.asarray(result, dtype=np.uint8)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("segmentation model returned a mask with unexpected dimensions")
    return alpha


def _grabcut_fallback(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    mx, my = max(1, int(w * 0.035)), max(1, int(h * 0.035))
    mask[:my, :] = cv2.GC_BGD
    mask[-my:, :] = cv2.GC_BGD
    mask[:, :mx] = cv2.GC_BGD
    mask[:, -mx:] = cv2.GC_BGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None,
        bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK,
    )
    return np.where(
        np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)), 255, 0
    ).astype(np.uint8)


def generate_subject_mask(
    image_path: Path,
    *,
    max_dimension: int = 1024,
    allow_heuristic_fallback: bool = True,
) -> dict:
    """Return a full-resolution white-foreground mask and honest diagnostics.

    The preferred model is U2-Net via rembg/ONNX Runtime CPU. If it is not
    installed or fails, GrabCut may be returned as a low-confidence heuristic.
    """
    if max_dimension < 128:
        raise ValueError("max_dimension must be >= 128")
    rgb, original_size = _load_rgb(image_path, max_dimension)
    h, w = rgb.shape[:2]
    if min(h, w) < 32:
        return {
            "status": "unavailable", "reason": "image_too_small",
            "mask": None, "width": original_size[0], "height": original_size[1],
        }

    model_error = None
    try:
        small_mask = _rembg_mask(rgb)
        method = "u2net_rembg_onnx_cpu"
        status = "automatic"
    except Exception as exc:  # optional model and weights may not be installed
        model_error = f"{type(exc).__name__}: {exc}"[:240]
        if not allow_heuristic_fallback:
            return {
                "status": "unavailable", "reason": "automatic_model_unavailable",
                "model_error": model_error, "mask": None,
                "width": original_size[0], "height": original_size[1],
            }
        try:
            small_mask = _grabcut_fallback(rgb)
        except cv2.error:
            return {
                "status": "unavailable", "reason": "segmentation_and_fallback_failed",
                "model_error": model_error, "mask": None,
                "width": original_size[0], "height": original_size[1],
            }
        method = "grabcut_heuristic_fallback"
        status = "low_confidence"

    quality = _mask_quality(small_mask)
    if not quality["plausible"]:
        status = "low_confidence"
    full_mask = cv2.resize(
        small_mask, original_size, interpolation=cv2.INTER_LINEAR
    )
    full_mask = np.where(full_mask >= 128, 255, 0).astype(np.uint8)
    return {
        "status": status,
        "method": method,
        "reason": (
            "model_generated_subject_mask_review_quality_before_object_matching"
            if method == "u2net_rembg_onnx_cpu"
            else "heuristic_fallback_not_semantic_subject_detection"
        ),
        "width": original_size[0],
        "height": original_size[1],
        "mask_width": int(w),
        "mask_height": int(h),
        "mask": full_mask,
        **quality,
        "model_error": model_error,
    }


def save_generated_mask(mask: np.ndarray, target: Path) -> None:
    """Atomically persist a generated mask separately from source photographs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.", suffix=".tmp.png",
            dir=target.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(
            temporary_path, format="PNG", optimize=True
        )
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
