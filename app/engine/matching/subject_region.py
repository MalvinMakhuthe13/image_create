from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def estimate_subject_region(image_path: Path, *, max_dimension: int = 800) -> dict:
    """Create a conservative, advisory GrabCut foreground proposal.

    This is not semantic object detection. It may include nearby surfaces or
    omit object parts; callers must expose its heuristic status.
    """
    if max_dimension < 128:
        raise ValueError("max_dimension must be >= 128")

    with Image.open(image_path) as source:
        rgb_image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = rgb_image.size
        rgb_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        rgb = np.asarray(rgb_image)

    h, w = rgb.shape[:2]
    if min(h, w) < 32:
        return {
            "status": "unavailable", "reason": "image_too_small",
            "width": width, "height": height, "mask": None,
        }

    # A narrow border is treated as background; the interior is initialized
    # as probable foreground and GrabCut refines it from colour statistics.
    margin_x = max(1, int(w * 0.035))
    margin_y = max(1, int(h * 0.035))
    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    mask[:margin_y, :] = cv2.GC_BGD
    mask[-margin_y:, :] = cv2.GC_BGD
    mask[:, :margin_x] = cv2.GC_BGD
    mask[:, -margin_x:] = cv2.GC_BGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None,
            bg_model, fg_model, 3, cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return {
            "status": "unavailable", "reason": "grabcut_failed",
            "width": width, "height": height, "mask": None,
        }

    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fraction = float(foreground.mean())
    if fraction < 0.01 or fraction > 0.98:
        return {
            "status": "low_confidence", "reason": "implausible_foreground_fraction",
            "width": width, "height": height, "mask": foreground,
            "foreground_fraction": round(fraction, 4),
        }
    return {
        "status": "heuristic",
        "reason": "grabcut_foreground_proposal_not_semantic_segmentation",
        "width": width, "height": height, "mask": foreground,
        "mask_width": w, "mask_height": h,
        "foreground_fraction": round(fraction, 4),
    }


def classify_keypoints(features, region: dict) -> dict:
    """Classify feature locations against a proposed foreground mask."""
    mask = region.get("mask")
    if mask is None or region.get("status") == "unavailable":
        return {
            "status": "unavailable", "total_keypoints": features.feature_count,
            "foreground_candidate": 0, "background_candidate": 0,
            "foreground_fraction": None, "labels": (),
        }

    mask_h, mask_w = mask.shape[:2]
    labels = []
    for x, y in features.keypoints:
        mx = min(mask_w - 1, max(0, int(x / region["width"] * mask_w)))
        my = min(mask_h - 1, max(0, int(y / region["height"] * mask_h)))
        labels.append(bool(mask[my, mx]))

    foreground = sum(labels)
    total = len(labels)
    return {
        "status": region["status"],
        "total_keypoints": total,
        "foreground_candidate": foreground,
        "background_candidate": total - foreground,
        "foreground_fraction": round(foreground / total, 4) if total else 0.0,
        "labels": tuple(labels),
        "confidence_note": (
            "Subject-region mask; foreground may include nearby surfaces "
            "and mask provenance is reported separately."
        ),
    }


def pair_subject_support(matches, geometry, left_regions: dict, right_regions: dict) -> dict:
    """Describe whether verified correspondences fall in proposed foreground regions."""
    inlier_indices = geometry.inlier_match_indices
    left_labels = left_regions.get("labels", ())
    right_labels = right_regions.get("labels", ())
    if not left_labels or not right_labels:
        return {
            "status": "unavailable",
            "verified_correspondences": len(inlier_indices),
            "both_foreground": None, "both_background": None,
            "mixed_region": None, "foreground_support_fraction": None,
            "warning": "subject_region_unavailable",
        }

    both_fg = both_bg = mixed = 0
    for index in inlier_indices:
        match = matches[index]
        lfg = left_labels[match.query_index]
        rfg = right_labels[match.train_index]
        if lfg and rfg:
            both_fg += 1
        elif not lfg and not rfg:
            both_bg += 1
        else:
            mixed += 1
    total = len(inlier_indices)
    fraction = both_fg / total if total else 0.0
    warning = (
        "most_verified_matches_are_outside_proposed_subject_regions"
        if total >= 8 and (both_bg + mixed) > both_fg else None
    )
    return {
        "status": "heuristic",
        "verified_correspondences": total,
        "both_foreground": both_fg,
        "both_background": both_bg,
        "mixed_region": mixed,
        "foreground_support_fraction": round(fraction, 4) if total else 0.0,
        "warning": warning,
        "confidence_note": (
            "Foreground means inside an automatic GrabCut proposal, not confirmed object pixels. "
            "This score is advisory and does not alter geometric verification or graph connectivity."
        ),
    }


def filter_features_to_region(features, region: dict):
    """Return features whose keypoints lie inside a supplied foreground mask.

    Descriptor rows and keypoint coordinates remain index-aligned. Coordinates
    stay in original-image pixels so geometric diagnostics remain comparable.
    """
    from app.engine.matching.models import ImageFeatures

    mask = region.get("mask")
    if mask is None:
        return ImageFeatures(
            image_index=features.image_index, keypoints=(), descriptors=None,
            image_width=features.image_width, image_height=features.image_height,
            scale_x=features.scale_x, scale_y=features.scale_y,
        )
    mask_h, mask_w = mask.shape[:2]
    keep = []
    for index, (x, y) in enumerate(features.keypoints):
        mx = min(mask_w - 1, max(0, int(x / region["width"] * mask_w)))
        my = min(mask_h - 1, max(0, int(y / region["height"] * mask_h)))
        if mask[my, mx] > 0:
            keep.append(index)
    descriptors = (
        features.descriptors[keep].copy()
        if features.descriptors is not None and keep else None
    )
    return ImageFeatures(
        image_index=features.image_index,
        keypoints=tuple(features.keypoints[index] for index in keep),
        descriptors=descriptors,
        image_width=features.image_width,
        image_height=features.image_height,
        scale_x=features.scale_x,
        scale_y=features.scale_y,
    )
