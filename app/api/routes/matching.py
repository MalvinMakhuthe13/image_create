from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image, ImageOps
from itertools import combinations
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.engine.matching.capture_advisor import recommend_capture_improvements
from app.engine.matching.sparse_reconstruction import reconstruct_sparse_scene
from app.engine.matching import (
    build_connectivity_graph,
    extract_features,
    match_descriptors,
    verify_geometry,
)
from app.engine.matching.subject_region import (
    classify_keypoints,
    pair_subject_support,
)
from app.engine.matching.automatic_segmentation import (
    generate_subject_mask,
    save_generated_mask,
)
from app.engine.matching.mask_boundary import (
    summarize_mask_boundary_distances,
)

router = APIRouter(prefix="/analysis", tags=["geometric matching"])
STORED_IMAGE_PATTERN = re.compile(
    r"^\d{3}_[0-9a-f]{12}\.(jpg|png|webp|tif|bmp)$", re.IGNORECASE
)
MAX_DIAGNOSTIC_OUTLIERS = 100


def _load_manual_mask(analysis_id: UUID, image_index: int, image_path: Path):
    mask_path = Path(settings.upload_dir).parent / "masks" / str(analysis_id) / f"{image_index:03d}.png"
    if not mask_path.is_file():
        return None
    try:
        with Image.open(mask_path) as source:
            source.load()
            mask = np.asarray(source.convert("L"), dtype=np.uint8) >= 128
        with Image.open(image_path) as source:
            width, height = ImageOps.exif_transpose(source).size
    except (OSError, ValueError):
        return None
    return mask if mask.shape == (height, width) else None


def _point_inside_manual_mask(image, keypoint_index, mask) -> bool:
    """Test an original-image keypoint against a same-sized binary manual mask."""
    if keypoint_index < 0 or keypoint_index >= len(image.keypoints):
        return False
    x, y = image.keypoints[keypoint_index]
    px, py = int(x), int(y)
    return (
        0 <= py < mask.shape[0]
        and 0 <= px < mask.shape[1]
        and bool(mask[py, px])
    )


def _inside_both_manual_masks(left, right, match, left_mask, right_mask):
    return (
        _point_inside_manual_mask(left, match.query_index, left_mask)
        and _point_inside_manual_mask(right, match.train_index, right_mask)
    )


def _mask_filter_diagnostics(left, right, matches, left_mask, right_mask) -> dict:
    """Explain how each manual mask independently filters descriptor matches."""
    left_inside = 0
    right_inside = 0
    both_inside = 0
    left_only = 0
    right_only = 0
    neither = 0

    for match in matches:
        in_left = _point_inside_manual_mask(left, match.query_index, left_mask)
        in_right = _point_inside_manual_mask(right, match.train_index, right_mask)
        left_inside += int(in_left)
        right_inside += int(in_right)
        both_inside += int(in_left and in_right)
        left_only += int(in_left and not in_right)
        right_only += int(not in_left and in_right)
        neither += int(not in_left and not in_right)

    total = len(matches)
    return {
        "status": "complete",
        "total_descriptor_matches": total,
        "eligible_by_left_mask": left_inside,
        "eligible_by_right_mask": right_inside,
        "eligible_by_both_masks": both_inside,
        "left_inside_right_outside": left_only,
        "left_outside_right_inside": right_only,
        "outside_both_masks": neither,
        "excluded_by_left_mask": total - left_inside,
        "excluded_by_right_mask": total - right_inside,
        "excluded_by_either_mask": total - both_inside,
        "left_mask_foreground_fraction": round(float(left_mask.mean()), 4),
        "right_mask_foreground_fraction": round(float(right_mask.mean()), 4),
        "interpretation": (
            "Counts are descriptor correspondences whose keypoint centres fall inside "
            "each original-image manual mask. They are diagnostic counts, not geometric "
            "verification or proof of 3D surface coverage."
        ),
    }


def _candidate_pairs(image_count: int, maximum: int) -> list[tuple[int, int]]:
    """Prioritise neighbouring capture frames, then sample remaining pairs evenly."""
    if image_count < 2:
        return []
    minimum_chain = image_count - 1
    if maximum < minimum_chain:
        raise ValueError(
            f"max_matching_pair_checks must be at least {minimum_chain} "
            "to preserve adjacent-frame checks"
        )

    selected = [(index, index + 1) for index in range(image_count - 1)]
    if len(selected) >= maximum:
        return selected[:maximum]

    selected_set = set(selected)
    remaining = [
        pair for pair in combinations(range(image_count), 2)
        if pair not in selected_set
    ]
    slots = maximum - len(selected)
    if len(remaining) <= slots:
        selected.extend(remaining)
    elif slots:
        last = len(remaining) - 1
        chosen_indices = sorted({
            round(i * last / max(1, slots - 1)) for i in range(slots)
        })
        selected.extend(remaining[index] for index in chosen_indices)
    return selected


def _pair_diagnostics(left, right, matches, result) -> list[dict]:
    """Return original-image pixel coordinates for inliers and a bounded outlier sample."""
    inlier_indices = set(result.inlier_match_indices)
    inlier_rows = sorted(inlier_indices)
    outlier_rows = [index for index in range(len(matches)) if index not in inlier_indices]
    outlier_rows = sorted(outlier_rows, key=lambda index: matches[index].distance)
    selected = inlier_rows + outlier_rows[:MAX_DIAGNOSTIC_OUTLIERS]

    diagnostics = []
    for index in selected:
        match = matches[index]
        left_x, left_y = left.keypoints[match.query_index]
        right_x, right_y = right.keypoints[match.train_index]
        diagnostics.append({
            "match_index": index,
            "left_keypoint_index": match.query_index,
            "right_keypoint_index": match.train_index,
            "left": {"x": round(left_x, 2), "y": round(left_y, 2)},
            "right": {"x": round(right_x, 2), "y": round(right_y, 2)},
            "descriptor_distance": round(match.distance, 3),
            "is_inlier": index in inlier_indices,
            "match_status": (
                "inlier" if index in inlier_indices
                else "outlier" if result.model != "none"
                else "unverified"
            ),
        })
    return diagnostics


GRID_SIZE = 4
NEAR_DUPLICATE_BUCKET_PX = 5.0
MIN_INLIERS_FOR_SPATIAL_WARNING = 8



SPATIAL_DEDUPLICATION_RADIUS_PX = 5.0


def _deduplicate_spatial_inliers(left, right, matches, inlier_indices):
    """Keep the best descriptor match for each distinct 2D-to-2D location pair.

    A candidate is a duplicate only when it is within the radius on BOTH
    source images. Four-dimensional spatial hashing limits exact comparisons
    to neighbouring cells while preserving the previous radius test.
    """
    radius = SPATIAL_DEDUPLICATION_RADIUS_PX
    radius_squared = radius * radius
    buckets: dict[tuple[int, int, int, int], list[int]] = {}
    retained: list[int] = []

    # Stable tie-breaking preserves input order when descriptor distances match.
    ordered = sorted(inlier_indices, key=lambda index: (matches[index].distance, index))
    for match_index in ordered:
        match = matches[match_index]
        left_x, left_y = left.keypoints[match.query_index]
        right_x, right_y = right.keypoints[match.train_index]
        coordinates = (left_x, left_y, right_x, right_y)
        cell = tuple(int(value // radius) for value in coordinates)

        duplicate = False
        for dx0 in (-1, 0, 1):
            if duplicate:
                break
            for dy0 in (-1, 0, 1):
                if duplicate:
                    break
                for dx1 in (-1, 0, 1):
                    if duplicate:
                        break
                    for dy1 in (-1, 0, 1):
                        neighbour = (
                            cell[0] + dx0,
                            cell[1] + dy0,
                            cell[2] + dx1,
                            cell[3] + dy1,
                        )
                        for kept_index in buckets.get(neighbour, ()):
                            kept = matches[kept_index]
                            kept_left_x, kept_left_y = left.keypoints[kept.query_index]
                            kept_right_x, kept_right_y = right.keypoints[kept.train_index]
                            left_distance_squared = (
                                (left_x - kept_left_x) ** 2
                                + (left_y - kept_left_y) ** 2
                            )
                            right_distance_squared = (
                                (right_x - kept_right_x) ** 2
                                + (right_y - kept_right_y) ** 2
                            )
                            if (
                                left_distance_squared <= radius_squared
                                and right_distance_squared <= radius_squared
                            ):
                                duplicate = True
                                break
                        if duplicate:
                            break
        if not duplicate:
            retained.append(match_index)
            buckets.setdefault(cell, []).append(match_index)

    return tuple(retained)


def _spatial_support_label(unique_count, spatial_warning):
    """An advisory label; it never changes geometric verification or graph edges."""
    if spatial_warning:
        return "spatially_concentrated"
    if unique_count >= 20:
        return "broad_2d_support"
    if unique_count >= 12:
        return "moderate_2d_support"
    return "limited_2d_support"


def _spatial_image_summary(image, matches, inlier_indices, side: str) -> dict:
    """Summarise where verified 2D correspondences fall within one source image."""
    width, height = image.image_width, image.image_height
    points = []
    if width > 0 and height > 0:
        for match_index in sorted(inlier_indices):
            match = matches[match_index]
            keypoint_index = match.query_index if side == "left" else match.train_index
            x, y = image.keypoints[keypoint_index]
            if 0 <= x < width and 0 <= y < height:
                points.append((float(x), float(y)))

    if not points:
        return {
            "inlier_points": 0, "grid_size": GRID_SIZE,
            "occupied_grid_cells": 0, "grid_cells": [],
            "bounding_box_area_fraction": 0.0,
            "max_grid_cell_fraction": 0.0,
            "near_duplicate_location_groups_5px": 0,
            "spatially_concentrated": False,
            "warning": "insufficient_spatial_data",
        }

    occupied = {}
    buckets = {}
    for x, y in points:
        col = min(GRID_SIZE - 1, int(x / width * GRID_SIZE))
        row = min(GRID_SIZE - 1, int(y / height * GRID_SIZE))
        occupied[(row, col)] = occupied.get((row, col), 0) + 1
        bucket = (
            int(x // NEAR_DUPLICATE_BUCKET_PX),
            int(y // NEAR_DUPLICATE_BUCKET_PX),
        )
        buckets[bucket] = buckets.get(bucket, 0) + 1

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    bbox_fraction = ((max(xs) - min(xs)) / width) * ((max(ys) - min(ys)) / height)
    max_cell_fraction = max(occupied.values()) / len(points)
    concentrated = (
        len(points) >= MIN_INLIERS_FOR_SPATIAL_WARNING
        and (len(occupied) < 3 or bbox_fraction < 0.01 or max_cell_fraction >= 0.75)
    )
    return {
        "inlier_points": len(points),
        "grid_size": GRID_SIZE,
        "occupied_grid_cells": len(occupied),
        "grid_cells": [
            {"row": row, "column": col, "inlier_count": count}
            for (row, col), count in sorted(occupied.items())
        ],
        "bounding_box_area_fraction": round(bbox_fraction, 5),
        "max_grid_cell_fraction": round(max_cell_fraction, 5),
        "near_duplicate_location_groups_5px": sum(
            1 for count in buckets.values() if count > 1
        ),
        "spatially_concentrated": concentrated,
        "warning": (
            "inliers_concentrated_in_small_image_region" if concentrated else None
        ),
    }



def _spatial_distribution(left, right, matches, result) -> dict:
    """Report raw spatial spread and deduplicated correspondence counts."""
    raw_indices = tuple(result.inlier_match_indices)
    unique_indices = _deduplicate_spatial_inliers(
        left, right, matches, raw_indices
    )

    # Keep the concentration test based on the raw RANSAC inliers. If we
    # deduplicate first, a tight cluster of repeated detections can collapse
    # below the minimum sample count and incorrectly lose its warning.
    left_summary = _spatial_image_summary(left, matches, raw_indices, "left")
    right_summary = _spatial_image_summary(right, matches, raw_indices, "right")
    unique_left_summary = _spatial_image_summary(
        left, matches, unique_indices, "left"
    )
    unique_right_summary = _spatial_image_summary(
        right, matches, unique_indices, "right"
    )
    concentrated = (
        left_summary["spatially_concentrated"]
        or right_summary["spatially_concentrated"]
    )
    unique_count = len(unique_indices)
    return {
        "method": "5px_mutual_image_coordinate_deduplication_then_4x4_grid",
        "deduplication_radius_px": SPATIAL_DEDUPLICATION_RADIUS_PX,
        "deduplication_rule": (
            "A correspondence is a duplicate only when both its left-image "
            "and right-image coordinates are within 5px of a retained pair."
        ),
        "raw_inlier_matches": len(raw_indices),
        "unique_spatial_inlier_matches": unique_count,
        "duplicate_spatial_inlier_matches_removed": len(raw_indices) - unique_count,
        "unique_inlier_fraction": round(
            unique_count / len(raw_indices), 4
        ) if raw_indices else 0.0,
        "interpretation": (
            "2D image-space distribution only; this does not establish 3D surface coverage."
        ),
        "left_image": left_summary,
        "right_image": right_summary,
        "unique_location_distribution": {
            "left_image": unique_left_summary,
            "right_image": unique_right_summary,
        },
        "spatial_support": _spatial_support_label(unique_count, concentrated),
        "warning": (
            "inlier_distribution_is_spatially_concentrated"
            if concentrated else None
        ),
        "confidence_note": (
            "Advisory: concentration warnings use raw RANSAC inliers so repeated detections "
            "cannot hide a tight cluster. Unique-location counts are reported separately. "
            "Advisory only; RANSAC verification and connectivity decisions are unchanged."
        ),
    }


@router.post("/{analysis_id}/matching")
def match_uploaded_analysis(analysis_id: UUID) -> dict:
    """Run bounded CPU geometric matching against images from a completed upload."""
    folder = Path(settings.upload_dir) / str(analysis_id)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Analysis upload was not found")

    image_paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and STORED_IMAGE_PATTERN.fullmatch(path.name)
    )
    if not image_paths:
        raise HTTPException(status_code=404, detail="No stored analysis images were found")

    try:
        pairs = _candidate_pairs(len(image_paths), settings.max_matching_pair_checks)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = []
    try:
        for index, path in enumerate(image_paths):
            features.append(
                extract_features(
                    path,
                    index,
                    max_dimension=settings.matching_max_dimension,
                    max_features=settings.matching_max_features,
                )
            )
    except (OSError, ValueError, cv2.error) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract features from stored image: {exc}",
        ) from exc

    # Subject-region support is populated below from the exact accepted mask
    # used for object-only extraction; no separate GrabCut proposal is needed.
    subject_regions = [None] * len(image_paths)
    subject_summaries = [None] * len(image_paths)

    # Manual masks take precedence. Otherwise use a cached/generated U2-Net mask.
    # Heuristic GrabCut proposals remain advisory and are NOT silently promoted
    # to authoritative object masks.
    automatic_mask_dir = Path(settings.upload_dir).parent / "automatic_masks" / str(analysis_id)
    manual_masks = []
    mask_sources = []
    automatic_summaries = []
    for index, image_path in enumerate(image_paths):
        manual_mask = _load_manual_mask(analysis_id, index, image_path)
        if manual_mask is not None:
            manual_masks.append(manual_mask)
            mask_sources.append("manual")
            automatic_summaries.append({
                "image_index": index, "status": "manual_override",
                "method": "user_supplied_mask",
            })
            continue

        cached_path = automatic_mask_dir / f"{index:03d}.png"
        region = None
        if cached_path.is_file():
            try:
                with Image.open(cached_path) as source:
                    source.load()
                    candidate = np.asarray(source.convert("L"), dtype=np.uint8)
                with Image.open(image_path) as source:
                    expected = ImageOps.exif_transpose(source).size
                if candidate.shape == (expected[1], expected[0]):
                    region = {
                        "status": "automatic", "method": "cached_u2net_mask",
                        "mask": candidate,
                        "foreground_fraction": round(float((candidate > 0).mean()), 5),
                        "plausible": 0.005 <= float((candidate > 0).mean()) <= 0.98,
                    }
            except (OSError, ValueError):
                region = None
        if region is None:
            region = generate_subject_mask(image_path)
            if region.get("status") == "automatic" and region.get("mask") is not None:
                save_generated_mask(region["mask"], cached_path)

        accepted = (
            region.get("status") == "automatic"
            and region.get("mask") is not None
            and bool(region.get("plausible", True))
        )
        manual_masks.append(
            (np.asarray(region["mask"], dtype=np.uint8) >= 128)
            if accepted else None
        )
        mask_sources.append("automatic" if accepted else "unavailable")
        automatic_summaries.append({
            key: value for key, value in region.items() if key != "mask"
        } | {"image_index": index, "accepted_for_object_matching": accepted})

    # The correspondence-support diagnostics must describe the same accepted
    # mask that object-only feature extraction actually uses, not the old
    # advisory GrabCut proposal computed earlier.
    for index, mask in enumerate(manual_masks):
        with Image.open(image_paths[index]) as source:
            oriented_size = ImageOps.exif_transpose(source).size
        if mask is None:
            region = {
                "status": "unavailable",
                "reason": "no_accepted_model_mask_or_manual_override",
                "width": oriented_size[0],
                "height": oriented_size[1],
                "mask": None,
            }
        else:
            region = {
                "status": mask_sources[index],
                "reason": "manual_override" if mask_sources[index] == "manual" else "accepted_u2net_mask",
                "width": oriented_size[0],
                "height": oriented_size[1],
                "mask": np.asarray(mask, dtype=np.uint8),
                "foreground_fraction": round(float(mask.mean()), 5),
            }
        classified = classify_keypoints(features[index], region)
        classified["mask"] = region["mask"]
        classified["width"] = region["width"]
        classified["height"] = region["height"]
        subject_regions[index] = classified
        subject_summaries[index] = {
            "image_index": index,
            "filename": image_paths[index].name,
            "segmentation": {
                key: value for key, value in region.items() if key != "mask"
            },
            "keypoint_region_summary": {
                key: value for key, value in classified.items()
                if key not in {"labels", "mask"}
            },
        }

    # Object-only extraction is independent of the all-scene baseline. The
    # detector sees only pixels inside the user's saved mask, with a one-pixel
    # eroded boundary and a higher-recall SIFT/CLAHE profile.
    object_features = []
    for index, image_path in enumerate(image_paths):
        mask = manual_masks[index]
        if mask is None:
            object_features.append(None)
            continue
        try:
            object_features.append(extract_features(
                image_path,
                index,
                max_dimension=settings.matching_max_dimension,
                max_features=max(8000, settings.matching_max_features * 2),
                mask=mask,
                feature_profile="object",
            ))
        except (OSError, ValueError, cv2.error) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract masked object features from image {index}: {exc}",
            ) from exc

    verified_pairs = []
    pair_results = []
    reconstruction_pair_rows = []
    adjacent_pairs_evaluated = 0
    non_adjacent_pairs_evaluated = 0

    for left_index, right_index in pairs:
        left, right = features[left_index], features[right_index]
        try:
            matches = match_descriptors(left, right)
        except cv2.error as exc:
            raise HTTPException(
                status_code=422,
                detail=f"CPU descriptor matching failed for pair {left_index}-{right_index}: {exc}",
            ) from exc
        result = verify_geometry(left, right, matches)
        verified_pairs.append((left_index, right_index, result))
        is_adjacent = right_index == left_index + 1
        adjacent_pairs_evaluated += int(is_adjacent)
        non_adjacent_pairs_evaluated += int(not is_adjacent)

        # Object-only RANSAC receives only all-scene descriptor matches for
        # which BOTH original-image endpoints fall inside their accepted automatic or manual masks.
        left_mask, right_mask = manual_masks[left_index], manual_masks[right_index]
        if left_mask is None or right_mask is None:
            object_only = {
                "status": "missing_automatic_or_manual_mask",
                "verified": False, "model": "none",
                "tentative_matches": 0, "inlier_matches": 0,
                "inlier_ratio": 0.0, "median_reprojection_error_px": None,
                "reason": "accepted_automatic_or_manual_masks_required_for_both_images",
                "inlier_match_indices": [], "diagnostic_correspondences": [],
                "filtered_match_count": 0,
                "mask_sources": [mask_sources[left_index], mask_sources[right_index]],
                "mask_filter_diagnostics": {
                    "status": "unavailable",
                    "reason": "valid_automatic_or_manual_masks_required_for_both_images",
                    "total_descriptor_matches": len(matches),
                },
            }
        else:
            mask_filter_diagnostics = _mask_filter_diagnostics(
                left, right, matches, left_mask, right_mask
            )
            object_left = object_features[left_index]
            object_right = object_features[right_index]
            raw_object_matches = match_descriptors(
                object_left, object_right,
                ratio_threshold=0.86,
                max_matches=6000,
            )
            # Enforce the authoritative masks again at original-image keypoint
            # coordinates. Detector masks constrain SIFT detection, while this
            # second gate guarantees every correspondence passed to RANSAC has
            # both endpoints inside the accepted foreground masks.
            object_matches = [
                match for match in raw_object_matches
                if _inside_both_manual_masks(
                    object_left, object_right, match, left_mask, right_mask
                )
            ]
            object_feature_mask_diagnostics = {
                "left": {
                    "detected_features": object_left.feature_count,
                    "keypoints_inside_mask": sum(
                        _point_inside_manual_mask(object_left, i, left_mask)
                        for i in range(object_left.feature_count)
                    ),
                },
                "right": {
                    "detected_features": object_right.feature_count,
                    "keypoints_inside_mask": sum(
                        _point_inside_manual_mask(object_right, i, right_mask)
                        for i in range(object_right.feature_count)
                    ),
                },
                "raw_descriptor_matches": len(raw_object_matches),
                "matches_with_both_endpoints_inside_masks": len(object_matches),
                "rejected_for_mask_violation": len(raw_object_matches) - len(object_matches),
                "ransac_input_matches": len(object_matches),
                "enforcement": "both_keypoint_centres_must_be_inside_their_saved_manual_masks",
            }
            object_result = verify_geometry(
                object_left, object_right, object_matches
            )
            object_only = {
                "status": "complete",
                **object_result.to_dict(),
                "feature_extraction": {
                    "method": "masked_sift_clahe_high_recall",
                    "left_object_features": object_left.feature_count,
                    "right_object_features": object_right.feature_count,
                    "ratio_threshold": 0.86,
                    "max_features_per_image": max(8000, settings.matching_max_features * 2),
                    "mask_boundary_eroded_px_at_working_resolution": 1,
                    "note": (
                        "Descriptors were detected independently inside each accepted automatic or manual "
                        "mask; this is not merely a post-filter of all-scene descriptors."
                    ),
                },
                "filtered_match_count": len(object_matches),
                "feature_mask_diagnostics": object_feature_mask_diagnostics,
                "raw_descriptor_match_count": len(raw_object_matches),
                "baseline_mask_comparison": mask_filter_diagnostics,
                "spatial_distribution": _spatial_distribution(
                    object_left, object_right, object_matches, object_result
                ),
                "mask_sources": [mask_sources[left_index], mask_sources[right_index]],
                "mask_filter_diagnostics": {
                    **mask_filter_diagnostics,
                    "interpretation": (
                        "Legacy all-scene descriptor comparison only. These counts do not "
                        "describe the independent object-only descriptor population."
                    ),
                },
                "boundary_distance_diagnostics": summarize_mask_boundary_distances(
                    left, right, matches, left_mask, right_mask
                ),
                "diagnostic_correspondences": _pair_diagnostics(
                    object_left, object_right, object_matches, object_result
                ),
                "note": (
                    "Independent object-only feature extraction, descriptor matching and "
                    "RANSAC. The all-scene baseline and connectivity graph remain separate."
                ),
            }

        reconstruction_pair_rows.append({
            "image_indices": [left_index, right_index],
            "_matches": matches,
            "_inliers": result.inlier_match_indices,
            "pose": (result.diagnostics or {}).get("relative_camera_pose"),
        })
        pair_results.append({
            "image_indices": [left_index, right_index],
            "filenames": [image_paths[left_index].name, image_paths[right_index].name],
            "pair_type": "adjacent" if is_adjacent else "non_adjacent",
            **result.to_dict(),
            "diagnostic_correspondences": _pair_diagnostics(left, right, matches, result),
            "spatial_distribution": _spatial_distribution(left, right, matches, result),
            "subject_region_support": pair_subject_support(
                matches, result,
                subject_regions[left_index], subject_regions[right_index],
            ),
            "object_only": object_only,
            "diagnostic_correspondences_note": (
                "Includes all geometric inliers and up to 100 closest-descriptor outliers. "
                "Coordinates are measured in the original uploaded images."
            ),
        })

    graph = build_connectivity_graph(len(image_paths), verified_pairs)
    sparse_reconstruction = reconstruct_sparse_scene(features, reconstruction_pair_rows)
    return {
        "analysis_id": str(analysis_id),
        "status": "complete",
        "processing": {
            "mode": "cpu_local",
            "gpu_required": False,
            "feature_extractor": "SIFT baseline + independent masked CLAHE-SIFT object pass",
            "pair_selection": "adjacent_frames_plus_evenly_sampled_pairs",
        },
        "summary": {
            "images_processed": len(image_paths),
            "pairs_evaluated": len(pairs),
            "adjacent_pairs_evaluated": adjacent_pairs_evaluated,
            "non_adjacent_pairs_evaluated": non_adjacent_pairs_evaluated,
            "pair_check_limit": settings.max_matching_pair_checks,
            "verified_pairs": graph["verified_edge_count"],
            "connected_components": len(graph["components"]),
            "isolated_images": len(graph["isolated_image_indices"]),
        },
        "images": [
            {
                "image_index": index,
                "filename": path.name,
                "feature_count": features[index].feature_count,
                "dimensions": {
                    "width": features[index].image_width,
                    "height": features[index].image_height,
                },
                "subject_mask": {
                    "status": mask_sources[index],
                    "foreground_fraction": (
                        round(float(manual_masks[index].mean()), 4)
                        if manual_masks[index] is not None else None
                    ),
                    "foreground_pixels": (
                        int(manual_masks[index].sum())
                        if manual_masks[index] is not None else None
                    ),
                    "total_pixels": (
                        int(manual_masks[index].size)
                        if manual_masks[index] is not None else None
                    ),
                },
            }
            for index, path in enumerate(image_paths)
        ],
        "pair_results": pair_results,
        "subject_region_diagnostics": {
            "method": "manual override; otherwise cached U2-Net ONNX CPU segmentation, with heuristic GrabCut fallback diagnostics",
            "status": (
                "available" if all(source in {"manual", "automatic"} for source in mask_sources)
                else "partial" if any(source in {"manual", "automatic"} for source in mask_sources)
                else "unavailable"
            ),
            "images": subject_summaries,
            "automatic_masks": automatic_summaries,
            "object_only_matching": (
                "A separate object-only RANSAC pass uses only matches whose two endpoints fall "
                "inside accepted U2-Net automatic masks or user manual overrides. Heuristic "
                "GrabCut fallbacks are never promoted to authoritative masks. Connectivity remains all-scene."
            ),
            "warning": (
                "Automatic masks are model predictions and may still omit details or include "
                "background. Review them when quality diagnostics are weak. Manual masks override "
                "automatic masks; heuristic GrabCut fallback is not used for object-only RANSAC."
            ),
        },
        "connectivity": graph,
        "sparse_reconstruction": sparse_reconstruction,
        "capture_recommendations": recommend_capture_improvements(
            [
                {
                    "image_index": index,
                    "feature_count": features[index].feature_count,
                }
                for index in range(len(features))
            ],
            pair_results,
            graph,
        ),
        "coverage_assessment": {
            "status": "not_assessed",
            "reason": (
                "Dense depth, fused surface geometry and occlusion-aware visibility are not "
                "implemented. Sparse camera/landmark results, when available, do not establish "
                "which object surfaces were observed or unseen."
            ),
        },
        "limitations": [
            "This endpoint performs CPU image matching and diagnostics, then attempts a separate "
            "calibrated sparse multi-view reconstruction when supported pose edges are available.",
            "Pairwise geometric inliers alone are not camera poses; sparse reconstruction remains "
            "a prototype and may be unavailable for uncalibrated, planar, low-parallax or disconnected data.",
            "The bounded pair schedule can omit valid links between images that were not compared.",
            "Disconnected images may be caused by blur, lighting, viewpoint changes, or missing overlap.",
            "Diagnostic coordinates are 2D feature correspondences, not 3D points or surface coverage.",
            "Spatial concentration warnings are heuristics and do not automatically invalidate a pair.",
            "Object-only diagnostics use masked keypoints, but the connectivity graph continues to use all-scene matching.",
            "Object-only RANSAC uses accepted automatic model masks or manual overrides; heuristic GrabCut fallbacks are not treated as semantic masks.",
        ],
    }
