from __future__ import annotations

import cv2
import numpy as np


DISTANCE_BANDS = (
    ("0_to_2_px", 0.0, 2.0),
    ("over_2_to_5_px", 2.0, 5.0),
    ("over_5_to_10_px", 5.0, 10.0),
    ("over_10_to_20_px", 10.0, 20.0),
    ("over_20_px", 20.0, float("inf")),
)


def _empty_histogram() -> dict[str, int]:
    return {name: 0 for name, _, _ in DISTANCE_BANDS}


def _band_for(distance: float) -> str:
    for name, lower, upper in DISTANCE_BANDS:
        if distance > lower or lower == 0.0:
            if distance <= upper:
                return name
    return "over_20_px"


def _side_distances(image, matches, mask, side: str) -> dict:
    histogram = _empty_histogram()
    distances: dict[int, float] = {}
    outside_count = 0
    inside_count = 0
    out_of_image_count = 0
    invalid_index_count = 0

    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("Manual mask must be a two-dimensional array.")

    if not mask.any():
        return {
            "status": "no_foreground_pixels",
            "outside_mask_endpoints": 0,
            "inside_mask_endpoints": 0,
            "out_of_image_endpoints": 0,
            "invalid_keypoint_indices": 0,
            "measurable_outside_endpoints": 0,
            "distance_bands": histogram,
            "distances_by_match": distances,
        }

    height, width = mask.shape
    distance_map = cv2.distanceTransform(
        np.ascontiguousarray((~mask).astype(np.uint8)),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )

    for match_index, match in enumerate(matches):
        keypoint_index = (
            match.query_index if side == "left" else match.train_index
        )

        if keypoint_index < 0 or keypoint_index >= len(image.keypoints):
            invalid_index_count += 1
            continue

        x, y = image.keypoints[keypoint_index]
        px, py = int(x), int(y)

        if not (0 <= px < width and 0 <= py < height):
            out_of_image_count += 1
            continue

        if mask[py, px]:
            inside_count += 1
            continue

        outside_count += 1
        distance = float(distance_map[py, px])
        distances[match_index] = distance
        histogram[_band_for(distance)] += 1

    return {
        "status": "complete",
        "outside_mask_endpoints": outside_count,
        "inside_mask_endpoints": inside_count,
        "out_of_image_endpoints": out_of_image_count,
        "invalid_keypoint_indices": invalid_index_count,
        "measurable_outside_endpoints": len(distances),
        "distance_bands": histogram,
        "distances_by_match": distances,
    }


def summarize_mask_boundary_distances(
    left,
    right,
    matches,
    left_mask,
    right_mask,
) -> dict:
    """Measure excluded match endpoints' distance to manual-mask foreground.

    This is diagnostic only. It does not modify masks or match eligibility.
    """
    left_result = _side_distances(left, matches, left_mask, "left")
    right_result = _side_distances(right, matches, right_mask, "right")

    pair_histogram = _empty_histogram()
    at_least_one_outside = 0
    both_outside = 0
    no_measurable_distance = 0

    for index in range(len(matches)):
        left_distance = left_result["distances_by_match"].get(index)
        right_distance = right_result["distances_by_match"].get(index)

        left_outside = left_distance is not None
        right_outside = right_distance is not None

        if left_outside or right_outside:
            at_least_one_outside += 1
        if left_outside and right_outside:
            both_outside += 1

        measurable = [
            distance
            for distance in (left_distance, right_distance)
            if distance is not None
        ]

        if not measurable:
            if left_outside or right_outside:
                no_measurable_distance += 1
            continue

        pair_histogram[_band_for(min(measurable))] += 1

    return {
        "status": "complete",
        "method": "opencv_precise_euclidean_distance_transform",
        "distance_unit": "pixels",
        "left_image": {
            key: value
            for key, value in left_result.items()
            if key != "distances_by_match"
        },
        "right_image": {
            key: value
            for key, value in right_result.items()
            if key != "distances_by_match"
        },
        "correspondences": {
            "total": len(matches),
            "at_least_one_outside_endpoint": at_least_one_outside,
            "both_endpoints_outside": both_outside,
            "outside_without_measurable_distance": no_measurable_distance,
            "distance_bands_using_nearest_endpoint": pair_histogram,
        },
        "interpretation": (
            "Distances describe excluded descriptor-match endpoints relative "
            "to the nearest manual-mask foreground pixel. They are advisory "
            "only and do not change mask eligibility or geometric verification."
        ),
    }
