from __future__ import annotations

import cv2
import numpy as np

from app.engine.matching.models import DescriptorMatch, GeometryResult, ImageFeatures
from app.engine.matching.pose_estimator import estimate_relative_pose


def _sampson_errors(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return first-order epipolar (Sampson) errors in pixels."""
    source_h = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    target_h = np.column_stack((target, np.ones(len(target), dtype=np.float64)))
    f_source = (matrix @ source_h.T).T
    ft_target = (matrix.T @ target_h.T).T
    numerator = np.sum(target_h * f_source, axis=1) ** 2
    denominator = (
        f_source[:, 0] ** 2 + f_source[:, 1] ** 2
        + ft_target[:, 0] ** 2 + ft_target[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _symmetric_transfer_errors(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Return symmetric homography transfer error in pixels."""
    inverse = np.linalg.inv(matrix)
    source_h = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    target_h = np.column_stack((target, np.ones(len(target), dtype=np.float64)))
    projected_target = (matrix @ source_h.T).T
    projected_source = (inverse @ target_h.T).T
    valid = (
        (np.abs(projected_target[:, 2]) > 1e-12)
        & (np.abs(projected_source[:, 2]) > 1e-12)
    )
    errors = np.full(len(source), np.inf, dtype=np.float64)
    forward = projected_target[valid, :2] / projected_target[valid, 2, None]
    backward = projected_source[valid, :2] / projected_source[valid, 2, None]
    errors[valid] = np.sqrt(
        0.5 * (
            np.sum((forward - target[valid]) ** 2, axis=1)
            + np.sum((backward - source[valid]) ** 2, axis=1)
        )
    )
    return errors


def _spread(points: np.ndarray, width: int, height: int) -> dict:
    if len(points) == 0 or width <= 0 or height <= 0:
        return {"occupied_grid_cells": 0, "bbox_area_fraction": 0.0}
    columns = np.clip((points[:, 0] / width * 4).astype(int), 0, 3)
    rows = np.clip((points[:, 1] / height * 4).astype(int), 0, 3)
    occupied = len(set(zip(rows.tolist(), columns.tolist())))
    bbox = (
        max(0.0, float(points[:, 0].max() - points[:, 0].min())) / width
    ) * (
        max(0.0, float(points[:, 1].max() - points[:, 1].min())) / height
    )
    return {"occupied_grid_cells": occupied, "bbox_area_fraction": round(bbox, 5)}


def verify_geometry(
    left: ImageFeatures,
    right: ImageFeatures,
    matches: list[DescriptorMatch],
    *,
    ransac_threshold_px: float = 2.5,
    minimum_inliers: int = 12,
    minimum_inlier_ratio: float = 0.20,
) -> GeometryResult:
    count = len(matches)
    if count < 8 or not left.keypoints or not right.keypoints:
        return GeometryResult(
            False, "none", count, 0, 0.0, None, "insufficient_matches",
            diagnostics={"model_status": "not_attempted"},
        )
    if ransac_threshold_px <= 0 or minimum_inliers < 4:
        raise ValueError("Invalid geometric verification thresholds")
    if not 0.0 <= minimum_inlier_ratio <= 1.0:
        raise ValueError("minimum_inlier_ratio must be in [0, 1]")

    try:
        src = np.float64([left.keypoints[m.query_index] for m in matches])
        dst = np.float64([right.keypoints[m.train_index] for m in matches])
    except (IndexError, TypeError) as exc:
        raise ValueError("A descriptor match references a missing keypoint") from exc
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Keypoint coordinates must be finite")

    candidates = []
    fundamental_matrix = None
    fundamental_mask = None
    homography_matrix = None
    homography_mask = None
    try:
        matrix, mask = cv2.findFundamentalMat(
            src.astype(np.float32), dst.astype(np.float32),
            cv2.FM_RANSAC, ransac_threshold_px, 0.999, 5000,
        )
        if matrix is not None and mask is not None and matrix.shape == (3, 3):
            fundamental_matrix = matrix.astype(np.float64)
            fundamental_mask = mask.reshape(-1).astype(bool)
            candidates.append(("fundamental", fundamental_mask))
    except cv2.error:
        pass
    try:
        matrix, mask = cv2.findHomography(
            src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC,
            ransac_threshold_px, maxIters=5000, confidence=0.999,
        )
        if matrix is not None and mask is not None and matrix.shape == (3, 3):
            homography_matrix = matrix.astype(np.float64)
            homography_mask = mask.reshape(-1).astype(bool)
            candidates.append(("homography", homography_mask))
    except cv2.error:
        pass

    if not candidates:
        return GeometryResult(
            False, "none", count, 0, 0.0, None, "ransac_failed",
            diagnostics={"model_status": "both_estimators_failed"},
        )

    model, inlier_mask = max(candidates, key=lambda item: int(item[1].sum()))
    inlier_indices = tuple(int(index) for index in np.flatnonzero(inlier_mask))
    inliers = len(inlier_indices)
    ratio = inliers / count

    if model == "fundamental" and fundamental_matrix is not None:
        residuals = _sampson_errors(fundamental_matrix, src, dst)
        residual_name = "median_sampson_error_px"
    elif model == "homography" and homography_matrix is not None:
        try:
            residuals = _symmetric_transfer_errors(homography_matrix, src, dst)
        except np.linalg.LinAlgError:
            residuals = np.full(count, np.inf)
        residual_name = "median_symmetric_transfer_error_px"
    else:
        residuals = np.full(count, np.inf)
        residual_name = "median_geometric_error_px"

    inlier_residuals = residuals[inlier_mask]
    finite_residuals = inlier_residuals[np.isfinite(inlier_residuals)]
    median_error = (
        round(float(np.median(finite_residuals)), 4)
        if len(finite_residuals) else None
    )

    left_inlier_points = src[inlier_mask]
    right_inlier_points = dst[inlier_mask]
    left_spread = _spread(
        left_inlier_points, left.image_width, left.image_height
    )
    right_spread = _spread(
        right_inlier_points, right.image_width, right.image_height
    )
    f_count = int(fundamental_mask.sum()) if fundamental_mask is not None else 0
    h_count = int(homography_mask.sum()) if homography_mask is not None else 0
    planar_dominant = h_count >= minimum_inliers and (
        f_count == 0 or h_count >= max(1, int(np.ceil(f_count * 1.15)))
    )
    spatially_concentrated = (
        inliers >= 8 and (
            left_spread["occupied_grid_cells"] < 3
            or right_spread["occupied_grid_cells"] < 3
            or left_spread["bbox_area_fraction"] < 0.01
            or right_spread["bbox_area_fraction"] < 0.01
        )
    )
    warnings = []
    if planar_dominant:
        warnings.append("homography_dominant_possible_planar_scene")
    if spatially_concentrated:
        warnings.append("inliers_spatially_concentrated")
    if f_count and h_count and abs(f_count - h_count) <= max(2, int(0.05 * max(f_count, h_count))):
        warnings.append("fundamental_and_homography_support_similar")

    if inliers < minimum_inliers:
        verified, reason = False, "insufficient_inliers"
    elif ratio < minimum_inlier_ratio:
        verified, reason = False, "low_inlier_ratio"
    else:
        verified, reason = True, "geometrically_consistent"

    diagnostics = {
        "model_status": "estimated",
        "candidate_inlier_counts": {
            "fundamental": f_count,
            "homography": h_count,
        },
        "selected_model": model,
        residual_name: median_error,
        "ransac_threshold_px": ransac_threshold_px,
        "left_image_inlier_spread": left_spread,
        "right_image_inlier_spread": right_spread,
        "possible_planar_or_low_parallax_scene": planar_dominant,
        "spatial_concentration_is_advisory": True,
        "spatially_concentrated": spatially_concentrated,
        "warnings": warnings,
        "interpretation": (
            "Model support and residuals are pairwise 2D diagnostics. A homography-dominant "
            "pair may be planar or have low parallax; this is not a calibrated camera pose."
        ),
    }
    if model == "fundamental" and fundamental_mask is not None and not planar_dominant:
        pose_inlier_indices = tuple(int(index) for index in np.flatnonzero(fundamental_mask))
        diagnostics["relative_camera_pose"] = estimate_relative_pose(
            left, right, matches, pose_inlier_indices,
            ransac_threshold_px=ransac_threshold_px,
        )
        diagnostics["relative_camera_pose"]["pose_source"] = "fundamental_matrix_inliers"
    else:
        diagnostics["relative_camera_pose"] = {
            "status": "unavailable",
            "reason": "homography_selected_or_planar_low_parallax_ambiguity",
        }
    return GeometryResult(
        verified, model, count, inliers, round(ratio, 4),
        median_error if model == "homography" else None,
        reason, inlier_indices, diagnostics,
    )
