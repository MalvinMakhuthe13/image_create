from __future__ import annotations

import cv2
import numpy as np

from app.engine.matching.models import ImageFeatures


def estimate_relative_pose(
    left: ImageFeatures,
    right: ImageFeatures,
    matches,
    inlier_indices,
    *,
    ransac_threshold_px: float = 2.5,
) -> dict:
    """Estimate relative rotation and translation direction when both cameras have intrinsics.

    Translation magnitude is not observable from a monocular image pair. EXIF-derived
    intrinsics are explicitly marked approximate and must not be presented as calibration.
    """
    if left.camera_matrix is None or right.camera_matrix is None:
        return {
            "status": "unavailable",
            "reason": "camera_intrinsics_missing",
        }
    if len(inlier_indices) < 5:
        return {
            "status": "unavailable",
            "reason": "insufficient_geometric_inliers",
        }
    try:
        left_pixels = np.float64([
            left.keypoints[matches[index].query_index] for index in inlier_indices
        ]).reshape(-1, 1, 2)
        right_pixels = np.float64([
            right.keypoints[matches[index].train_index] for index in inlier_indices
        ]).reshape(-1, 1, 2)
        left_normalized = cv2.undistortPoints(
            left_pixels, left.camera_matrix, None
        ).reshape(-1, 2)
        right_normalized = cv2.undistortPoints(
            right_pixels, right.camera_matrix, None
        ).reshape(-1, 2)
        focal = float(np.mean([
            left.camera_matrix[0, 0], left.camera_matrix[1, 1],
            right.camera_matrix[0, 0], right.camera_matrix[1, 1],
        ]))
        threshold_normalized = ransac_threshold_px / max(focal, 1.0)
        essential, mask = cv2.findEssentialMat(
            left_normalized, right_normalized, np.eye(3),
            method=cv2.RANSAC, prob=0.999, threshold=threshold_normalized,
        )
        if essential is None or mask is None:
            return {"status": "unavailable", "reason": "essential_matrix_estimation_failed"}
        if essential.shape[0] > 3:
            essential = essential[:3, :]
        count, rotation, translation, pose_mask = cv2.recoverPose(
            essential, left_normalized, right_normalized, np.eye(3), mask=mask
        )
        if count < 5:
            return {"status": "unavailable", "reason": "cheirality_support_too_low"}
        pose_inliers = pose_mask.reshape(-1).astype(bool)
        projection_left = np.hstack((np.eye(3), np.zeros((3, 1))))
        projection_right = np.hstack((rotation, translation))
        homogeneous = cv2.triangulatePoints(
            projection_left, projection_right,
            left_normalized[pose_inliers].T, right_normalized[pose_inliers].T,
        )
        points_3d = (homogeneous[:3] / np.maximum(homogeneous[3], 1e-12)).T
        points_right = (rotation @ points_3d.T + translation).T
        positive_depth = (points_3d[:, 2] > 0) & (points_right[:, 2] > 0)
        triangulated = points_3d[positive_depth]
        if len(triangulated):
            left_projected = triangulated[:, :2] / triangulated[:, 2, None]
            right_projected_3d = (rotation @ triangulated.T + translation).T
            right_projected = right_projected_3d[:, :2] / right_projected_3d[:, 2, None]
            left_focal = float(np.mean([left.camera_matrix[0, 0], left.camera_matrix[1, 1]]))
            right_focal = float(np.mean([right.camera_matrix[0, 0], right.camera_matrix[1, 1]]))
            left_error = np.linalg.norm(
                left_projected - left_normalized[pose_inliers][positive_depth], axis=1
            ) * left_focal
            right_error = np.linalg.norm(
                right_projected - right_normalized[pose_inliers][positive_depth], axis=1
            ) * right_focal
            reprojection_error = float(np.median(np.concatenate((left_error, right_error))))
        else:
            reprojection_error = None
        return {
            "status": "estimated",
            "rotation_matrix_left_to_right": np.round(rotation, 7).tolist(),
            "translation_direction_left_to_right": np.round(
                translation.reshape(3), 7
            ).tolist(),
            "translation_scale": "unknown_monocular_scale",
            "pose_inlier_count": int(count),
            "sparse_triangulation": {
                "status": "estimated" if len(triangulated) >= 5 else "insufficient_positive_depth_points",
                "triangulated_point_count": int(len(triangulated)),
                "positive_depth_fraction": round(float(len(triangulated) / max(count, 1)), 4),
                "median_reprojection_error_px": (
                    round(reprojection_error, 4) if reprojection_error is not None else None
                ),
                "scale": "arbitrary_pairwise_monocular_scale",
                "coverage_claim": "sparse_pairwise_points_only_not_dense_surface_coverage",
            },
            "essential_matrix_inlier_count": int(mask.sum()),
            "intrinsics_sources": [
                left.intrinsics_source or "provided_camera_matrix",
                right.intrinsics_source or "provided_camera_matrix",
            ],
            "calibration_quality": (
                "approximate_exif_intrinsics"
                if "exif_35mm_equivalent_approximation" in {
                    left.intrinsics_source, right.intrinsics_source
                }
                else "user_or_external_calibration"
            ),
            "limitations": [
                "Translation magnitude and metric scale are unknown.",
                "No lens distortion model was available.",
                "EXIF 35mm-equivalent focal length yields approximate intrinsics only.",
                "Pairwise pose estimates require global pose-graph optimization before use as a camera trajectory.",
            ],
        }
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return {"status": "unavailable", "reason": "relative_pose_estimation_failed"}
