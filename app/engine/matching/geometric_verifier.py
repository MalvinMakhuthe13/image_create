from __future__ import annotations

import cv2
import numpy as np

from app.engine.matching.models import DescriptorMatch, GeometryResult, ImageFeatures


def verify_geometry(left: ImageFeatures, right: ImageFeatures, matches: list[DescriptorMatch], *,
                    ransac_threshold_px: float = 2.5, minimum_inliers: int = 12,
                    minimum_inlier_ratio: float = 0.20) -> GeometryResult:
    count = len(matches)
    if count < 8 or not left.keypoints or not right.keypoints:
        return GeometryResult(False, "none", count, 0, 0.0, None, "insufficient_matches")
    if ransac_threshold_px <= 0 or minimum_inliers < 4:
        raise ValueError("Invalid geometric verification thresholds")
    src = np.float32([left.keypoints[m.query_index] for m in matches])
    dst = np.float32([right.keypoints[m.train_index] for m in matches])
    candidates = []
    try:
        matrix, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, ransac_threshold_px, 0.999)
        if matrix is not None and mask is not None and matrix.shape == (3, 3):
            candidates.append(("fundamental", mask.reshape(-1).astype(bool)))
    except cv2.error:
        pass
    try:
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold_px,
                                     maxIters=3000, confidence=0.999)
        if mask is not None:
            candidates.append(("homography", mask.reshape(-1).astype(bool)))
    except cv2.error:
        pass
    if not candidates:
        return GeometryResult(False, "none", count, 0, 0.0, None, "ransac_failed")
    model, inlier_mask = max(candidates, key=lambda item: int(item[1].sum()))
    inliers = int(inlier_mask.sum())
    ratio = inliers / count
    verified = inliers >= minimum_inliers and ratio >= minimum_inlier_ratio
    reason = "geometrically_consistent" if verified else "insufficient_geometric_support"
    return GeometryResult(verified, model, count, inliers, round(ratio, 4), None, reason)
