from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.engine.matching.connectivity_graph import build_connectivity_graph
from app.engine.matching.feature_extractor import extract_features
from app.engine.matching.geometric_verifier import verify_geometry
from app.engine.matching.models import GeometryResult
from app.engine.matching.pair_matcher import match_descriptors


def texture(seed: int, size: int = 720) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    for _ in range(18):
        point = (int(rng.integers(30, size - 30)), int(rng.integers(30, size - 30)))
        cv2.circle(image, point, int(rng.integers(8, 28)), (255, 255, 255), 2)
    return image

def test_sift_verifies_transformed_pair(tmp_path):
    first = texture(7)
    transform = cv2.getRotationMatrix2D((360, 360), 7, 0.98)
    transform[0, 2] += 9
    second = cv2.warpAffine(first, transform, (720, 720))
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(first).save(p1)
    Image.fromarray(second).save(p2)
    left, right = extract_features(p1, 0), extract_features(p2, 1)
    matches = match_descriptors(left, right)
    result = verify_geometry(left, right, matches)
    assert left.feature_count > 50
    assert len(matches) >= 12
    assert result.verified
    assert result.model in {"fundamental", "homography"}
    assert result.inlier_matches >= 12

def test_unrelated_images_do_not_connect(tmp_path):
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(texture(11)).save(p1)
    Image.fromarray(texture(99)).save(p2)
    left, right = extract_features(p1, 0), extract_features(p2, 1)
    result = verify_geometry(left, right, match_descriptors(left, right))
    graph = build_connectivity_graph(2, [(0, 1, result)])
    assert not result.verified
    assert graph["verified_edge_count"] == 0
    assert graph["components"] == [[0], [1]]
    assert graph["isolated_image_indices"] == [0, 1]

def test_graph_components():
    good = GeometryResult(True, "fundamental", 30, 24, 0.8, None, "geometrically_consistent")
    bad = GeometryResult(False, "none", 30, 2, 0.0667, None, "insufficient_geometric_support")
    graph = build_connectivity_graph(4, [(0, 1, good), (1, 2, good), (2, 3, bad)])
    assert graph["components"] == [[0, 1, 2], [3]]
    assert graph["isolated_image_indices"] == [3]
    assert graph["verified_edge_count"] == 2


def test_geometry_reports_residuals_and_model_competition(tmp_path):
    first = texture(41)
    transform = cv2.getRotationMatrix2D((360, 360), 4, 1.0)
    transform[0, 2] += 12
    second = cv2.warpAffine(first, transform, (720, 720))
    p1, p2 = tmp_path / "geometry-a.png", tmp_path / "geometry-b.png"
    Image.fromarray(first).save(p1)
    Image.fromarray(second).save(p2)
    left, right = extract_features(p1, 0), extract_features(p2, 1)
    result = verify_geometry(left, right, match_descriptors(left, right))
    assert result.diagnostics is not None
    assert result.diagnostics["model_status"] == "estimated"
    assert result.diagnostics["candidate_inlier_counts"]["fundamental"] >= 0
    assert result.diagnostics["candidate_inlier_counts"]["homography"] >= 0
    assert result.diagnostics["selected_model"] == result.model
    assert result.median_reprojection_error_px is not None


def test_geometry_rejects_concentrated_correspondence_cluster():
    from app.engine.matching.geometric_verifier import verify_geometry
    from app.engine.matching.models import DescriptorMatch, ImageFeatures

    points = tuple((float(100 + i % 4), float(100 + i // 4)) for i in range(40))
    shifted = tuple((x + 8.0, y + 5.0) for x, y in points)
    descriptors = np.ones((40, 128), dtype=np.float32)
    left = ImageFeatures(0, points, descriptors, 1000, 1000, 1.0, 1.0)
    right = ImageFeatures(1, shifted, descriptors, 1000, 1000, 1.0, 1.0)
    matches = [DescriptorMatch(i, i, float(i)) for i in range(40)]
    result = verify_geometry(left, right, matches, minimum_inliers=8)
    assert result.diagnostics is not None
    assert result.diagnostics["spatially_concentrated"] is True
    assert not result.verified
    assert result.reason == "spatially_concentrated_inliers"


def test_relative_pose_uses_intrinsics_and_reports_unknown_translation_scale():
    from app.engine.matching.models import DescriptorMatch, ImageFeatures
    from app.engine.matching.pose_estimator import estimate_relative_pose

    rng = np.random.default_rng(18)
    points_3d = np.column_stack((
        rng.uniform(-1.5, 1.5, 80),
        rng.uniform(-1.0, 1.0, 80),
        rng.uniform(4.0, 9.0, 80),
    ))
    focal = 900.0
    camera = np.array([
        [focal, 0.0, 640.0],
        [0.0, focal, 480.0],
        [0.0, 0.0, 1.0],
    ])
    angle = np.deg2rad(3.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    translation = np.array([0.35, 0.02, 0.01])

    def project(points):
        normalized = points[:, :2] / points[:, 2, None]
        pixels = normalized * focal + np.array([640.0, 480.0])
        return tuple(map(tuple, pixels))

    left_points = project(points_3d)
    right_3d = (rotation @ points_3d.T).T + translation
    right_points = project(right_3d)
    left = ImageFeatures(0, left_points, None, 1280, 960, 1.0, 1.0, camera, "test_calibration")
    right = ImageFeatures(1, right_points, None, 1280, 960, 1.0, 1.0, camera, "test_calibration")
    matches = [DescriptorMatch(i, i, 1.0) for i in range(len(left_points))]

    pose = estimate_relative_pose(left, right, matches, range(len(matches)))
    assert pose["status"] == "estimated"
    assert pose["pose_inlier_count"] >= 70
    assert pose["translation_scale"] == "unknown_monocular_scale"
    assert pose["calibration_quality"] == "user_or_external_calibration"


def test_relative_pose_is_explicitly_unavailable_without_intrinsics():
    from app.engine.matching.models import DescriptorMatch, ImageFeatures
    from app.engine.matching.pose_estimator import estimate_relative_pose

    points = tuple((float(i), float(i * 2)) for i in range(10))
    left = ImageFeatures(0, points, None, 100, 100, 1.0, 1.0)
    right = ImageFeatures(1, points, None, 100, 100, 1.0, 1.0)
    matches = [DescriptorMatch(i, i, 1.0) for i in range(10)]
    assert estimate_relative_pose(left, right, matches, range(10)) == {
        "status": "unavailable", "reason": "camera_intrinsics_missing"
    }


def test_capture_advisor_only_recommends_from_reported_evidence():
    from app.engine.matching.capture_advisor import recommend_capture_improvements

    images = [
        {"image_index": 0, "feature_count": 30},
        {"image_index": 1, "feature_count": 500},
    ]
    pairs = [{
        "image_indices": [0, 1],
        "pair_type": "adjacent",
        "verified": False,
        "reason": "insufficient_inliers",
        "tentative_matches": 18,
        "inlier_matches": 4,
    }]
    graph = {
        "image_count": 2,
        "components": [[0], [1]],
        "isolated_image_indices": [0, 1],
    }
    recommendations = recommend_capture_improvements(images, pairs, graph)
    assert recommendations
    assert all("evidence" in item and "confidence" in item for item in recommendations)
    assert any(item["image_indices"] == [0, 1] for item in recommendations)
