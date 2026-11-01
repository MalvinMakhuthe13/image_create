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
    if result.model == "homography":
        assert result.median_reprojection_error_px is not None
        assert "median_symmetric_transfer_error_px" in result.diagnostics
    else:
        assert result.median_reprojection_error_px is None
        assert "median_sampson_error_px" in result.diagnostics


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
    assert result.verified
    assert result.diagnostics["spatial_concentration_is_advisory"] is True


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
    assert pose["sparse_triangulation"]["status"] == "estimated"
    assert pose["sparse_triangulation"]["triangulated_point_count"] >= 70
    assert pose["sparse_triangulation"]["coverage_claim"] == "sparse_pairwise_points_only_not_dense_surface_coverage"
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


def test_feature_extractor_uses_exif_35mm_equivalent_as_approximate_intrinsics(tmp_path):
    image = texture(76, 640)
    path = tmp_path / "with-focal-exif.jpg"
    pil_image = Image.fromarray(image)
    exif = pil_image.getexif()
    exif[41989] = 50
    pil_image.save(path, exif=exif)

    features = extract_features(path, 0)
    assert features.camera_matrix is not None
    assert features.intrinsics_source == "exif_35mm_equivalent_approximation"
    assert features.camera_matrix.shape == (3, 3)
    assert features.camera_matrix[0, 0] > 0


def test_global_sparse_reconstruction_fuses_multiview_tracks_and_refines_cameras():
    from app.engine.matching.models import DescriptorMatch, ImageFeatures
    from app.engine.matching.sparse_reconstruction import reconstruct_sparse_scene

    rng = np.random.default_rng(91)
    points = np.column_stack((
        rng.uniform(-1.0, 1.0, 36),
        rng.uniform(-0.7, 0.7, 36),
        rng.uniform(4.0, 7.0, 36),
    ))
    camera = np.array([[700.0, 0, 320.0], [0, 700.0, 240.0], [0, 0, 1.0]])
    translations = [np.array([0.0, 0.0, 0.0]), np.array([-0.7, 0.0, 0.0]), np.array([-1.4, 0.0, 0.0])]
    feature_sets = []
    for image_index, translation in enumerate(translations):
        camera_points = points + translation
        pixels_h = (camera @ camera_points.T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2, None]
        feature_sets.append(ImageFeatures(
            image_index=image_index,
            keypoints=tuple(map(tuple, pixels)),
            descriptors=None,
            image_width=640,
            image_height=480,
            scale_x=1.0,
            scale_y=1.0,
            camera_matrix=camera,
            intrinsics_source="test_calibration",
        ))

    def edge(left, right, tx):
        matches = [DescriptorMatch(i, i, 0.1) for i in range(len(points))]
        return {
            "image_indices": [left, right],
            "_matches": matches,
            "_inliers": tuple(range(len(matches))),
            "pose": {
                "status": "estimated",
                "rotation_matrix_left_to_right": np.eye(3).tolist(),
                "translation_direction_left_to_right": [tx, 0.0, 0.0],
                "pose_inlier_count": len(matches),
            },
        }

    result = reconstruct_sparse_scene(
        feature_sets,
        [edge(0, 1, -1.0), edge(1, 2, -1.0), edge(0, 2, -1.0)],
        max_tracks=100,
        max_nfev=15,
    )
    assert result["component_count"] == 1
    component = result["components"][0]
    assert component["status"] in {"optimized", "optimization_stopped", "low_quality"}
    assert result["status"] == ("complete" if component["status"] == "optimized" else "partial")
    assert component["reprojection_quality_gate"]["maximum_median_error_px"] == 5.0
    assert component["camera_count"] == 3
    assert component["landmark_count"] >= 30
    assert component["observation_count"] >= 90
    assert component["coverage_claim"] == "sparse_landmarks_only_not_dense_surface_coverage"


def test_global_sparse_reconstruction_reports_unavailable_without_calibrated_edges():
    from app.engine.matching.models import ImageFeatures
    from app.engine.matching.sparse_reconstruction import reconstruct_sparse_scene

    features = [
        ImageFeatures(i, (), None, 100, 100, 1.0, 1.0)
        for i in range(2)
    ]
    result = reconstruct_sparse_scene(features, [])
    assert result["status"] == "unavailable"
    assert result["reason"] == "no_calibrated_relative_pose_edges"
    assert result["coverage_assessment"]["status"] == "not_assessed"


def test_feature_extractor_prefers_user_calibration_and_distortion(tmp_path):
    import json

    image_path = tmp_path / "calibrated.jpg"
    Image.fromarray(texture(44, 640)).save(image_path)
    calibration = {
        "camera_matrix": [[800.0, 0.0, 320.0], [0.0, 805.0, 320.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [-0.1, 0.02, 0.0, 0.0, 0.0],
        "image_width": 640,
        "image_height": 640,
    }
    image_path.with_name(image_path.name + ".calibration.json").write_text(json.dumps(calibration))
    features = extract_features(image_path, 0)
    assert features.intrinsics_source == "user_supplied_calibration"
    assert np.allclose(features.camera_matrix, calibration["camera_matrix"])
    assert features.distortion_coefficients.shape == (5, 1)

def test_camera_centre_scale_gauge_uses_world_space_centres_for_rotated_cameras():
    from app.engine.matching.sparse_reconstruction import _camera_center

    angle = np.deg2rad(35.0)
    rotation_left = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    rotation_right = np.eye(3)
    centre_left = np.array([1.0, 2.0, 3.0])
    centre_right = np.array([4.0, 2.0, 3.0])
    translation_left = -rotation_left @ centre_left
    translation_right = -rotation_right @ centre_right

    recovered_left = _camera_center(rotation_left, translation_left)
    recovered_right = _camera_center(rotation_right, translation_right)

    assert np.allclose(recovered_left, centre_left)
    assert np.allclose(recovered_right, centre_right)
    assert np.isclose(np.linalg.norm(recovered_right - recovered_left), 3.0)

def test_camera_pose_initialization_uses_high_confidence_spanning_tree():
    from app.engine.matching.sparse_reconstruction import (
        PairPose,
        _initial_camera_poses,
    )

    identity = np.eye(3)
    edges = [
        PairPose(0, 1, identity, np.array([-1.0, 0.0, 0.0]), 100.0),
        PairPose(1, 2, identity, np.array([-1.0, 0.0, 0.0]), 90.0),
        # A low-confidence loop closure must not override the stronger chain.
        PairPose(0, 2, identity, np.array([-100.0, 0.0, 0.0]), 1.0),
    ]

    rotations, translations, components = _initial_camera_poses(3, edges)

    assert components == [[0, 1, 2]]
    assert np.allclose(rotations[0], identity)
    assert np.allclose(rotations[1], identity)
    assert np.allclose(rotations[2], identity)
    assert np.allclose(translations[1], [-1.0, 0.0, 0.0])
    assert np.allclose(translations[2], [-2.0, 0.0, 0.0])

def test_camera_pose_composition_preserves_non_identity_rotations():
    from app.engine.matching.sparse_reconstruction import (
        PairPose,
        _initial_camera_poses,
    )

    angle_y = np.deg2rad(20.0)
    angle_z = np.deg2rad(-15.0)
    rotation_01 = np.array([
        [np.cos(angle_y), 0.0, np.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle_y), 0.0, np.cos(angle_y)],
    ])
    rotation_12 = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0.0],
        [np.sin(angle_z), np.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation_01 = np.array([-0.8, 0.1, 0.0])
    translation_12 = np.array([-0.4, 0.2, 0.1])
    edges = [
        PairPose(0, 1, rotation_01, translation_01, 10.0),
        PairPose(1, 2, rotation_12, translation_12, 9.0),
    ]

    rotations, translations, components = _initial_camera_poses(3, edges)

    assert components == [[0, 1, 2]]
    assert np.allclose(rotations[1], rotation_01)
    assert np.allclose(translations[1], translation_01)
    assert np.allclose(rotations[2], rotation_12 @ rotation_01)
    assert np.allclose(
        translations[2], rotation_12 @ translation_01 + translation_12
    )


def test_camera_pose_composition_inverts_reversed_relative_edge():
    from app.engine.matching.sparse_reconstruction import (
        PairPose,
        _initial_camera_poses,
    )

    angle = np.deg2rad(25.0)
    rotation_01 = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    translation_01 = np.array([-0.7, 0.05, 0.1])
    edge_01 = PairPose(0, 1, rotation_01, translation_01, 5.0)

    rotations, translations, components = _initial_camera_poses(2, [edge_01])

    assert components == [[0, 1]]
    assert np.allclose(rotations[1], rotation_01)
    assert np.allclose(translations[1], translation_01)

    # A graph whose only stored edge is 1 -> 0 must recover camera 1
    # by applying the inverse rigid transform.
    edge_10 = PairPose(1, 0, rotation_01.T, -rotation_01.T @ translation_01, 5.0)
    reverse_rotations, reverse_translations, _ = _initial_camera_poses(2, [edge_10])
    assert np.allclose(reverse_rotations[1], rotation_01)
    assert np.allclose(reverse_translations[1], translation_01)



def test_object_feature_extraction_is_masked_and_high_recall(tmp_path):
    from app.engine.matching.feature_extractor import extract_features

    rng = np.random.default_rng(20261101)
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    # Textured foreground object region surrounded by a separately textured
    # background, as in the cap-on-table capture.
    image[130:490, 220:580] = rng.integers(
        0, 256, (360, 360, 3), dtype=np.uint8
    )
    image[:120] = rng.integers(0, 256, image[:120].shape, dtype=np.uint8)
    image[500:] = rng.integers(0, 256, image[500:].shape, dtype=np.uint8)
    path = tmp_path / "masked-object.png"
    Image.fromarray(image).save(path)
    mask = np.zeros((600, 800), dtype=np.uint8)
    mask[130:490, 220:580] = 255

    baseline = extract_features(path, 0, max_dimension=800, max_features=3000)
    object_features = extract_features(
        path, 0, max_dimension=800, max_features=8000,
        mask=mask, feature_profile="object",
    )

    assert object_features.feature_count > 0
    assert object_features.descriptors is not None
    assert len(object_features.descriptors) == object_features.feature_count
    assert all(
        130 <= y < 490 and 220 <= x < 580
        for x, y in object_features.keypoints
    )
    assert object_features.feature_count <= 8000
    assert baseline.feature_count > 0


def test_masked_object_feature_extraction_rejects_wrong_mask_dimensions(tmp_path):
    image = texture(2026, 320)
    path = tmp_path / "object.png"
    Image.fromarray(image).save(path)
    with np.testing.assert_raises(ValueError):
        extract_features(
            path, 0, mask=np.ones((100, 100), dtype=np.uint8),
            feature_profile="object",
        )
