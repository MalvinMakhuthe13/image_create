from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import settings
from app.main import app

client = TestClient(app)


def image_bytes():
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    for x in range(20, 620, 40):
        draw.rectangle((x, 30, x + 18, 450), outline=(x % 255, 90, 180), width=3)
    draw.ellipse((180, 100, 460, 380), outline="black", width=8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_matching_endpoint_handles_single_image_without_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    upload = client.post(
        "/api/v1/analysis/images",
        files=[("files", ("single.png", image_bytes(), "image/png"))],
    )
    assert upload.status_code == 200
    analysis_id = upload.json()["analysis_id"]

    response = client.post(f"/api/v1/analysis/{analysis_id}/matching")
    assert response.status_code == 200
    body = response.json()
    assert body["processing"]["mode"] == "cpu_local"
    assert body["processing"]["gpu_required"] is False
    assert body["summary"]["images_processed"] == 1
    assert body["summary"]["pairs_evaluated"] == 0
    assert body["connectivity"]["components"] == [[0]]
    assert body["coverage_assessment"]["status"] == "not_assessed"
    assert "Dense depth" in body["coverage_assessment"]["reason"]
    assert "sparse_reconstruction" in body
    assert any("sparse multi-view reconstruction" in item for item in body["limitations"])


def test_matching_endpoint_returns_404_for_unknown_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    response = client.post(f"/api/v1/analysis/{uuid4()}/matching")
    assert response.status_code == 404


def test_candidate_pair_scheduler_keeps_adjacent_pairs():
    from app.api.routes.matching import _candidate_pairs

    pairs = _candidate_pairs(6, 8)
    assert all((index, index + 1) in pairs for index in range(5))
    assert len(pairs) == 8


def test_diagnostics_include_coordinates_and_pair_types(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    upload = client.post(
        "/api/v1/analysis/images",
        files=[
            ("files", ("first.png", image_bytes(), "image/png")),
            ("files", ("second.png", image_bytes(), "image/png")),
        ],
    )
    assert upload.status_code == 200
    response = client.post(
        f"/api/v1/analysis/{upload.json()['analysis_id']}/matching"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["adjacent_pairs_evaluated"] == 1
    pair = body["pair_results"][0]
    assert pair["pair_type"] == "adjacent"
    assert "diagnostic_correspondences" in pair
    assert "object_only" in pair
    assert "diagnostic_correspondences" in pair["object_only"]
    assert "dimensions" in body["images"][0]
    assert body["images"][0]["manual_mask"]["status"] == "missing_or_invalid"
    assert body["pair_results"][0]["object_only"]["mask_filter_diagnostics"]["status"] == "unavailable"

def test_spatial_distribution_flags_concentrated_inliers():
    from types import SimpleNamespace

    from app.api.routes.matching import _spatial_distribution
    from app.engine.matching.models import DescriptorMatch, GeometryResult

    points = tuple((100.0 + i * 0.2, 100.0 + i * 0.2) for i in range(8))
    left = SimpleNamespace(image_width=640, image_height=480, keypoints=points)
    right = SimpleNamespace(image_width=640, image_height=480, keypoints=points)
    matches = [DescriptorMatch(i, i, 10.0) for i in range(8)]
    result = GeometryResult(
        verified=True, model="fundamental", tentative_matches=8,
        inlier_matches=8, inlier_ratio=1.0, median_reprojection_error_px=None,
        reason="geometrically_consistent", inlier_match_indices=tuple(range(8)),
    )

    summary = _spatial_distribution(left, right, matches, result)
    assert summary["left_image"]["occupied_grid_cells"] == 1
    assert summary["right_image"]["occupied_grid_cells"] == 1
    assert summary["left_image"]["spatially_concentrated"] is True
    assert summary["warning"] == "inlier_distribution_is_spatially_concentrated"
    assert summary["left_image"]["near_duplicate_location_groups_5px"] >= 1


def test_spatial_distribution_accepts_well_spread_inliers():
    from types import SimpleNamespace

    from app.api.routes.matching import _spatial_distribution
    from app.engine.matching.models import DescriptorMatch, GeometryResult

    points = tuple(
        (float(x), float(y))
        for y in (40, 160, 280, 400)
        for x in (40, 160, 280, 400)
    )
    left = SimpleNamespace(image_width=640, image_height=480, keypoints=points)
    right = SimpleNamespace(image_width=640, image_height=480, keypoints=points)
    matches = [DescriptorMatch(i, i, 10.0) for i in range(len(points))]
    result = GeometryResult(
        verified=True, model="fundamental", tentative_matches=len(points),
        inlier_matches=len(points), inlier_ratio=1.0, median_reprojection_error_px=None,
        reason="geometrically_consistent", inlier_match_indices=tuple(range(len(points))),
    )

    summary = _spatial_distribution(left, right, matches, result)
    assert summary["left_image"]["occupied_grid_cells"] >= 8
    assert summary["right_image"]["occupied_grid_cells"] >= 8
    assert summary["warning"] is None



def test_spatial_deduplication_removes_repeated_locations_but_keeps_distinct_pairs():
    from types import SimpleNamespace

    from app.api.routes.matching import _deduplicate_spatial_inliers
    from app.engine.matching.models import DescriptorMatch

    left_points = (
        (100.0, 100.0),
        (101.0, 101.0),  # duplicate of point 0
        (300.0, 300.0),
        (500.0, 350.0),
    )
    right_points = (
        (200.0, 200.0),
        (201.0, 201.0),  # duplicate of point 0 on both images
        (400.0, 300.0),
        (550.0, 400.0),
    )
    left = SimpleNamespace(keypoints=left_points)
    right = SimpleNamespace(keypoints=right_points)
    matches = [
        DescriptorMatch(0, 0, 12.0),
        DescriptorMatch(1, 1, 8.0),  # best descriptor-distance representative
        DescriptorMatch(2, 2, 10.0),
        DescriptorMatch(3, 3, 11.0),
    ]

    unique = _deduplicate_spatial_inliers(left, right, matches, (0, 1, 2, 3))

    assert unique == (1, 2, 3)


def test_spatial_deduplication_requires_closeness_on_both_images():
    from types import SimpleNamespace

    from app.api.routes.matching import _deduplicate_spatial_inliers
    from app.engine.matching.models import DescriptorMatch

    left = SimpleNamespace(keypoints=((100.0, 100.0), (101.0, 101.0)))
    right = SimpleNamespace(keypoints=((200.0, 200.0), (220.0, 220.0)))
    matches = [
        DescriptorMatch(0, 0, 5.0),
        DescriptorMatch(1, 1, 6.0),
    ]

    assert _deduplicate_spatial_inliers(left, right, matches, (0, 1)) == (0, 1)


def test_spatial_distribution_reports_raw_and_unique_inliers():
    from types import SimpleNamespace

    from app.api.routes.matching import _spatial_distribution
    from app.engine.matching.models import DescriptorMatch, GeometryResult

    points = (
        (100.0, 100.0),
        (101.0, 101.0),
        (300.0, 100.0),
        (100.0, 300.0),
        (300.0, 300.0),
        (500.0, 100.0),
        (500.0, 300.0),
        (600.0, 400.0),
    )
    left = SimpleNamespace(image_width=640, image_height=480, keypoints=points)
    right = SimpleNamespace(image_width=640, image_height=480, keypoints=points)
    matches = [DescriptorMatch(i, i, float(i + 1)) for i in range(len(points))]
    result = GeometryResult(
        verified=True, model="fundamental", tentative_matches=8,
        inlier_matches=8, inlier_ratio=1.0, median_reprojection_error_px=None,
        reason="geometrically_consistent", inlier_match_indices=tuple(range(8)),
    )

    summary = _spatial_distribution(left, right, matches, result)

    assert summary["raw_inlier_matches"] == 8
    assert summary["unique_spatial_inlier_matches"] == 7
    assert summary["duplicate_spatial_inlier_matches_removed"] == 1
    assert summary["unique_inlier_fraction"] == 0.875
    assert summary["confidence_note"].startswith("Advisory")



def test_subject_region_support_is_advisory_and_does_not_change_geometry():
    from app.engine.matching.models import DescriptorMatch, GeometryResult
    from app.engine.matching.subject_region import pair_subject_support

    matches = [DescriptorMatch(i, i, 1.0) for i in range(8)]
    geometry = GeometryResult(
        verified=True, model="fundamental", tentative_matches=8,
        inlier_matches=8, inlier_ratio=1.0, median_reprojection_error_px=None,
        reason="geometrically_consistent", inlier_match_indices=tuple(range(8)),
    )
    left = {"labels": (True, True, False, False, False, False, False, False)}
    right = {"labels": (True, False, False, True, False, False, False, False)}

    support = pair_subject_support(matches, geometry, left, right)

    assert support["both_foreground"] == 1
    assert support["both_background"] == 5
    assert support["mixed_region"] == 2
    assert support["warning"] == "most_verified_matches_are_outside_proposed_subject_regions"
    assert geometry.verified is True


def test_classify_keypoints_uses_region_mask_coordinates():
    from types import SimpleNamespace

    import numpy as np

    from app.engine.matching.subject_region import classify_keypoints

    features = SimpleNamespace(
        feature_count=3,
        keypoints=((25.0, 25.0), (75.0, 25.0), (50.0, 75.0)),
    )
    region = {
        "status": "heuristic", "width": 100, "height": 100,
        "mask": np.array([[1, 0], [0, 1]], dtype=np.uint8),
    }

    result = classify_keypoints(features, region)

    assert result["foreground_candidate"] == 2
    assert result["background_candidate"] == 1
    assert result["foreground_fraction"] == 0.6667


def test_filter_features_to_region_keeps_only_foreground_descriptors():
    import numpy as np
    from app.engine.matching.models import ImageFeatures
    from app.engine.matching.subject_region import filter_features_to_region

    features = ImageFeatures(
        image_index=0,
        keypoints=((25.0, 25.0), (75.0, 25.0), (25.0, 75.0)),
        descriptors=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        image_width=100, image_height=100, scale_x=1.0, scale_y=1.0,
    )
    region = {
        "width": 100, "height": 100,
        "mask": np.array([[1, 0], [0, 1]], dtype=np.uint8),
    }

    filtered = filter_features_to_region(features, region)

    assert filtered.keypoints == ((25.0, 25.0),)
    assert filtered.descriptors.tolist() == [[1.0, 1.0]]
    assert filtered.image_width == 100 and filtered.image_height == 100



def test_manual_mask_match_filter_requires_both_endpoints_inside_masks():
    import numpy as np
    from types import SimpleNamespace

    from app.api.routes.matching import _inside_both_manual_masks
    from app.engine.matching.models import DescriptorMatch

    left = SimpleNamespace(keypoints=((10.0, 10.0), (80.0, 80.0)))
    right = SimpleNamespace(keypoints=((20.0, 20.0), (90.0, 90.0)))
    left_mask = np.zeros((100, 100), dtype=bool)
    right_mask = np.zeros((100, 100), dtype=bool)
    left_mask[5:30, 5:30] = True
    right_mask[15:35, 15:35] = True

    assert _inside_both_manual_masks(
        left, right, DescriptorMatch(0, 0, 1.0), left_mask, right_mask
    )
    assert not _inside_both_manual_masks(
        left, right, DescriptorMatch(1, 1, 1.0), left_mask, right_mask
    )


def test_object_only_matching_requires_saved_manual_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    upload = client.post(
        "/api/v1/analysis/images",
        files=[
            ("files", ("first.png", image_bytes(), "image/png")),
            ("files", ("second.png", image_bytes(), "image/png")),
        ],
    )
    analysis_id = upload.json()["analysis_id"]
    response = client.post(f"/api/v1/analysis/{analysis_id}/matching")
    assert response.status_code == 200
    pair = response.json()["pair_results"][0]
    assert pair["object_only"]["status"] == "missing_manual_mask"
    assert pair["object_only"]["reason"] == "manual_masks_required_for_both_images"



def test_object_only_reruns_geometry_using_saved_manual_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    upload = client.post(
        "/api/v1/analysis/images",
        files=[
            ("files", ("first.png", image_bytes(), "image/png")),
            ("files", ("second.png", image_bytes(), "image/png")),
        ],
    )
    analysis_id = upload.json()["analysis_id"]
    mask_dir = tmp_path.parent / "masks" / analysis_id
    mask_dir.mkdir(parents=True)
    full_foreground = Image.new("L", (640, 480), 255)
    full_foreground.save(mask_dir / "000.png")
    full_foreground.save(mask_dir / "001.png")

    response = client.post(f"/api/v1/analysis/{analysis_id}/matching")
    assert response.status_code == 200
    pair = response.json()["pair_results"][0]
    object_result = pair["object_only"]
    assert object_result["status"] == "complete"
    assert object_result["mask_sources"] == ["manual", "manual"]
    assert object_result["feature_extraction"]["method"] == "masked_sift_clahe_high_recall"
    assert object_result["feature_extraction"]["left_object_features"] > 0
    assert object_result["feature_extraction"]["right_object_features"] > 0
    assert object_result["tentative_matches"] == object_result["filtered_match_count"]
    assert len(object_result["diagnostic_correspondences"]) > 0



def test_correspondence_status_distinguishes_unverified_from_ransac_outlier():
    from types import SimpleNamespace

    from app.api.routes.matching import _pair_diagnostics
    from app.engine.matching.models import DescriptorMatch, GeometryResult

    left = SimpleNamespace(keypoints=((10.0, 10.0), (20.0, 20.0)))
    right = SimpleNamespace(keypoints=((11.0, 11.0), (21.0, 21.0)))
    matches = [DescriptorMatch(0, 0, 4.0), DescriptorMatch(1, 1, 5.0)]

    no_model = GeometryResult(
        False, "none", 2, 0, 0.0, None, "insufficient_matches", ()
    )
    rows = _pair_diagnostics(left, right, matches, no_model)
    assert [row["match_status"] for row in rows] == ["unverified", "unverified"]

    fitted_model = GeometryResult(
        False, "fundamental", 2, 1, 0.5, None, "insufficient_inliers", (0,)
    )
    rows = _pair_diagnostics(left, right, matches, fitted_model)
    assert [row["match_status"] for row in rows] == ["inlier", "outlier"]


def test_mask_filter_diagnostics_count_each_mask_independently():
    from types import SimpleNamespace

    import numpy as np

    from app.api.routes.matching import _mask_filter_diagnostics
    from app.engine.matching.models import DescriptorMatch

    left = SimpleNamespace(keypoints=((1.0, 1.0), (8.0, 8.0), (1.0, 8.0)))
    right = SimpleNamespace(keypoints=((1.0, 1.0), (1.0, 8.0), (8.0, 8.0)))
    left_mask = np.zeros((10, 10), dtype=bool)
    right_mask = np.zeros((10, 10), dtype=bool)
    left_mask[1, 1] = True
    right_mask[1, 1] = True
    right_mask[8, 8] = True
    matches = [
        DescriptorMatch(0, 0, 1.0),  # inside both
        DescriptorMatch(1, 1, 2.0),  # left outside, right inside
        DescriptorMatch(2, 2, 3.0),  # left inside? no; right outside
    ]

    result = _mask_filter_diagnostics(left, right, matches, left_mask, right_mask)
    assert result["total_descriptor_matches"] == 3
    assert result["eligible_by_left_mask"] == 1
    assert result["eligible_by_right_mask"] == 2
    assert result["eligible_by_both_masks"] == 1
    assert result["left_outside_right_inside"] == 1
    assert result["outside_both_masks"] == 1
    assert result["excluded_by_either_mask"] == 2

def test_mask_boundary_distances_measure_outside_keypoints():
    import numpy as np
    from types import SimpleNamespace

    from app.engine.matching.mask_boundary import (
        summarize_mask_boundary_distances,
    )
    from app.engine.matching.models import DescriptorMatch

    left = SimpleNamespace(
        keypoints=((2.0, 5.0), (5.0, 5.0), (9.0, 5.0))
    )
    right = SimpleNamespace(
        keypoints=((5.0, 2.0), (5.0, 5.0), (5.0, 9.0))
    )

    left_mask = np.zeros((12, 12), dtype=bool)
    right_mask = np.zeros((12, 12), dtype=bool)
    left_mask[4:7, 4:7] = True
    right_mask[4:7, 4:7] = True

    matches = [
        DescriptorMatch(0, 0, 1.0),
        DescriptorMatch(1, 1, 1.0),
        DescriptorMatch(2, 2, 1.0),
    ]

    result = summarize_mask_boundary_distances(
        left, right, matches, left_mask, right_mask
    )

    assert result["status"] == "complete"
    assert result["left_image"]["outside_mask_endpoints"] == 2
    assert result["right_image"]["outside_mask_endpoints"] == 2
    assert result["correspondences"]["at_least_one_outside_endpoint"] == 2
    assert result["correspondences"]["both_endpoints_outside"] == 2
    pair_bands = result["correspondences"]["distance_bands_using_nearest_endpoint"]
    assert pair_bands["0_to_2_px"] == 1
    assert pair_bands["over_2_to_5_px"] == 1


def test_mask_boundary_distances_report_out_of_image_and_invalid_indices():
    import numpy as np
    from types import SimpleNamespace

    from app.engine.matching.mask_boundary import (
        summarize_mask_boundary_distances,
    )
    from app.engine.matching.models import DescriptorMatch

    left = SimpleNamespace(keypoints=((-2.0, 5.0), (5.0, 5.0)))
    right = SimpleNamespace(keypoints=((5.0, 5.0), (5.0, 5.0)))

    mask = np.zeros((12, 12), dtype=bool)
    mask[4:7, 4:7] = True

    matches = [
        DescriptorMatch(0, 0, 1.0),
        DescriptorMatch(99, 1, 1.0),
    ]

    result = summarize_mask_boundary_distances(
        left, right, matches, mask, mask
    )

    assert result["left_image"]["out_of_image_endpoints"] == 1
    assert result["left_image"]["invalid_keypoint_indices"] == 1
    assert result["left_image"]["measurable_outside_endpoints"] == 0


def test_mask_boundary_distances_handle_empty_masks():
    import numpy as np
    from types import SimpleNamespace

    from app.engine.matching.mask_boundary import (
        summarize_mask_boundary_distances,
    )
    from app.engine.matching.models import DescriptorMatch

    image = SimpleNamespace(keypoints=((2.0, 2.0),))
    empty_mask = np.zeros((10, 10), dtype=bool)

    result = summarize_mask_boundary_distances(
        image,
        image,
        [DescriptorMatch(0, 0, 1.0)],
        empty_mask,
        empty_mask,
    )

    assert result["left_image"]["status"] == "no_foreground_pixels"
    assert result["right_image"]["status"] == "no_foreground_pixels"
    assert result["left_image"]["measurable_outside_endpoints"] == 0


def test_spatial_deduplication_handles_adjacent_hash_cells_and_exact_radius():
    from types import SimpleNamespace

    from app.api.routes.matching import _deduplicate_spatial_inliers
    from app.engine.matching.models import DescriptorMatch

    left = SimpleNamespace(keypoints=((4.9, 10.0), (5.1, 10.0), (20.0, 20.0)))
    right = SimpleNamespace(keypoints=((9.9, 10.0), (10.1, 10.0), (30.0, 30.0)))
    matches = [
        DescriptorMatch(0, 0, 4.0),
        DescriptorMatch(1, 1, 5.0),  # across a 5px hash-cell boundary; same location
        DescriptorMatch(2, 2, 6.0),
    ]
    assert _deduplicate_spatial_inliers(left, right, matches, (0, 1, 2)) == (0, 2)

    left = SimpleNamespace(keypoints=((10.0, 10.0), (15.0, 10.0)))
    right = SimpleNamespace(keypoints=((20.0, 20.0), (25.0, 20.0)))
    matches = [DescriptorMatch(0, 0, 1.0), DescriptorMatch(1, 1, 2.0)]
    assert _deduplicate_spatial_inliers(left, right, matches, (0, 1)) == (0,)


def test_object_only_result_includes_its_own_spatial_distribution(tmp_path, monkeypatch):
    from pathlib import Path

    from PIL import Image

    from app.core.config import settings

    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    upload_dir = settings.upload_dir
    upload = client.post(
        "/api/v1/analysis/images",
        files=[
            ("files", ("first.png", image_bytes(), "image/png")),
            ("files", ("second.png", image_bytes(), "image/png")),
        ],
    )
    assert upload.status_code == 200
    analysis_id = upload.json()["analysis_id"]

    mask_dir = Path(upload_dir).parent / "masks" / analysis_id
    mask_dir.mkdir(parents=True)
    for index in range(2):
        Image.new("L", (640, 480), 255).save(mask_dir / f"{index:03d}.png")

    response = client.post(f"/api/v1/analysis/{analysis_id}/matching")
    assert response.status_code == 200
    object_only = response.json()["pair_results"][0]["object_only"]
    assert object_only["status"] == "complete"
    assert "spatial_distribution" in object_only
    assert object_only["spatial_distribution"]["raw_inlier_matches"] == object_only["inlier_matches"]


def test_spatial_hash_dedup_matches_pairwise_reference_on_seeded_data():
    import numpy as np
    from types import SimpleNamespace

    from app.api.routes.matching import _deduplicate_spatial_inliers
    from app.engine.matching.models import DescriptorMatch

    rng = np.random.default_rng(20261101)
    left_points = []
    right_points = []
    matches = []
    for index in range(180):
        if index and index % 3:
            source = int(rng.integers(0, len(left_points)))
            left_base = left_points[source]
            right_base = right_points[source]
            left_point = tuple(
                float(value) + float(rng.uniform(-4.9, 4.9)) for value in left_base
            )
            right_point = tuple(
                float(value) + float(rng.uniform(-4.9, 4.9)) for value in right_base
            )
        else:
            left_point = tuple(float(value) for value in rng.uniform(0, 1000, 2))
            right_point = tuple(float(value) for value in rng.uniform(0, 1000, 2))
        left_points.append(left_point)
        right_points.append(right_point)
        matches.append(DescriptorMatch(index, index, float(rng.uniform(0, 100))))

    left = SimpleNamespace(keypoints=tuple(left_points))
    right = SimpleNamespace(keypoints=tuple(right_points))
    expected = []
    for index in sorted(range(len(matches)), key=lambda item: (matches[item].distance, item)):
        candidate_left = left.keypoints[index]
        candidate_right = right.keypoints[index]
        duplicate = False
        for kept_index in expected:
            kept_left = left.keypoints[kept_index]
            kept_right = right.keypoints[kept_index]
            left_d2 = sum((a - b) ** 2 for a, b in zip(candidate_left, kept_left))
            right_d2 = sum((a - b) ** 2 for a, b in zip(candidate_right, kept_right))
            if left_d2 <= 25.0 and right_d2 <= 25.0:
                duplicate = True
                break
        if not duplicate:
            expected.append(index)

    assert _deduplicate_spatial_inliers(left, right, matches, range(len(matches))) == tuple(expected)

def test_manual_mask_loader_uses_exif_oriented_dimensions(tmp_path, monkeypatch):
    from PIL import Image
    from app.api.routes.matching import _load_manual_mask
    from app.core.config import settings
    from uuid import uuid4
    import numpy as np

    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = uuid4()
    image_dir = settings.upload_dir / str(analysis_id)
    image_dir.mkdir(parents=True)
    image_path = image_dir / "001_0123456789ab.jpg"
    source = Image.new("RGB", (64, 48), "white")
    exif = source.getexif()
    exif[274] = 6
    source.save(image_path, format="JPEG", exif=exif)

    mask_path = tmp_path / "masks" / str(analysis_id) / "000.png"
    mask_path.parent.mkdir(parents=True)
    Image.new("L", (48, 64), 255).save(mask_path)

    mask = _load_manual_mask(analysis_id, 0, image_path)
    assert mask is not None
    assert mask.shape == (64, 48)
    assert bool(np.all(mask))

