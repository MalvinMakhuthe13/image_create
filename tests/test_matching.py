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
