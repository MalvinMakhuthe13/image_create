from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.api.routes import matching as matching_route


def _textured_image(seed: int = 7, size: int = 720) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    for _ in range(24):
        point = (
            int(rng.integers(40, size - 40)),
            int(rng.integers(40, size - 40)),
        )
        cv2.circle(image, point, int(rng.integers(8, 24)), (255, 255, 255), 2)
    return image


def test_matching_uses_accepted_automatic_masks_without_manual_painting(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = "0c4a4e5b-8e88-4d0f-8d7d-93f9c6b0d4b8"
    folder = settings.upload_dir / analysis_id
    folder.mkdir(parents=True)
    first = _textured_image()
    transform = cv2.getRotationMatrix2D((360, 360), 3, 0.99)
    second = cv2.warpAffine(first, transform, (720, 720))
    paths = [folder / "001_0123456789ab.png", folder / "002_abcdef012345.png"]
    Image.fromarray(first).save(paths[0])
    Image.fromarray(second).save(paths[1])
    original_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ]

    def fake_segmentation(image_path):
        with Image.open(image_path) as source:
            width, height = source.size
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (width // 2, height // 2), int(width * 0.46), 255, -1)
        return {
            "status": "automatic",
            "method": "test_model",
            "mask": mask,
            "plausible": True,
            "foreground_fraction": float((mask > 0).mean()),
        }

    monkeypatch.setattr(matching_route, "generate_subject_mask", fake_segmentation)
    response = TestClient(app).post(f"/api/v1/analysis/{analysis_id}/matching")
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["images"][0]["subject_mask"]["status"] == "automatic"
    pair = report["pair_results"][0]
    assert pair["object_only"]["mask_sources"] == ["automatic", "automatic"]
    assert pair["object_only"]["status"] == "complete"
    assert pair["object_only"]["feature_extraction"]["ratio_threshold"] == 0.78
    for path, original_hash in zip(paths, original_hashes):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    cached = settings.upload_dir.parent / "automatic_masks" / analysis_id / "000.png"
    assert cached.is_file()



def test_low_confidence_fallback_is_not_used_for_object_only_ransac(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = "2a5b4e3f-6c7d-4b8a-9c1d-0e2f3a4b5c6d"
    folder = settings.upload_dir / analysis_id
    folder.mkdir(parents=True)
    first = _textured_image(seed=13)
    second = cv2.warpAffine(
        first, cv2.getRotationMatrix2D((360, 360), 2, 1.0), (720, 720)
    )
    for index, image in enumerate((first, second), start=1):
        Image.fromarray(image).save(
            folder / f"{index:03d}_{index:012x}.png"
        )

    def low_confidence_fallback(image_path):
        return {
            "status": "low_confidence",
            "method": "grabcut_heuristic_fallback",
            "mask": np.full((720, 720), 255, dtype=np.uint8),
            "plausible": True,
            "foreground_fraction": 1.0,
        }

    monkeypatch.setattr(
        matching_route, "generate_subject_mask", low_confidence_fallback
    )
    response = TestClient(app).post(
        f"/api/v1/analysis/{analysis_id}/matching"
    )
    assert response.status_code == 200, response.text
    pair = response.json()["pair_results"][0]
    assert pair["object_only"]["status"] == "missing_automatic_or_manual_mask"
    assert pair["object_only"]["mask_sources"] == ["unavailable", "unavailable"]
    assert pair["verified"] in (True, False)  # all-scene geometry remains independent
