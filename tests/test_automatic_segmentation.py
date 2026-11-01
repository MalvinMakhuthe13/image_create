from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from app.engine.matching import automatic_segmentation as segmentation


def test_model_mask_is_resized_to_full_oriented_source_dimensions(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    model_mask = np.zeros((1024, 1024), dtype=np.uint8)
    model_mask[40:160, 40:200] = 255
    monkeypatch.setattr(segmentation, "_rembg_mask", lambda rgb: model_mask[:rgb.shape[0], :rgb.shape[1]])
    result = segmentation.generate_subject_mask(image_path, max_dimension=256)
    assert result["status"] == "automatic"
    assert result["method"] == "u2net_rembg_onnx_cpu"
    assert result["mask"].shape == (240, 320)
    assert set(np.unique(result["mask"])).issubset({0, 255})


def test_missing_model_uses_explicit_low_confidence_fallback(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (240, 180), "white").save(image_path)
    monkeypatch.setattr(
        segmentation, "_rembg_mask",
        lambda rgb: (_ for _ in ()).throw(ImportError("rembg not installed")),
    )
    monkeypatch.setattr(
        segmentation, "_grabcut_fallback",
        lambda rgb: np.pad(
            np.full((rgb.shape[0] - 40, rgb.shape[1] - 40), 255, dtype=np.uint8),
            ((20, 20), (20, 20)),
        ),
    )
    result = segmentation.generate_subject_mask(image_path)
    assert result["status"] == "low_confidence"
    assert result["method"] == "grabcut_heuristic_fallback"
    assert result["mask"].shape == (180, 240)
    assert "ImportError" in result["model_error"]


def test_model_failure_can_be_configured_to_fail_closed(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (240, 180), "white").save(image_path)
    monkeypatch.setattr(
        segmentation, "_rembg_mask",
        lambda rgb: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    result = segmentation.generate_subject_mask(
        image_path, allow_heuristic_fallback=False
    )
    assert result["status"] == "unavailable"
    assert result["mask"] is None


def test_saved_mask_is_separate_atomic_png(tmp_path):
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:20, 5:25] = 255
    target = tmp_path / "automatic_masks" / "000.png"
    segmentation.save_generated_mask(mask, target)
    with Image.open(target) as saved:
        assert saved.mode == "L"
        assert np.array_equal(np.asarray(saved), mask)
    assert not list(target.parent.glob(".*.tmp.png"))



@pytest.mark.skipif(
    os.getenv("IMAGE_CREATE_RUN_MODEL_SMOKE") != "1",
    reason="real U2-Net CPU smoke test runs in the dedicated optional-model CI job",
)
def test_real_u2net_model_runs_on_cpu_without_heuristic_fallback(tmp_path):
    image_path = tmp_path / "salient-object.png"
    image = Image.new("RGB", (512, 512), (245, 245, 245))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.ellipse((112, 82, 400, 430), fill=(25, 70, 130))
    draw.rectangle((190, 150, 330, 190), fill=(235, 190, 35))
    image.save(image_path)

    result = segmentation.generate_subject_mask(
        image_path, max_dimension=512, allow_heuristic_fallback=False
    )
    assert result["status"] == "automatic"
    assert result["method"] == "u2net_rembg_onnx_cpu"
    assert result["mask"].shape == (512, 512)
    assert 0.005 <= result["foreground_fraction"] <= 0.98
