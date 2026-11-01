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


def test_matching_endpoint_returns_404_for_unknown_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    response = client.post(f"/api/v1/analysis/{uuid4()}/matching")
    assert response.status_code == 404


def test_candidate_pair_scheduler_keeps_adjacent_pairs():
    from app.api.routes.matching import _candidate_pairs

    pairs = _candidate_pairs(6, 8)
    assert all((index, index + 1) in pairs for index in range(5))
    assert len(pairs) == 8
