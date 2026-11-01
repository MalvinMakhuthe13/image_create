from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import settings
from app.main import app

client = TestClient(app)


def png_bytes():
    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 800, 600), fill="navy")
    draw.ellipse((250, 180, 650, 580), fill="orange")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_analyzes_images_and_groups_exact_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    payload = png_bytes()
    response = client.post("/api/v1/analysis/images", files=[
        ("files", ("front.png", payload, "image/png")),
        ("files", ("front-copy.png", payload, "image/png")),
    ])
    assert response.status_code == 200
    body = response.json()
    assert body["processing"] == {"mode": "cpu_local", "gpu_required": False}
    assert body["summary"]["images_received"] == 2
    assert body["summary"]["exact_duplicate_groups"] == 1
    assert body["coverage_assessment"]["status"] == "not_assessed"
    assert all(item["metrics"]["width"] == 900 for item in body["images"])
    assert all((tmp_path / body["analysis_id"] / item["stored_filename"]).exists() for item in body["images"])


def test_upload_rejects_non_image(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    response = client.post("/api/v1/analysis/images", files=[("files", ("notes.txt", b"not image", "text/plain"))])
    assert response.status_code == 415


def test_upload_rejects_corrupt_image(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    response = client.post("/api/v1/analysis/images", files=[("files", ("broken.png", b"not png", "image/png"))])
    assert response.status_code == 400


def test_failed_multi_image_upload_removes_partial_analysis_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    response = client.post(
        "/api/v1/analysis/images",
        files=[
            ("files", ("valid.png", png_bytes(), "image/png")),
            ("files", ("broken.png", b"not an image", "image/png")),
        ],
    )
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_non_development_api_requires_configured_api_key(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "api_key", "test-secret-key")
    denied = client.get("/api/v1/jobs")
    assert denied.status_code == 401
    accepted = client.get("/api/v1/health", headers={"X-API-Key": "test-secret-key"})
    assert accepted.status_code == 200


def test_non_development_api_fails_closed_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "api_key", None)
    response = client.get("/api/v1/health")
    assert response.status_code == 503
