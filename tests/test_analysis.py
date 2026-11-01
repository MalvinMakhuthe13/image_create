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


def test_image_quality_report_includes_blur_clipping_and_entropy_metrics(tmp_path):
    from app.engine.analysis.image_analyzer import analyze_image

    image_path = tmp_path / "quality.png"
    Image.new("RGB", (900, 700), (255, 255, 255)).save(image_path)
    metrics, extension = analyze_image(image_path, 40_000_000)
    report = metrics.to_dict()
    assert extension == ".png"
    assert report["laplacian_variance"] >= 0
    assert report["edge_directionality"] >= 0
    assert 0 <= report["clipped_fraction"] <= 1
    assert 0 <= report["histogram_entropy_bits"] <= 8
    assert "significant_highlight_or_shadow_clipping" in report["quality_flags"]


def test_sqlite_job_store_survives_store_reinstantiation(tmp_path):
    from uuid import uuid4

    from app.schemas.jobs import JobResponse
    from app.services.job_store import PersistentJobStore

    database = tmp_path / "jobs.sqlite3"
    first_store = PersistentJobStore(database)
    job_id = uuid4()
    job = JobResponse(
        id=job_id,
        status="queued",
        generation_mode="verified",
        message="Persisted test job",
    )
    first_store[job_id] = job

    restarted_store = PersistentJobStore(database)
    restored = restarted_store.get(job_id)
    assert restored is not None
    assert restored.id == job_id
    assert restored.status == "queued"
    assert restored.generation_mode == "verified"
    assert len(restarted_store) == 1
