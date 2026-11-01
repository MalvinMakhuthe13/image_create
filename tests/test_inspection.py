from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app


def _make_upload(folder, name="001_0123456789ab.jpg", size=(64, 48)):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(folder / name, format="JPEG")


def test_inspection_workspace_and_image_route(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = uuid4()
    folder = settings.upload_dir / str(analysis_id)
    _make_upload(folder)
    client = TestClient(app)

    page = client.get(f"/api/v1/analysis/{analysis_id}/inspection")
    assert page.status_code == 200
    assert str(analysis_id) in page.text
    image = client.get(f"/api/v1/analysis/{analysis_id}/images/0")
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/")


def test_manual_mask_save_and_dimensions_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = uuid4()
    folder = settings.upload_dir / str(analysis_id)
    _make_upload(folder)
    client = TestClient(app)

    mask_bytes = BytesIO()
    mask = Image.new("L", (64, 48), 0)
    for x in range(10, 40):
        for y in range(8, 36):
            mask.putpixel((x, y), 255)
    mask.save(mask_bytes, format="PNG")
    response = client.put(
        f"/api/v1/analysis/{analysis_id}/masks/0",
        files={"file": ("mask.png", mask_bytes.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"
    assert 0 < response.json()["foreground_fraction"] < 1

    saved = client.get(f"/api/v1/analysis/{analysis_id}/masks/0")
    assert saved.status_code == 200
    assert saved.headers["content-type"] == "image/png"

    wrong = BytesIO()
    Image.new("L", (20, 20), 255).save(wrong, format="PNG")
    rejected = client.put(
        f"/api/v1/analysis/{analysis_id}/masks/0",
        files={"file": ("wrong.png", wrong.getvalue(), "image/png")},
    )
    assert rejected.status_code == 422


def test_mask_routes_reject_unknown_image_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = uuid4()
    _make_upload(settings.upload_dir / str(analysis_id))
    client = TestClient(app)
    assert client.get(f"/api/v1/analysis/{analysis_id}/images/9").status_code == 404
    assert client.get(f"/api/v1/analysis/{analysis_id}/masks/9").status_code == 404
