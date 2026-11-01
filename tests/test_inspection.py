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

def test_mask_and_calibration_dimensions_follow_exif_orientation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = uuid4()
    folder = settings.upload_dir / str(analysis_id)
    folder.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (64, 48), "white")
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    filename = "001_0123456789ab.jpg"
    image.save(folder / filename, format="JPEG", exif=exif)

    client = TestClient(app)
    mask_bytes = BytesIO()
    Image.new("L", (48, 64), 255).save(mask_bytes, format="PNG")
    saved_mask = client.put(
        f"/api/v1/analysis/{analysis_id}/masks/0",
        files={"file": ("mask.png", mask_bytes.getvalue(), "image/png")},
    )
    assert saved_mask.status_code == 200
    assert saved_mask.json()["width"] == 48
    assert saved_mask.json()["height"] == 64

    calibration = client.put(
        f"/api/v1/analysis/{analysis_id}/calibration/0",
        json={
            "camera_matrix": [
                [500.0, 0.0, 24.0],
                [0.0, 500.0, 32.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [],
        },
    )
    assert calibration.status_code == 200
    assert calibration.json()["dimensions"] == {"width": 48, "height": 64}




def test_automatic_mask_route_serves_cached_mask_only(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    analysis_id = uuid4()
    folder = settings.upload_dir / str(analysis_id)
    _make_upload(folder)
    client = TestClient(app)

    missing = client.get(f"/api/v1/analysis/{analysis_id}/automatic-masks/0")
    assert missing.status_code == 404

    mask_path = settings.upload_dir.parent / "automatic_masks" / str(analysis_id) / "000.png"
    mask_path.parent.mkdir(parents=True)
    Image.new("L", (64, 48), 0).save(mask_path)
    response = client.get(f"/api/v1/analysis/{analysis_id}/automatic-masks/0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
