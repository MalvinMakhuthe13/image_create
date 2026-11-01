from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_job_returns_queued_without_fake_frame_count():
    response = client.post(
        "/api/v1/jobs",
        json={"generation_mode": "both", "budget": {"maximum_frames": 80}},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"
    assert "planned_frame_count" not in body


def test_unknown_job_returns_404():
    import uuid

    response = client.get(f"/api/v1/jobs/{uuid.uuid4()}")
    assert response.status_code == 404
