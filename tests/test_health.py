from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_root_serves_capture_studio():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Analyse capture set" in response.text
    assert "Open visual inspection workspace" in response.text
    assert "coverage as not assessed" in response.text

