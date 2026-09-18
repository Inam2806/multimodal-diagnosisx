from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from multimodal_diagnosis.api import app, xray_service


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_non_image_content_type() -> None:
    response = client.post(
        "/v1/diagnose/xray",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_invalid_image_is_rejected_without_loading_model() -> None:
    response = client.post(
        "/v1/diagnose/xray",
        files={"image": ("image.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400


def test_valid_image_reaches_inference(monkeypatch) -> None:
    image = Image.new("L", (4, 4), color=128)
    payload = BytesIO()
    image.save(payload, format="PNG")

    monkeypatch.setattr(
        xray_service.model,
        "predict",
        lambda _: {"Finding A": 0.9, "Finding B": 0.1},
    )
    response = client.post(
        "/v1/diagnose/xray",
        files={"image": ("image.png", payload.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["label"] == "Finding A"
