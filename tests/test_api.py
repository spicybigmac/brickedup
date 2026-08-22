import time
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import main


def _wait(client: TestClient, job_id: str, stage: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] == stage:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job never reached {stage}: {job}")


def test_demo_pipeline_reaches_pdf(monkeypatch):
    monkeypatch.setattr(
        main,
        "reconstruct",
        lambda _image, _dir, _update: (
            [
                {"x": x, "y": y, "z": 0, "color": "#C84832"}
                for y in range(2)
                for x in range(4)
            ],
            "demo",
        ),
    )
    monkeypatch.setattr(
        main,
        "generate_copy",
        lambda _bricks, _bom: {
            "title": "Test Build",
            "intro": "Build it.",
            "safety": "Small parts.",
            "tips": ["Sort.", "Align.", "Press."],
        },
    )
    image_bytes = BytesIO()
    Image.new("RGB", (64, 64), "#C84832").save(image_bytes, format="PNG")
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/styles.css").status_code == 200
        assert client.get("/app.js").status_code == 200
        assert client.get("/favicon.svg").status_code == 200
        assert client.get("/favicon.ico").status_code == 200
        response = client.post(
            "/api/jobs",
            files={"image": ("object.png", image_bytes.getvalue(), "image/png")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        _wait(client, job_id, "model_ready")

        response = client.post(f"/api/jobs/{job_id}/legolize")
        assert response.status_code == 202
        assert response.json()["stage"] in {"legolizing", "lego_ready"}
        lego = _wait(client, job_id, "lego_ready")
        assert lego["piece_count"] == 1

        response = client.post(f"/api/jobs/{job_id}/manual")
        assert response.status_code == 202
        assert response.json()["stage"] in {"manual_generating", "complete"}
        _wait(client, job_id, "complete")
        pdf = client.get(f"/api/jobs/{job_id}/manual.pdf")
        assert pdf.status_code == 200
        assert pdf.content.startswith(b"%PDF")


def test_upload_rejects_non_image_content():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/jobs",
            files={"image": ("object.png", b"not-an-image", "image/png")},
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a readable image"


def test_generated_model_is_exposed_without_leaking_local_path(monkeypatch, tmp_path):
    model = tmp_path / "generated.glb"
    model.write_bytes(b"glTF-test")
    monkeypatch.setattr(
        main,
        "reconstruct",
        lambda _image, _dir, _update: (
            [{"x": 0, "y": 0, "z": 0, "color": "#3366CC"}],
            "stable-fast-3d",
            model,
        ),
    )
    image_bytes = BytesIO()
    Image.new("RGB", (32, 32), "#3366CC").save(image_bytes, format="PNG")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/jobs",
            files={"image": ("object.png", image_bytes.getvalue(), "image/png")},
        )
        job = _wait(client, response.json()["id"], "model_ready")
        assert job["model_url"].endswith("/model.glb")
        assert "model_path" not in job
        downloaded = client.get(job["model_url"])
        assert downloaded.status_code == 200
        assert downloaded.content == b"glTF-test"
