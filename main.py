from __future__ import annotations

import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.lego import dimensions, greedy_pack
from app.manual import build_pdf, generate_copy
from app.stable_fast_3d import reconstruct
from app.store import store


load_dotenv()
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", tempfile.gettempdir())) / "bricked-up"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = ROOT / "static"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKER_THREADS", "2")))

app = FastAPI(title="Bricked Up", version="1.0.0")


def _public(job: dict) -> dict:
    hidden = {"image_path", "work_dir", "manual_path", "model_path"}
    return {key: value for key, value in job.items() if key not in hidden}


def _job_or_404(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Build not found")
    return job


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as source:
            source.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a readable image") from exc


def _run_reconstruction(job_id: str) -> None:
    job = store.get(job_id)
    if not job:
        return
    try:
        store.update(job_id, stage="reconstructing", progress=8, message="Preparing image")

        def update(progress: int, message: str) -> None:
            store.update(job_id, progress=progress, message=message)

        reconstruction = reconstruct(Path(job["image_path"]), Path(job["work_dir"]), update)
        # Accept the old two-value shape for local test doubles and extensions.
        voxels, mode = reconstruction[:2]
        model_path = reconstruction[2] if len(reconstruction) > 2 else None
        store.update(
            job_id,
            stage="model_ready",
            progress=100,
            message="Voxel model ready",
            mode=mode,
            model_path=str(model_path) if model_path else None,
            model_url=f"/api/jobs/{job_id}/model.glb" if model_path else None,
            model_voxels=voxels,
            model_dimensions=dimensions(voxels),
        )
    except Exception as exc:
        store.update(job_id, stage="failed", message="Reconstruction failed", error=str(exc))


def _run_legolize(job_id: str) -> None:
    job = store.get(job_id)
    if not job:
        return
    try:
        store.update(job_id, stage="legolizing", progress=18, message="Finding the largest bricks that fit")
        bricks, bom = greedy_pack(job.get("model_voxels", []))
        store.update(
            job_id,
            stage="lego_ready",
            progress=100,
            message="Brick model ready",
            lego_bricks=bricks,
            piece_count=len(bricks),
            bill_of_materials=bom,
        )
    except Exception as exc:
        store.update(job_id, stage="failed", message="Brick conversion failed", error=str(exc))


def _run_manual(job_id: str) -> None:
    job = store.get(job_id)
    if not job:
        return
    try:
        store.update(job_id, stage="manual_generating", progress=30, message="Writing build guidance")
        copy = generate_copy(job["lego_bricks"], job["bill_of_materials"])
        path = Path(job["work_dir"]) / "instructions.pdf"
        build_pdf(path, job["lego_bricks"], job["bill_of_materials"], copy)
        store.update(
            job_id,
            stage="complete",
            progress=100,
            message="Instructions ready",
            manual_ready=True,
            manual_path=str(path),
        )
    except Exception as exc:
        store.update(job_id, stage="failed", message="Manual generation failed", error=str(exc))


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "reconstruction": "demo" if os.getenv("SF3D_DEMO_MODE", "").lower() in {"1", "true", "yes"} else "stable-fast-3d",
        "manual_copy": "gemini" if os.getenv("GEMINI_API_KEY") else "local",
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    image: Annotated[UploadFile, File(...)],
) -> dict:
    suffix = Path(image.filename or "object.png").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, JPEG, or WebP image")
    job_id = uuid.uuid4().hex
    work_dir = DATA_DIR / job_id
    work_dir.mkdir(parents=True)
    image_path = work_dir / f"input{suffix}"
    size = 0
    with image_path.open("wb") as output:
        while chunk := await image.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                output.close()
                image_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
            output.write(chunk)
    try:
        _validate_image(image_path)
    except HTTPException:
        image_path.unlink(missing_ok=True)
        raise
    job = store.create(
        {
            "id": job_id,
            "stage": "queued",
            "progress": 0,
            "message": "Build queued",
            "mode": "pending",
            "model_voxels": [],
            "model_dimensions": {},
            "lego_bricks": [],
            "piece_count": 0,
            "bill_of_materials": {},
            "manual_ready": False,
            "error": None,
            "image_path": str(image_path),
            "work_dir": str(work_dir),
            "manual_path": None,
            "model_path": None,
            "model_url": None,
        }
    )
    executor.submit(_run_reconstruction, job_id)
    return _public(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _public(_job_or_404(job_id))


@app.post("/api/jobs/{job_id}/legolize", status_code=202)
def legolize(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job["stage"] not in {"model_ready", "lego_ready", "complete"}:
        raise HTTPException(status_code=409, detail="The voxel model is not ready")
    if job["stage"] == "model_ready":
        store.update(job_id, stage="legolizing", progress=0, message="Brick conversion queued")
        executor.submit(_run_legolize, job_id)
    return _public(store.get(job_id))


@app.post("/api/jobs/{job_id}/manual", status_code=202)
def create_manual(job_id: str) -> dict:
    job = _job_or_404(job_id)
    if job["stage"] not in {"lego_ready", "complete"}:
        raise HTTPException(status_code=409, detail="The brick model is not ready")
    if not job["manual_ready"]:
        store.update(job_id, stage="manual_generating", progress=0, message="Manual generation queued")
        executor.submit(_run_manual, job_id)
    return _public(store.get(job_id))


@app.get("/api/jobs/{job_id}/manual.pdf")
def download_manual(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    path = Path(job["manual_path"]) if job.get("manual_path") else None
    if not path or not path.exists():
        raise HTTPException(status_code=409, detail="The instruction manual is not ready")
    return FileResponse(path, media_type="application/pdf", filename=f"brick-build-{job_id[:8]}.pdf")


@app.get("/api/jobs/{job_id}/model.glb")
def download_model(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    path = Path(job["model_path"]) if job.get("model_path") else None
    if not path or not path.exists():
        raise HTTPException(status_code=409, detail="The generated 3D model is not ready")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"generated-model-{job_id[:8]}.glb")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="spa")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
