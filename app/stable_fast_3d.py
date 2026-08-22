from __future__ import annotations

import os
import random
import re
import shutil
import time
from html import unescape
from pathlib import Path
from typing import Any, Callable

import numpy as np
import httpx
import trimesh
from gradio_client import Client, handle_file
from PIL import Image
from scipy.spatial import cKDTree


PALETTE = ["#C84832", "#E8B843", "#315F9D", "#EEE8DA"]
DEFAULT_SPACE = "Upsampler/stable-fast-3d"
FALLBACK_SPACE = "stabilityai/stable-fast-3d"
DEFAULT_TARGET_STUDS = 32
DEFAULT_MAX_VOXELS = 35_000
PROVIDER_HUGGINGFACE = "huggingface"
PROVIDER_NGROK = "ngrok"


def _demo_voxels(seed: str) -> list[dict[str, Any]]:
    """Create a deterministic toy robot when explicit offline mode is enabled."""
    rng = random.Random(seed)
    cells: set[tuple[int, int, int, str]] = set()

    def box(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, color: str) -> None:
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    cells.add((x, y, z, color))

    box(4, 12, 3, 9, 3, 10, PALETTE[0])
    box(5, 11, 4, 8, 10, 15, PALETTE[1])
    box(2, 4, 4, 8, 5, 10, PALETTE[2])
    box(12, 14, 4, 8, 5, 10, PALETTE[2])
    box(5, 8, 4, 8, 0, 3, PALETTE[2])
    box(8, 11, 4, 8, 0, 3, PALETTE[2])
    box(6, 7, 3, 4, 12, 13, "#1C1C1A")
    box(9, 10, 3, 4, 12, 13, "#1C1C1A")
    if rng.random() > -1:
        box(7, 9, 3, 4, 10, 11, "#EEE8DA")
    return [
        {"x": x, "y": y, "z": z, "color": color}
        for x, y, z, color in sorted(cells)
    ]


def _mesh_from_file(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
        if not isinstance(loaded, trimesh.Trimesh):
            raise RuntimeError("The generated scene contains no mesh geometry")
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError("Stable Fast 3D returned unsupported geometry")
    return loaded


def _rgb_from_image(image: Image.Image, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample normalized UV coordinates from a Pillow image."""
    source = image.convert("RGBA")
    pixels = np.asarray(source)
    x = np.clip(np.rint(u * (source.width - 1)), 0, source.width - 1).astype(int)
    y = np.clip(np.rint(v * (source.height - 1)), 0, source.height - 1).astype(int)
    rgba = pixels[y, x].astype(float)
    alpha = rgba[:, 3:4] / 255.0
    # Transparent input pixels are composited onto the app's off-white canvas.
    return np.rint(rgba[:, :3] * alpha + np.array([243, 239, 230]) * (1 - alpha)).astype(np.uint8)


def _surface_colors(mesh: trimesh.Trimesh, points: np.ndarray, source_image_path: Path | None) -> np.ndarray:
    """Transfer the generated mesh texture (or input image) onto voxel centers."""
    closest, _distance, face_ids = trimesh.proximity.closest_point(mesh, points)
    triangles = np.asarray(mesh.triangles)[face_ids]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, closest)
    face_vertices = np.asarray(mesh.faces)[face_ids]
    visual = mesh.visual

    if getattr(visual, "kind", None) == "texture" and getattr(visual, "uv", None) is not None:
        material = getattr(visual, "material", None)
        texture = getattr(material, "image", None)
        # Stable Fast 3D exports a GLTF PBR material. Trimesh exposes its color
        # atlas as baseColorTexture rather than the legacy `image` attribute.
        if texture is None:
            texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            texture = getattr(material, "diffuseTexture", None)
        triangle_uv = np.asarray(visual.uv)[face_vertices]
        uv = np.einsum("ni,nij->nj", barycentric, triangle_uv)
        if texture is not None and len(uv) == len(points):
            if not isinstance(texture, Image.Image):
                texture = Image.fromarray(np.asarray(texture))
            # GLTF UV origin is bottom-left; Pillow's is top-left.
            return _rgb_from_image(texture, uv[:, 0], 1.0 - uv[:, 1])

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        triangle_colors = np.asarray(vertex_colors)[face_vertices, :3].astype(float)
        colors = np.einsum("ni,nij->nj", barycentric, triangle_colors)
        # Trimesh's default ColorVisuals is uniform gray; prefer the user's image.
        if np.ptp(colors.astype(float), axis=0).max() > 4 or source_image_path is None:
            return np.clip(np.rint(colors), 0, 255).astype(np.uint8)

    if source_image_path and source_image_path.exists():
        with Image.open(source_image_path) as source:
            bounds = mesh.bounds
            span = np.maximum(bounds[1] - bounds[0], 1e-9)
            u = (points[:, 0] - bounds[0, 0]) / span[0]
            v = 1.0 - (points[:, 1] - bounds[0, 1]) / span[1]
            return _rgb_from_image(source, u, v)

    return np.tile(np.array([200, 72, 50], dtype=np.uint8), (len(points), 1))


def _quantize_colors(rgb: np.ndarray, color_count: int = 32) -> list[str]:
    """Keep image fidelity while limiting colors enough for useful brick packing."""
    strip = Image.fromarray(rgb.reshape(1, -1, 3), mode="RGB")
    quantized = strip.quantize(colors=min(color_count, len(rgb)), method=Image.Quantize.MEDIANCUT).convert("RGB")
    values = np.asarray(quantized).reshape(-1, 3)
    return [f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in values]


def _bounded_resolution(extents: np.ndarray, target_studs: int, max_voxels: int) -> int:
    """Choose a resolution whose complete bounding grid fits the voxel budget."""
    target = max(1, int(target_studs))
    budget = max(1, int(max_voxels))
    normalized = np.asarray(extents, dtype=float) / max(float(np.max(extents)), 1e-9)
    for resolution in range(target, 0, -1):
        # Voxel centers include both ends of a mesh extent. This is deliberately
        # an upper bound: occupied cells can never exceed the enclosing grid.
        grid_shape = np.ceil(normalized * resolution).astype(np.int64) + 1
        if int(np.prod(grid_shape)) <= budget:
            return resolution
    return 1


def _voxel_colors(
    mesh: trimesh.Trimesh,
    surface_points: np.ndarray,
    filled_points: np.ndarray,
    source_image_path: Path | None,
) -> list[str]:
    """Color surface cells precisely, then cheaply propagate into the interior."""
    if not len(surface_points):
        surface_points = filled_points
    surface_rgb = _surface_colors(mesh, surface_points, source_image_path)
    if len(surface_points) == len(filled_points) and np.array_equal(surface_points, filled_points):
        rgb = surface_rgb
    else:
        # Interior cells have no visible mesh surface to sample. Give each one
        # the color of its nearest surface cell instead of running thousands of
        # additional nearest-triangle queries against the high-poly source mesh.
        nearest = cKDTree(surface_points).query(filled_points, workers=1)[1]
        rgb = surface_rgb[nearest]
    return _quantize_colors(rgb)


def voxelize(
    path: Path,
    target_studs: int = DEFAULT_TARGET_STUDS,
    source_image_path: Path | None = None,
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> list[dict[str, Any]]:
    mesh = _mesh_from_file(path)
    if mesh.is_empty:
        raise RuntimeError("The generated mesh is empty")
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds[0])
    longest = max(float(value) for value in mesh.extents)
    if longest <= 0:
        raise RuntimeError("The generated mesh has invalid dimensions")
    resolution = _bounded_resolution(mesh.extents, target_studs, max_voxels)
    mesh.apply_scale(resolution / longest)
    surface_matrix = mesh.voxelized(pitch=1.0)
    surface_points = np.rint(surface_matrix.points).astype(int)
    points = np.rint(surface_matrix.fill().points).astype(int)
    if len(points) > max_voxels and resolution > 1:
        # The bounding-grid estimate is conservative, but keep one proportional
        # retry for unusual voxelizer rounding instead of decrementing and fully
        # rebuilding the grid many times.
        next_resolution = max(
            1,
            min(
                resolution - 1,
                int(resolution * (max_voxels / len(points)) ** (1 / 3) * 0.98),
            ),
        )
        mesh.apply_scale(next_resolution / resolution)
        surface_matrix = mesh.voxelized(pitch=1.0)
        surface_points = np.rint(surface_matrix.points).astype(int)
        points = np.rint(surface_matrix.fill().points).astype(int)
    if not len(points):
        raise RuntimeError("The generated mesh could not be voxelized")
    colors_by_point = _voxel_colors(mesh, surface_points, points, source_image_path)
    points -= points.min(axis=0)
    return [
        {
            # Stable Fast 3D GLBs are Y-up. The app's build grid is Z-up, so
            # source Y becomes build Z and source Z becomes build depth.
            "x": int(point[0]),
            "y": int(point[2]),
            "z": int(point[1]),
            "color": color,
        }
        for point, color in zip(points, colors_by_point)
    ]


def _export_voxel_model(voxels: list[dict[str, Any]], path: Path) -> None:
    centers = np.array([[v["x"], v["z"], v["y"]] for v in voxels], dtype=float)
    mesh = trimesh.voxel.ops.multibox(centers, pitch=0.96)
    source_colors = np.array(
        [[int(v["color"][i : i + 2], 16) for i in (1, 3, 5)] for v in voxels],
        dtype=np.uint8,
    )
    nearest = cKDTree(centers).query(mesh.triangles_center, workers=-1)[1]
    alpha = np.full((len(mesh.faces), 1), 255, dtype=np.uint8)
    mesh.visual.face_colors = np.hstack((source_colors[nearest], alpha))
    mesh.export(path)


def _model_path(result: Any) -> Path:
    """Extract the downloaded GLB path from Gradio's result variants."""
    value = result[-1] if isinstance(result, (tuple, list)) else result
    if isinstance(value, dict):
        value = value.get("path") or value.get("name") or value.get("url")
    if hasattr(value, "path"):
        value = value.path
    if not value:
        raise RuntimeError("Stable Fast 3D returned no model file")
    path = Path(str(value))
    if not path.exists():
        raise RuntimeError("The Stable Fast 3D model download was not found")
    return path


def _default_endpoint(space_id: str) -> str:
    if space_id.lower() == "upsampler/stable-fast-3d":
        return "/image_to_glb"
    return "/run_button"


def _providers() -> list[tuple[str, str]]:
    configured_space = os.getenv("HF_SPACE_ID", DEFAULT_SPACE)
    configured_endpoint = os.getenv("HF_SPACE_API_NAME") or _default_endpoint(configured_space)
    providers = [(configured_space, configured_endpoint)]
    for space_id in (DEFAULT_SPACE, FALLBACK_SPACE):
        candidate = (space_id, _default_endpoint(space_id))
        if candidate not in providers:
            providers.append(candidate)
    return providers


def _provider_error_reason(exc: Exception) -> str:
    """Expose actionable scheduler failures without leaking arbitrary internals."""
    message = unescape(re.sub(r"<[^>]+>", "", str(exc))).strip()
    lowered = message.lower()
    if "zerogpu" in lowered or "gpu quota" in lowered or "zerogpu runs limit" in lowered:
        return message
    if "no gpu was available" in lowered:
        return message
    return type(exc).__name__


def _generate_glb(image_path: Path, token: str | None) -> tuple[Path, str]:
    errors: list[str] = []
    for space_id, api_name in _providers():
        try:
            client = Client(space_id, hf_token=token, verbose=False)
            foreground_ratio = float(os.getenv("SF3D_FOREGROUND_RATIO", "0.85"))
            result = client.predict(
                input_image=handle_file(str(image_path)),
                foreground_ratio=foreground_ratio,
                remesh_option=os.getenv("SF3D_REMESH", "None"),
                vertex_count=int(os.getenv("SF3D_VERTEX_COUNT", "-1")),
                texture_size=int(os.getenv("SF3D_TEXTURE_SIZE", "1024")),
                api_name=api_name,
            )
            return _model_path(result), space_id
        except Exception as exc:
            errors.append(f"{space_id}: {_provider_error_reason(exc)}")
    attempted = ", ".join(errors)
    raise RuntimeError(f"Stable Fast 3D remote inference failed. Providers attempted: {attempted}.")


def _generate_ngrok_glb(image_path: Path, work_dir: Path) -> Path:
    """Call the user's Kaggle-hosted Stable Fast 3D service through ngrok."""
    base_url = os.getenv("SF3D_NGROK_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("SF3D_PROVIDER is ngrok, but SF3D_NGROK_URL is not configured")
    try:
        parsed = httpx.URL(base_url)
    except Exception as exc:
        raise RuntimeError("SF3D_NGROK_URL is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise RuntimeError("SF3D_NGROK_URL must be an http:// or https:// URL")

    endpoint = base_url if parsed.path.rstrip("/").endswith("/generate") else f"{base_url}/generate"
    timeout = float(os.getenv("SF3D_NGROK_TIMEOUT_SECONDS", "300"))
    texture_resolution = int(os.getenv("SF3D_TEXTURE_SIZE", "1024"))
    remesh_option = os.getenv("SF3D_NGROK_REMESH", "triangle").strip().lower()

    try:
        with image_path.open("rb") as source:
            response = httpx.post(
                endpoint,
                files={"file": (image_path.name, source, "application/octet-stream")},
                data={
                    "texture_resolution": str(texture_resolution),
                    "remesh_option": remesh_option,
                },
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=timeout,
                follow_redirects=True,
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Kaggle/ngrok Stable Fast 3D timed out after {timeout:g} seconds") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach the Kaggle/ngrok Stable Fast 3D API: {type(exc).__name__}") from exc

    if response.status_code >= 400:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("detail") or payload.get("error") or "") if isinstance(payload, dict) else ""
        except ValueError:
            detail = response.text.strip()
        suffix = f": {detail[:300]}" if detail else ""
        raise RuntimeError(f"Kaggle/ngrok Stable Fast 3D returned HTTP {response.status_code}{suffix}")
    if not response.content.startswith(b"glTF"):
        content_type = response.headers.get("content-type", "unknown")
        raise RuntimeError(
            "Kaggle/ngrok Stable Fast 3D did not return a binary GLB "
            f"(content type: {content_type})"
        )

    output = work_dir / "kaggle-stable-fast-3d.glb"
    output.write_bytes(response.content)
    return output


def reconstruct(
    image_path: Path,
    work_dir: Path,
    update: Callable[[int, str], None],
) -> tuple[list[dict[str, Any]], str, Path]:
    if os.getenv("SF3D_DEMO_MODE", "").lower() in {"1", "true", "yes"}:
        update(48, "Demo mode: shaping a sample voxel model")
        time.sleep(0.4)
        voxels = _demo_voxels(image_path.stem)
        model_path = work_dir / "demo-model.glb"
        _export_voxel_model(voxels, model_path)
        return voxels, "demo", model_path

    provider = os.getenv("SF3D_PROVIDER", PROVIDER_HUGGINGFACE).strip().lower()
    if provider in {PROVIDER_HUGGINGFACE, "hf", "hugging-face"}:
        update(14, "Connecting to Stable Fast 3D on Hugging Face")
        update(28, "Removing the background and preparing a 512 × 512 input")
        downloaded, _provider = _generate_glb(image_path, os.getenv("HF_TOKEN") or None)
        mode = "stable-fast-3d"
    elif provider == PROVIDER_NGROK:
        update(14, "Connecting to Stable Fast 3D on Kaggle through ngrok")
        update(28, "Uploading the prepared image to Kaggle")
        downloaded = _generate_ngrok_glb(image_path, work_dir)
        mode = "stable-fast-3d-ngrok"
    else:
        raise RuntimeError("SF3D_PROVIDER must be 'huggingface' or 'ngrok'")
    update(76, "Voxelizing the generated mesh")
    model_path = work_dir / "stable-fast-3d.glb"
    shutil.copy2(downloaded, model_path)
    return (
        voxelize(
            model_path,
            target_studs=int(os.getenv("VOXEL_TARGET_STUDS", str(DEFAULT_TARGET_STUDS))),
            source_image_path=image_path,
            max_voxels=int(os.getenv("VOXEL_MAX_CELLS", str(DEFAULT_MAX_VOXELS))),
        ),
        mode,
        model_path,
    )
