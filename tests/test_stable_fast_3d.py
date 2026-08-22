from pathlib import Path

import pytest
import trimesh
import numpy as np
from PIL import Image

from app import stable_fast_3d


def test_voxelize_fills_a_generated_mesh(tmp_path: Path):
    model = tmp_path / "box.glb"
    trimesh.creation.box(extents=(2, 2, 2)).export(model)
    voxels = stable_fast_3d.voxelize(model, target_studs=6)
    assert voxels
    assert {"x", "y", "z", "color"} <= voxels[0].keys()
    assert min(voxel["x"] for voxel in voxels) == 0


def test_voxelize_converts_y_up_mesh_to_z_up_build_grid(tmp_path: Path):
    model = tmp_path / "upright.glb"
    trimesh.creation.box(extents=(2, 6, 1)).export(model)
    voxels = stable_fast_3d.voxelize(model, target_studs=12)
    width = max(voxel["x"] for voxel in voxels) + 1
    depth = max(voxel["y"] for voxel in voxels) + 1
    height = max(voxel["z"] for voxel in voxels) + 1
    assert height > width > depth


def test_voxelize_uses_input_image_colors_when_mesh_has_no_texture(tmp_path: Path):
    model = tmp_path / "plain.glb"
    trimesh.creation.box(extents=(4, 4, 1)).export(model)
    image = Image.new("RGB", (64, 64), "#FF0000")
    for x in range(32, 64):
        for y in range(64):
            image.putpixel((x, y), (0, 0, 255))
    image_path = tmp_path / "source.png"
    image.save(image_path)

    voxels = stable_fast_3d.voxelize(model, target_studs=8, source_image_path=image_path)
    assert {voxel["color"] for voxel in voxels} == {"#FF0000", "#0000FF"}


def test_voxelize_reads_gltf_pbr_base_color_texture(tmp_path: Path):
    model = tmp_path / "textured.glb"
    mesh = trimesh.creation.box(extents=(4, 4, 1))
    uv = np.column_stack(
        (
            (mesh.vertices[:, 0] - mesh.bounds[0, 0]) / mesh.extents[0],
            (mesh.vertices[:, 1] - mesh.bounds[0, 1]) / mesh.extents[1],
        )
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (8, 8), "#12C878")
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.export(model)

    voxels = stable_fast_3d.voxelize(model, target_studs=8)
    assert {voxel["color"] for voxel in voxels} == {"#12C878"}


def test_surface_colors_interpolate_uvs_across_nearest_triangle():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    texture = Image.new("RGB", (3, 2))
    for y in range(2):
        texture.putpixel((0, y), (255, 0, 0))
        texture.putpixel((1, y), (0, 255, 0))
        texture.putpixel((2, y), (0, 0, 255))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture),
    )

    colors = stable_fast_3d._surface_colors(mesh, np.array([[1, 1, 0]], dtype=float), None)

    assert colors.tolist() == [[0, 255, 0]]


def test_voxel_cap_reduces_resolution_without_punching_holes(tmp_path: Path):
    model = tmp_path / "dense-box.glb"
    trimesh.creation.box(extents=(2, 2, 2)).export(model)

    voxels = stable_fast_3d.voxelize(model, target_studs=32, max_voxels=35_000)
    dimensions = [max(voxel[axis] for voxel in voxels) + 1 for axis in ("x", "y", "z")]

    assert len(voxels) <= 35_000
    assert len(voxels) == int(np.prod(dimensions))


def test_voxel_cap_is_applied_before_expensive_voxelization(monkeypatch, tmp_path: Path):
    model = tmp_path / "bounded-box.glb"
    trimesh.creation.box(extents=(2, 2, 2)).export(model)
    calls = []
    original = trimesh.Trimesh.voxelized

    def tracked_voxelized(mesh, *args, **kwargs):
        calls.append(args[0] if args else kwargs.get("pitch"))
        return original(mesh, *args, **kwargs)

    monkeypatch.setattr(trimesh.Trimesh, "voxelized", tracked_voxelized)

    voxels = stable_fast_3d.voxelize(model, target_studs=32, max_voxels=35_000)

    assert len(calls) == 1
    assert len(voxels) <= 35_000


def test_voxelize_only_runs_mesh_color_lookup_for_surface_cells(monkeypatch, tmp_path: Path):
    model = tmp_path / "colored-box.glb"
    trimesh.creation.box(extents=(2, 2, 2)).export(model)
    sampled = []
    original = stable_fast_3d._surface_colors

    def tracked_surface_colors(mesh, points, source_image_path):
        sampled.append(len(points))
        return original(mesh, points, source_image_path)

    monkeypatch.setattr(stable_fast_3d, "_surface_colors", tracked_surface_colors)

    voxels = stable_fast_3d.voxelize(model, target_studs=12)

    assert sampled[0] < len(voxels)


def test_generation_fails_over_to_healthy_space(monkeypatch, tmp_path: Path):
    image = tmp_path / "object.png"
    image.write_bytes(b"image")
    model = tmp_path / "model.glb"
    model.write_bytes(b"glb")
    calls = []

    class FakeClient:
        def __init__(self, space_id, **_kwargs):
            self.space_id = space_id

        def predict(self, **kwargs):
            calls.append((self.space_id, kwargs["api_name"]))
            if self.space_id == "stabilityai/stable-fast-3d":
                raise RuntimeError("hidden upstream exception")
            return str(model)

    monkeypatch.setenv("HF_SPACE_ID", "stabilityai/stable-fast-3d")
    monkeypatch.delenv("HF_SPACE_API_NAME", raising=False)
    monkeypatch.setattr(stable_fast_3d, "Client", FakeClient)
    monkeypatch.setattr(stable_fast_3d, "handle_file", lambda path: path)

    result, provider = stable_fast_3d._generate_glb(image, None)
    assert result == model
    assert provider == "Upsampler/stable-fast-3d"
    assert calls == [
        ("stabilityai/stable-fast-3d", "/run_button"),
        ("Upsampler/stable-fast-3d", "/image_to_glb"),
    ]


def test_generation_reports_all_provider_failure(monkeypatch, tmp_path: Path):
    class BrokenClient:
        def __init__(self, space_id, **_kwargs):
            self.space_id = space_id

        def predict(self, **_kwargs):
            raise RuntimeError("provider internals must not leak")

    image = tmp_path / "object.png"
    image.write_bytes(b"image")
    monkeypatch.setenv("HF_SPACE_ID", "Upsampler/stable-fast-3d")
    monkeypatch.delenv("HF_SPACE_API_NAME", raising=False)
    monkeypatch.setattr(stable_fast_3d, "Client", BrokenClient)
    monkeypatch.setattr(stable_fast_3d, "handle_file", lambda path: path)

    with pytest.raises(RuntimeError, match="remote inference failed") as error:
        stable_fast_3d._generate_glb(image, None)
    assert "provider internals" not in str(error.value)


def test_generation_exposes_quota_numbers_but_strips_html(monkeypatch, tmp_path: Path):
    class QuotaClient:
        def __init__(self, space_id, **_kwargs):
            self.space_id = space_id

        def predict(self, **_kwargs):
            raise RuntimeError(
                "You have exceeded your ZeroGPU quota (30s requested vs. 12s left). "
                '<a href="https://example.com">Try again later</a>'
            )

    image = tmp_path / "object.png"
    image.write_bytes(b"image")
    monkeypatch.setenv("HF_SPACE_ID", "Upsampler/stable-fast-3d")
    monkeypatch.delenv("HF_SPACE_API_NAME", raising=False)
    monkeypatch.setattr(stable_fast_3d, "Client", QuotaClient)
    monkeypatch.setattr(stable_fast_3d, "handle_file", lambda path: path)

    with pytest.raises(RuntimeError) as error:
        stable_fast_3d._generate_glb(image, None)
    assert "30s requested vs. 12s left" in str(error.value)
    assert "<a" not in str(error.value)


def test_ngrok_generation_posts_expected_form_and_saves_glb(monkeypatch, tmp_path: Path):
    image = tmp_path / "object.png"
    image.write_bytes(b"png-data")
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b"glTF-binary-model"
        headers = {"content-type": "model/gltf-binary"}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs["data"]
        captured["timeout"] = kwargs["timeout"]
        captured["header"] = kwargs["headers"]["ngrok-skip-browser-warning"]
        captured["file"] = kwargs["files"]["file"][1].read()
        return FakeResponse()

    monkeypatch.setenv("SF3D_NGROK_URL", "https://example.ngrok-free.app/")
    monkeypatch.setenv("SF3D_NGROK_TIMEOUT_SECONDS", "275")
    monkeypatch.setenv("SF3D_TEXTURE_SIZE", "2048")
    monkeypatch.setenv("SF3D_NGROK_REMESH", "Triangle")
    monkeypatch.setattr(stable_fast_3d.httpx, "post", fake_post)

    result = stable_fast_3d._generate_ngrok_glb(image, tmp_path)

    assert result.read_bytes() == b"glTF-binary-model"
    assert captured == {
        "url": "https://example.ngrok-free.app/generate",
        "data": {"texture_resolution": "2048", "remesh_option": "triangle"},
        "timeout": 275.0,
        "header": "true",
        "file": b"png-data",
    }


def test_ngrok_generation_rejects_non_glb_response(monkeypatch, tmp_path: Path):
    image = tmp_path / "object.png"
    image.write_bytes(b"png-data")

    class FakeResponse:
        status_code = 200
        content = b"<html>ngrok error</html>"
        headers = {"content-type": "text/html"}

    monkeypatch.setenv("SF3D_NGROK_URL", "https://example.ngrok-free.app")
    monkeypatch.setattr(stable_fast_3d.httpx, "post", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="did not return a binary GLB"):
        stable_fast_3d._generate_ngrok_glb(image, tmp_path)


def test_reconstruct_can_select_ngrok_provider(monkeypatch, tmp_path: Path):
    image = tmp_path / "object.png"
    image.write_bytes(b"image")
    downloaded = tmp_path / "download.glb"
    downloaded.write_bytes(b"glTF-model")
    calls = []
    monkeypatch.setenv("SF3D_PROVIDER", "ngrok")
    monkeypatch.setattr(
        stable_fast_3d,
        "_generate_ngrok_glb",
        lambda image_path, work_dir: calls.append((image_path, work_dir)) or downloaded,
    )
    monkeypatch.setattr(
        stable_fast_3d,
        "voxelize",
        lambda path, **kwargs: calls.append((path, kwargs)) or [{"x": 0, "y": 0, "z": 0, "color": "#123456"}],
    )

    voxels, mode, model_path = stable_fast_3d.reconstruct(image, tmp_path, lambda *_args: None)

    assert voxels[0]["color"] == "#123456"
    assert mode == "stable-fast-3d-ngrok"
    assert model_path.read_bytes() == b"glTF-model"
    assert calls[0] == (image, tmp_path)
