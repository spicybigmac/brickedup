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


def test_local_relief_is_upright_and_preserves_image_colors(tmp_path: Path):
    image = Image.new("RGB", (12, 20), "white")
    for x in range(3, 9):
        for y in range(2, 18):
            image.putpixel((x, y), (255, 0, 0) if x < 6 else (0, 0, 255))
    image_path = tmp_path / "object.png"
    image.save(image_path)

    voxels = stable_fast_3d._image_relief_voxels(image_path, target_studs=20)

    width = max(voxel["x"] for voxel in voxels) + 1
    depth = max(voxel["y"] for voxel in voxels) + 1
    height = max(voxel["z"] for voxel in voxels) + 1
    assert height > width > depth
    assert {voxel["color"] for voxel in voxels} == {"#FF0000", "#0000FF"}


def test_reconstruct_falls_back_locally_when_spaces_fail(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "object.png"
    Image.new("RGB", (10, 18), "#C84832").save(image_path)
    monkeypatch.setattr(
        stable_fast_3d,
        "_generate_glb",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("quota exhausted")),
    )
    messages = []

    voxels, mode, model_path = stable_fast_3d.reconstruct(
        image_path, tmp_path, lambda progress, message: messages.append((progress, message))
    )

    assert voxels
    assert mode == "local-fallback"
    assert model_path.exists()
    assert any("Free GPU unavailable" in message for _, message in messages)
