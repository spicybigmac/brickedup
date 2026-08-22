# Bricked Up

A hackathon-ready single-page app that turns one object image into a generated 3D model, voxelizes it, greedily packs the voxels into brick-shaped pieces, and produces a downloadable build manual.

## Run locally

Requires Python 3.11+.

```bash
cd bricked-up
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Open <http://localhost:8000>. The application itself has no account system or database.

By default, the backend calls the public `Upsampler/stable-fast-3d` Hugging Face Space and then tries the official Space if the primary provider fails. Public ZeroGPU Spaces have per-account/IP usage limits, and the official Space's stateful web workflow is not always callable through its public API. If all providers fail, the job shows the remote inference error. `HF_TOKEN` is optional for public Spaces. Add `GEMINI_API_KEY` to let Gemini name the generated build. The diagram layout and geometrically accurate layer steps are generated locally.

For offline development, set `SF3D_DEMO_MODE=1`. The UI labels the resulting procedural geometry as a demo.

## Input requirements

Upload one PNG, JPG/JPEG, or WebP image no larger than 10 MB.

- Use one object, fully visible in the frame.
- Prefer a plain or transparent background with clear separation around the object.
- Use even lighting and avoid blur.
- A square image works best. Stable Fast 3D removes the background and prepares a 512 × 512 model input.

## What each part does

- `main.py` serves the SPA, validates real image content with Pillow, exposes the API, and runs longer work in a thread pool.
- `app/stable_fast_3d.py` calls a Stable Fast 3D Hugging Face Space, tries a second provider when necessary, and downloads the generated GLB. Stable Fast 3D meshes are converted from Y-up into the app's Z-up build grid and filled into a target 32-stud voxel model capped at 35,000 cells. Colors use the closest surface triangle and barycentrically interpolated UV coordinates to sample the GLB texture accurately; vertex colors and front-projected source-image colors remain fallbacks. The result is adaptively quantized to 32 colors.
- `app/lego.py` scans each height/color layer and greedily places the largest common rectangular brick that fits. Odd layers reverse the preferred orientation to reduce aligned seams.
- `app/manual.py` optionally asks Gemini for a short build title, then uses ReportLab to create a diagram-first manual: color part icons, exact top-down stud grids, ghosted already-built support, consistent front markers, height meters, and placement arrows with very little text. Dense layers are split into smaller steps so every required part remains visible in the callouts.
- `app/store.py` holds job state in memory for this one-instance hackathon deployment.
- `static/` contains the no-build SPA and Three.js previews, including synchronized generated-mesh and voxel cameras.
- `render.yaml` contains the Render web-service definition.

## API flow

1. `POST /api/jobs` with multipart field `image`.
2. Poll `GET /api/jobs/{id}` until `stage` is `model_ready`.
3. Preview the source GLB from the returned `model_url` (`GET /api/jobs/{id}/model.glb`).
4. `POST /api/jobs/{id}/legolize`, then poll until `lego_ready`.
5. `POST /api/jobs/{id}/manual`, then poll until `complete`.
6. Download `GET /api/jobs/{id}/manual.pdf`.

`POST /api/jobs` returns `202 Accepted` after the image is validated and the background task is queued. It does not mean 3D generation has finished. Poll the returned job with `GET /api/jobs/{id}`; a completed provider failure appears there as `stage: "failed"` with a user-facing error. The GET request itself remains `200 OK` because the job resource was retrieved successfully, even when the work represented by that resource failed.

## Stable Fast 3D settings

- `HF_SPACE_ID` selects the primary Space and defaults to `Upsampler/stable-fast-3d`.
- `HF_SPACE_API_NAME` selects its Gradio endpoint and defaults to `/image_to_glb` for the primary Space.
- `HF_TOKEN` is an optional server-side Hugging Face token. It is never exposed to the browser.
- `SF3D_FOREGROUND_RATIO` defaults to `0.85`.
- `SF3D_REMESH` defaults to `None`; valid Space choices are `None`, `Triangle`, and `Quad`.
- `SF3D_VERTEX_COUNT` defaults to `-1`, which preserves the generated topology.
- `SF3D_TEXTURE_SIZE` defaults to `1024`.
- `VOXEL_TARGET_STUDS` defaults to `32`; dense meshes automatically step down only as far as needed to respect the cell cap.
- `VOXEL_MAX_CELLS` defaults to `35000`. The converter lowers the whole resolution instead of deleting every nth cell, so capped models remain contiguous.
- `SF3D_DEMO_MODE=1` disables the Space call and generates local demo geometry.

## MVP trade-offs

Stable Fast 3D infers unseen geometry from one image, so the result is fast and visually complete but not a measurement-accurate scan. The voxel grid intentionally discards fine surface and texture detail so it can be represented by rectangular bricks. The greedy packer is deterministic and easy to explain, but it is not a global minimum-part solver or a structural simulation.

For production, persist jobs in a database and object storage, move generation into a durable queue, use a private duplicated Space for predictable capacity, and validate colors and availability against a parts inventory.

## Render deployment

Create a **Blueprint** or **Web Service**, not a Static Site, from this repository. `render.yaml` installs the Python dependencies, starts Uvicorn on Render's assigned `$PORT`, and checks `/api/health`. Add `HF_TOKEN` and `GEMINI_API_KEY` as secret environment variables, then deploy and open the web service's single `onrender.com` URL. FastAPI serves both `static/` and `/api`, so the browser's relative API requests stay on the same origin and need no CORS configuration.

If a Static Site was created already, it can display the HTML but has no Python process behind `/api`. Create a new Web Service/Blueprint and use its URL instead. Confirm `https://YOUR-SERVICE.onrender.com/api/health` returns JSON before testing an upload. Free services spin down when idle and use an ephemeral filesystem, so generated jobs and downloads disappear on restart; upgrade the web service or add managed persistence before a live judging session if reliability matters.

## Tests

```bash
pytest -q
```

LEGO is a trademark of the LEGO Group, which does not sponsor or endorse this project.
