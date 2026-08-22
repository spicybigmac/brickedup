# Bricked Up

Bricked Up turns one object image into a generated 3D mesh, a colored voxel model, a brick-ready build, and a downloadable diagram-first PDF manual.

```text
Image → Stable Fast 3D GLB → colored voxels → greedy brick packing → PDF manual
```

This is a single-page FastAPI + Three.js application. The default local workflow uses a Kaggle-hosted Stable Fast 3D server.

> **Input:** one PNG, JPG/JPEG, or WebP image up to 10 MB.

## Quick start

### 1. Install prerequisites

- Python **3.12** is the tested version. Python 3.11+ should also work.
- `pip` and a shell.
- For real 3D generation, choose one provider in [Choose a 3D provider](#choose-a-3d-provider).
**- During Judging Phase, use the following SF3D_NGROK_URL: https://during-kimono-even.ngrok-free.dev**

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
cd bricked-up
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
cd bricked-up
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Activation changes the `python` and `pip` commands in the current terminal so they use this project's isolated `.venv` folder. Open a new terminal later? Activate it again before running the app.

### 3. Install dependencies and create local configuration

macOS/Linux:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in an editor and choose a provider below. Do not commit `.env`; it can contain private API tokens and your ngrok URL.

### 4. Start the app

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000).

For automatic backend reloads while editing Python:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Confirm the server is healthy:

```bash
curl http://localhost:8000/api/health
```

You should receive JSON with `"ok": true` and the active reconstruction provider.

## Choose a 3D provider

Set exactly one mode in `.env`.

### Option A — Kaggle + ngrok (recommended for a live demo)

Run Stable Fast 3D in your Kaggle notebook, start its ngrok tunnel, then copy the current public tunnel URL into `.env`:

# NOTE: during the Judging period, I will be running a Kaggle server: Use the following SF3D_NGROK_URL: https://during-kimono-even.ngrok-free.dev

```env
SF3D_PROVIDER=ngrok
SF3D_NGROK_URL=https://xxxx.ngrok-free.app
SF3D_NGROK_TIMEOUT_SECONDS=300
SF3D_NGROK_REMESH=triangle
```

Bricked Up sends a multipart request to `https://xxxx.ngrok-free.app/generate` with:

```text
file                 uploaded image
texture_resolution   1024 by default
remesh_option        triangle by default
```

The Kaggle endpoint must return the raw binary GLB body—not JSON, HTML, or an ngrok warning page. Keep the Kaggle notebook and ngrok tunnel running while using the app. A tunnel URL usually changes after restarting the notebook, so update `SF3D_NGROK_URL` and restart FastAPI when it changes.

### Option B — Hugging Face Space

```env
SF3D_PROVIDER=huggingface
HF_SPACE_ID=Upsampler/stable-fast-3d
HF_SPACE_API_NAME=/image_to_glb
# Optional, but useful for rate limits:
HF_TOKEN=
```

The backend tries the configured Space and then a second Stable Fast 3D Space if the first one fails. Public ZeroGPU Spaces can be queued, temporarily unavailable, or quota-limited even with a valid Hugging Face token.

## Recommended local settings

Use:

```env
VOXEL_TARGET_STUDS=28
VOXEL_MAX_CELLS=25000
WORKER_THREADS=2
```

This reduces dense models by roughly 25–35% while retaining the recognizable shape and texture. `VOXEL_TARGET_STUDS` is the quality control; `VOXEL_MAX_CELLS` is a hard safety limit.

## Use the app

1. Upload a clear image of one object.
2. Wait while Stable Fast 3D produces the GLB and the backend creates a colored voxel model.
3. Orbit the generated mesh and voxel model together on the comparison page.
4. Select **Convert to bricks**. The packer creates the brick layout and parts manifest.
5. Select **Generate instructions** and download the PDF.

For the best source image, use one clearly isolated object, a plain/transparent background, even lighting, sharp focus, and minimal occlusion. A square image generally works best.

## Local configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `SF3D_PROVIDER` | `huggingface` | `huggingface` or `ngrok` reconstruction route. |
| `SF3D_NGROK_URL` | empty | Kaggle tunnel base URL. `/generate` is added automatically. |
| `SF3D_NGROK_TIMEOUT_SECONDS` | `300` | Maximum time waiting for the Kaggle response. |
| `SF3D_NGROK_REMESH` | `triangle` | Remesh option sent to the Kaggle endpoint. |
| `HF_SPACE_ID` | `Upsampler/stable-fast-3d` | Primary Hugging Face Space. |
| `HF_SPACE_API_NAME` | `/image_to_glb` | Gradio API endpoint for the primary Space. |
| `HF_TOKEN` | empty | Optional Hugging Face token; never sent to the browser. |
| `SF3D_FOREGROUND_RATIO` | `0.85` | Foreground setting for Hugging Face generation. |
| `SF3D_REMESH` | `None` | Hugging Face remesh setting: `None`, `Triangle`, or `Quad`. |
| `SF3D_VERTEX_COUNT` | `-1` | Hugging Face mesh topology setting. |
| `SF3D_TEXTURE_SIZE` | `1024` | Requested texture size for the remote model. |
| `VOXEL_TARGET_STUDS` | `32` | Desired longest dimension of the voxel grid. |
| `VOXEL_MAX_CELLS` | `35000` | Maximum occupied voxel cells. |
| `SF3D_DEMO_MODE` | `0` | Set to `1` for offline demo geometry. |
| `GEMINI_API_KEY` | empty | Optional: generates a short build title. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for the title request. |
| `PORT` | `8000` | Local server port. |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size. |
| `WORKER_THREADS` | `2` | Number of background-job worker threads. |

## How the pipeline works

1. **3D reconstruction:** Stable Fast 3D returns a textured GLB from the input image.
2. **Voxelization:** Trimesh normalizes the mesh onto a bounded integer grid, rasterizes its surface, fills enclosed volume, and samples its GLB texture with barycentric UV interpolation. Surface colors are propagated to interior cells, then reduced to a 32-color palette.
3. **Brick packing:** Each exact color and height layer is processed independently. A greedy algorithm tries the largest legal rectangular brick footprint first, guaranteeing each voxel is covered exactly once.
4. **Instructions:** ReportLab creates a vector PDF from the exact brick coordinates: inventory, stud grids, colored part callouts, ghosted support, placement arrows, front marker, and layer meter. Gemini is optional and supplies only a short title; it does not decide geometry or placements.

## API flow

```text
POST /api/jobs
  → poll GET /api/jobs/{id} until stage is model_ready
  → POST /api/jobs/{id}/legolize
  → poll until lego_ready
  → POST /api/jobs/{id}/manual
  → poll until complete
  → GET /api/jobs/{id}/manual.pdf
```

`POST /api/jobs` returns `202 Accepted` after validation and queueing. It does not mean reconstruction has completed. A later job lookup may return HTTP `200 OK` with `stage: "failed"`; that means the job record was found successfully, but its background work failed. Read its `error` field.

The job store is process-local. Restarting FastAPI clears active job records and causes old `/api/jobs/{id}` links to return `404`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `No module named scipy` | The virtual environment is inactive or dependencies were not installed. Run `source .venv/bin/activate` and `pip install -r requirements.txt`. |
| Stable Fast 3D unavailable / ZeroGPU message | The public Space is queued, out of quota, or unavailable. Retry later, use Kaggle/ngrok, or enable demo mode. |
| Kaggle/ngrok request fails | Confirm the notebook and tunnel are still running, the URL is current, and the endpoint returns binary GLB from `/generate`. Then restart FastAPI after changing `.env`. |
| `/api/jobs/{id}` returns `404` | The server restarted or the ID belongs to a different process; create a new build. |
| The browser still shows an old UI | Hard-refresh the page after frontend changes. |
| Voxelization is slow | Use `VOXEL_TARGET_STUDS=28` and `VOXEL_MAX_CELLS=25000`; avoid starting multiple builds simultaneously. |
| Colors are less accurate on hidden surfaces | Single-image reconstruction must infer unseen geometry. Exterior colors use the GLB texture; enclosed cells inherit nearby surface colors. |

## Tests

With the virtual environment active:

```bash
pytest -q
```

## Project layout

- `main.py` — FastAPI server, static-site hosting, API endpoints, validation, and background jobs.
- `app/stable_fast_3d.py` — Stable Fast 3D providers, GLB validation, voxelization, and color transfer.
- `app/lego.py` — Greedy voxel-to-brick packing and bill of materials.
- `app/manual.py` — Deterministic vector PDF manual generator and optional Gemini title request.
- `app/store.py` — In-memory job state.
- `static/` — Single-page interface and Three.js viewers.
- `.env.example` — safe configuration template.
- `render.yaml` — Render Web Service deployment configuration.

