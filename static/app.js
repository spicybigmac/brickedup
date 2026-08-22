import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/loaders/GLTFLoader.js";

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACTIVE_STAGES = new Set(["queued", "reconstructing", "legolizing", "manual_generating"]);
const STORAGE_KEY = "bricked-up-job";
const views = Object.fromEntries([...document.querySelectorAll("[data-view]")].map((el) => [el.dataset.view, el]));
const railSteps = [...document.querySelectorAll("#progress-rail li")];

const state = {
  job: null,
  file: null,
  currentView: "landing",
  lastAction: null,
  pollTimer: null,
  renderers: {},
};

const stageToView = {
  queued: "processing",
  reconstructing: "processing",
  model_ready: "voxel",
  legolizing: "processing",
  lego_ready: "lego",
  manual_generating: "processing",
  complete: "manual",
  failed: "error",
};

const viewStep = { landing: 0, upload: 0, processing: 1, voxel: 2, lego: 3, manual: 4, error: -1 };

function showView(name) {
  Object.entries(views).forEach(([key, el]) => {
    const active = key === name;
    el.classList.toggle("is-active", active);
    el.setAttribute("aria-hidden", String(!active));
  });
  state.currentView = name;
  updateRail(viewStep[name]);
  if (name !== "voxel") disposeViewer("voxel");
  if (name !== "lego") disposeViewer("lego");
  window.scrollTo({ top: 0, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  document.querySelector(`[data-view="${name}"] h1, [data-view="${name}"] h2`)?.focus?.({ preventScroll: true });
}

function updateRail(activeIndex) {
  railSteps.forEach((step, index) => {
    step.classList.toggle("is-complete", activeIndex > index);
    step.classList.toggle("is-active", activeIndex === index);
    if (activeIndex === index) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  });
}

function resetBuild() {
  stopPolling();
  disposeViewer("voxel");
  disposeViewer("lego");
  state.job = null;
  state.file = null;
  sessionStorage.removeItem(STORAGE_KEY);
  document.querySelector("#upload-form").reset();
  document.querySelector("#upload-button").disabled = true;
  document.querySelector("#file-readout").hidden = true;
  document.querySelector("#drop-zone").classList.remove("has-file");
  document.querySelector("#drop-title").textContent = "Drop your object image here";
  document.querySelector("#drop-detail").textContent = "or choose a file · PNG, JPG, or WebP";
  setMessage("");
  showView("landing");
}

function setMessage(message, kind = "error") {
  const el = document.querySelector("#upload-message");
  el.textContent = message;
  el.dataset.kind = kind;
}

async function inspectImage(file) {
  if (!file) return;
  setMessage("");
  state.file = null;
  document.querySelector("#upload-button").disabled = true;
  document.querySelector("#file-readout").hidden = true;
  document.querySelector("#drop-zone").classList.remove("has-file");
  document.querySelector("#drop-title").textContent = "Drop your object image here";
  document.querySelector("#drop-detail").textContent = "or choose a file · PNG, JPG, or WebP";

  const supportedMime = new Set(["image/png", "image/jpeg", "image/webp"]);
  const supportedExtension = /\.(png|jpe?g|webp)$/i.test(file.name);
  if ((!supportedMime.has(file.type) && file.type) || !supportedExtension) {
    setMessage("Choose a PNG, JPG, JPEG, or WebP image.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    setMessage("That image is over 10 MB. Choose a smaller file and try again.");
    return;
  }

  try {
    const dimensions = await readImageDimensions(file);
    state.file = file;
    const readout = document.querySelector("#file-readout");
    readout.textContent = `${file.name} · ${dimensions.width} × ${dimensions.height} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
    readout.hidden = false;
    document.querySelector("#drop-zone").classList.add("has-file");
    document.querySelector("#drop-title").textContent = "Image ready for the workshop";
    document.querySelector("#drop-detail").textContent = "Choose another file to replace it";
    document.querySelector("#upload-button").disabled = false;
  } catch {
    setMessage("We couldn’t read that image. Try exporting it as PNG, JPG, or WebP.");
  }
}

async function readImageDimensions(file) {
  if ("createImageBitmap" in window) {
    const bitmap = await createImageBitmap(file);
    const dimensions = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    if (!dimensions.width || !dimensions.height) throw new Error("empty image");
    return dimensions;
  }
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      const dimensions = { width: image.naturalWidth, height: image.naturalHeight };
      URL.revokeObjectURL(url);
      resolve(dimensions);
    };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("image decode error")); };
    image.src = url;
  });
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("The backend isn’t reachable. Start it with python main.py, then try again.");
  }
  let payload = null;
  try { payload = await response.json(); } catch { /* PDF or empty response */ }
  if (!response.ok) {
    const detail = payload?.detail;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail || payload?.error || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function isMissingJob(error) {
  return error?.status === 404 || error?.message === "Build not found";
}

async function uploadImage(event) {
  event.preventDefault();
  if (!state.file) {
    setMessage("Choose a readable PNG, JPG, JPEG, or WebP image up to 10 MB.");
    return;
  }
  const button = document.querySelector("#upload-button");
  button.disabled = true;
  const body = new FormData();
  body.append("image", state.file);
  showProcessing("reconstructing", 4, "Preparing your image for Stable Fast 3D…");
  state.lastAction = "upload";
  try {
    state.job = await api("/api/jobs", { method: "POST", body });
    saveJob();
    applyJob(state.job);
    if (ACTIVE_STAGES.has(state.job.stage)) startPolling();
  } catch (error) {
    fail(error.message);
  } finally {
    button.disabled = false;
  }
}

function showProcessing(stage, progress, message) {
  const isBricks = stage === "legolizing";
  const isManual = stage === "manual_generating";
  document.querySelector("#processing-kicker").textContent = isManual ? "05 / Manual" : isBricks ? "04 / Bricks" : "02 / Generate";
  document.querySelector("#processing-title").textContent = isManual ? "Writing the build book." : isBricks ? "Finding the best fit." : "Building the shape.";
  document.querySelector("#processing-message").textContent = message || "Working through the next stage…";
  document.querySelector("#progress-bar").style.width = `${Math.max(0, Math.min(100, progress || 0))}%`;
  document.querySelector("#progress-percent").textContent = `${Math.round(progress || 0)}%`;
  document.querySelector("#progress-stage").textContent = stage.replaceAll("_", " ").toUpperCase();
  showView("processing");
  updateRail(isManual ? 4 : isBricks ? 3 : 1);
}

function saveJob() {
  if (state.job?.id) sessionStorage.setItem(STORAGE_KEY, state.job.id);
}

function applyJob(job) {
  state.job = job;
  saveJob();
  if (job.stage === "failed" || job.error) return fail(job.error || job.message || "The build could not be completed.");
  if (ACTIVE_STAGES.has(job.stage)) {
    showProcessing(job.stage, job.progress, job.message);
    return;
  }
  const view = stageToView[job.stage] || "processing";
  showView(view);
  if (view === "voxel") populateVoxel(job);
  if (view === "lego") populateLego(job);
  if (view === "manual") populateManual(job);
}

function startPolling() {
  stopPolling();
  const tick = async () => {
    if (!state.job?.id) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(state.job.id)}`);
      applyJob(job);
      if (ACTIVE_STAGES.has(job.stage)) state.pollTimer = setTimeout(tick, 1250);
    } catch (error) { fail(error.message); }
  };
  state.pollTimer = setTimeout(tick, 800);
}

function stopPolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

async function startLegolize() {
  if (!state.job?.id) return;
  state.lastAction = "legolize";
  showProcessing("legolizing", Math.max(62, state.job.progress || 0), "Testing large bricks against the voxel grid…");
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/legolize`, { method: "POST" });
    applyJob(job);
    if (ACTIVE_STAGES.has(job.stage)) startPolling();
  } catch (error) { fail(error.message); }
}

async function generateManual() {
  if (!state.job?.id) return;
  state.lastAction = "manual";
  showProcessing("manual_generating", Math.max(88, state.job.progress || 0), "Turning layers and parts into clear build steps…");
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/manual`, { method: "POST" });
    applyJob(job);
    if (ACTIVE_STAGES.has(job.stage)) startPolling();
  } catch (error) { fail(error.message); }
}

function populateVoxel(job) {
  const voxels = job.model_voxels || job.voxels || [];
  const dimensions = job.model_dimensions || dimensionsOf(voxels);
  document.querySelector("#model-dimensions").textContent = Array.isArray(dimensions)
    ? dimensions.join(" × ")
    : `${dimensions.width ?? dimensions.x ?? 0} × ${dimensions.depth ?? dimensions.y ?? 0} × ${dimensions.height ?? dimensions.z ?? 0}`;
  document.querySelector("#voxel-count").textContent = Number(voxels.length).toLocaleString();
  const isDemo = job.mode === "demo";
  const modeTag = document.querySelector("#demo-tag");
  const modeNotice = document.querySelector("#demo-notice");
  modeTag.hidden = !isDemo;
  modeNotice.hidden = !isDemo;
  if (isDemo) {
    modeTag.textContent = "DEMO MODEL";
    modeNotice.querySelector("strong").textContent = "Demo geometry";
    modeNotice.querySelector("span").innerHTML = "Disable <code>SF3D_DEMO_MODE</code> to generate with Stable Fast 3D through the public Hugging Face Space.";
  }
  requestAnimationFrame(() => createComparisonViewer(voxels, job.model_url));
}

function populateLego(job) {
  const bricks = job.lego_bricks || job.bricks || [];
  document.querySelector("#piece-count").textContent = Number(job.piece_count ?? bricks.length).toLocaleString();
  const bom = job.bill_of_materials || makeBom(bricks);
  const target = document.querySelector("#bill-of-materials");
  target.innerHTML = "";
  const entries = Array.isArray(bom) ? bom : Object.entries(bom).map(([name, count]) => ({ name, count }));
  if (!entries.length) target.innerHTML = '<p class="empty-parts">The parts manifest will appear here once the greedy fit is complete.</p>';
  entries.slice(0, 18).forEach((part) => {
    const row = document.createElement("div");
    row.className = "part-row";
    const color = cssColor(part.color || part.hex || "#c84832");
    const label = part.name || part.part_name || part.part || `${part.w || part.width || 1} × ${part.d || part.depth || 1} brick`;
    row.innerHTML = `<span class="part-swatch" style="background:${color}"></span><span></span><strong></strong>`;
    row.children[1].textContent = label;
    row.children[2].textContent = `× ${part.count ?? part.quantity ?? 0}`;
    target.append(row);
  });
  requestAnimationFrame(() => createViewer("lego", bricks, "lego-viewer"));
}

function populateManual(job) {
  const link = document.querySelector("#download-button");
  link.href = job.manual_url || `/api/jobs/${encodeURIComponent(job.id)}/manual.pdf`;
}

function fail(message) {
  stopPolling();
  document.querySelector("#error-message").textContent = message;
  showView("error");
}

async function retry() {
  if (state.lastAction === "legolize") return startLegolize();
  if (state.lastAction === "manual") return generateManual();
  if (state.lastAction === "recover") return recoverJob();
  showView("upload");
}

function dimensionsOf(items) {
  if (!items.length) return [0, 0, 0];
  const max = items.reduce((acc, item) => [Math.max(acc[0], Number(item.x) || 0), Math.max(acc[1], Number(item.y) || 0), Math.max(acc[2], Number(item.z) || 0)], [0, 0, 0]);
  return max.map((value) => value + 1);
}

function makeBom(bricks) {
  const grouped = new Map();
  bricks.forEach((brick) => {
    const w = brick.w || brick.width || 1;
    const d = brick.d || brick.depth || 1;
    const color = brick.color || "#c84832";
    const key = `${w} × ${d} brick|${color}`;
    grouped.set(key, { name: `${w} × ${d} brick`, color, count: (grouped.get(key)?.count || 0) + 1 });
  });
  return [...grouped.values()].sort((a, b) => b.count - a.count);
}

function cssColor(value) {
  if (typeof value === "number") return `#${value.toString(16).padStart(6, "0")}`;
  if (Array.isArray(value)) return `rgb(${value.slice(0, 3).join(",")})`;
  const named = { red: "#c84832", yellow: "#e8b843", blue: "#2854a6", black: "#252521", white: "#e9e4da", green: "#477a55", tan: "#b79a71", orange: "#d86d2c", gray: "#77736b", grey: "#77736b" };
  return named[String(value).toLowerCase()] || String(value || "#c84832");
}

function setSceneLighting(scene) {
  scene.background = new THREE.Color(0xe8e1d4);
  scene.fog = new THREE.Fog(0xe8e1d4, 45, 110);
  scene.add(new THREE.HemisphereLight(0xfffbef, 0x55534e, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 3.2);
  key.position.set(18, 30, 16);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);
}

function createRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.tabIndex = 0;
  container.append(renderer.domElement);
  return renderer;
}

function centerAndNormalize(group) {
  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  group.position.sub(center);
  const maxSize = Math.max(size.x, size.y, size.z, 1);
  group.scale.setScalar(18 / maxSize);
  return 18;
}

function addComparisonGround(scene, size = 18) {
  const grid = new THREE.GridHelper(32, 16, 0x8c877e, 0xc8c0b4);
  grid.position.y = -size / 2 - .56;
  scene.add(grid);
}

function createComparisonViewer(voxels, modelUrl) {
  disposeViewer("voxel");
  const modelContainer = document.getElementById("generated-viewer");
  const voxelContainer = document.getElementById("voxel-viewer");
  if (!modelContainer || !voxelContainer) return;
  modelContainer.innerHTML = '<p class="viewer-loading">Loading generated model…</p>';
  voxelContainer.innerHTML = "";
  if (!voxels?.length) {
    voxelContainer.innerHTML = '<p class="viewer-error"><strong>No voxel geometry</strong><span>The conversion returned an empty model.</span></p>';
    showModelFallback(modelContainer, "Voxel preview is unavailable for comparison.");
    return;
  }

  const scenes = [new THREE.Scene(), new THREE.Scene()];
  scenes.forEach(setSceneLighting);
  // Keep the generated mesh photographic, but present voxels like a clean CAD
  // drawing: no distance haze and no light-dependent colour shifts.
  scenes[1].fog = null;
  const cameras = [
    new THREE.PerspectiveCamera(36, 1, .1, 300),
    new THREE.PerspectiveCamera(36, 1, .1, 300),
  ];
  const renderers = [createRenderer(modelContainer), createRenderer(voxelContainer)];
  modelContainer.querySelector(".viewer-loading")?.remove();

  const voxelGroup = new THREE.Group();
  scenes[1].add(voxelGroup);
  addVoxels(voxelGroup, voxels);
  const viewSize = centerAndNormalize(voxelGroup);
  scenes.forEach((scene) => addComparisonGround(scene, viewSize));
  cameras.forEach((camera) => camera.position.set(viewSize * 1.35, viewSize * .95, viewSize * 1.55));

  const controls = cameras.map((camera, index) => {
    const control = new OrbitControls(camera, renderers[index].domElement);
    control.enableDamping = true;
    control.dampingFactor = .065;
    control.minDistance = viewSize * .8;
    control.maxDistance = viewSize * 4.5;
    control.target.set(0, 0, 0);
    control.update();
    return control;
  });

  let syncing = false;
  const syncCamera = (sourceIndex) => {
    if (syncing) return;
    syncing = true;
    const targetIndex = sourceIndex === 0 ? 1 : 0;
    cameras[targetIndex].position.copy(cameras[sourceIndex].position);
    cameras[targetIndex].quaternion.copy(cameras[sourceIndex].quaternion);
    cameras[targetIndex].zoom = cameras[sourceIndex].zoom;
    cameras[targetIndex].updateProjectionMatrix();
    controls[targetIndex].target.copy(controls[sourceIndex].target);
    controls[targetIndex].update();
    syncing = false;
  };
  const syncHandlers = [() => syncCamera(0), () => syncCamera(1)];
  controls.forEach((control, index) => control.addEventListener("change", syncHandlers[index]));
  syncCamera(0);

  let record;
  const loop = () => {
    if (record.disposed) return;
    controls.forEach((control) => control.update());
    renderers.forEach((renderer, index) => renderer.render(scenes[index], cameras[index]));
    record.frame = requestAnimationFrame(loop);
  };

  const resize = new ResizeObserver(() => {
    [modelContainer, voxelContainer].forEach((container, index) => {
      if (!container.clientWidth || !container.clientHeight) return;
      cameras[index].aspect = container.clientWidth / container.clientHeight;
      cameras[index].updateProjectionMatrix();
      renderers[index].setSize(container.clientWidth, container.clientHeight);
    });
  });
  resize.observe(modelContainer);
  resize.observe(voxelContainer);

  record = {
    scenes, cameras, renderers, controls, syncHandlers, frame: 0, resize, disposed: false,
    initial: cameras[0].position.clone(),
    loadToken: Symbol("comparison-load"),
  };
  state.renderers.voxel = record;
  loop();

  if (!modelUrl) {
    showModelFallback(modelContainer, "The generated GLB was not provided. You can still inspect and convert the voxel model.");
    return;
  }
  const loadToken = record.loadToken;
  new GLTFLoader().load(modelUrl, (gltf) => {
    if (state.renderers.voxel?.loadToken !== loadToken) {
      disposeObject(gltf.scene);
      return;
    }
    gltf.scene.traverse((item) => {
      if (!item.isMesh) return;
      item.castShadow = true;
      item.receiveShadow = true;
    });
    scenes[0].add(gltf.scene);
    centerAndNormalize(gltf.scene);
  }, undefined, () => {
    if (state.renderers.voxel?.loadToken === loadToken) {
      showModelFallback(modelContainer, "The generated GLB could not be loaded. The voxel preview and brick conversion still work.");
    }
  });
}

function showModelFallback(container, message) {
  container.querySelector(".viewer-error")?.remove();
  const fallback = document.createElement("p");
  fallback.className = "viewer-error";
  fallback.innerHTML = "<strong>3D preview unavailable</strong><span></span>";
  fallback.querySelector("span").textContent = message;
  container.append(fallback);
}

function createViewer(kind, items, elementId) {
  disposeViewer(kind);
  const container = document.getElementById(elementId);
  if (!container || !items?.length) {
    if (container) container.innerHTML = '<p class="empty-parts" style="padding:24px">No geometry was returned.</p>';
    return;
  }
  container.innerHTML = "";
  const scene = new THREE.Scene();
  setSceneLighting(scene);
  const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, .1, 300);
  const renderer = createRenderer(container);

  const group = new THREE.Group();
  scene.add(group);
  if (kind === "voxel") addVoxels(group, items);
  else addBricks(group, items);

  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  group.position.sub(center);
  const maxSize = Math.max(size.x, size.y, size.z, 1);
  camera.position.set(maxSize * 1.35, maxSize * .95, maxSize * 1.55);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = .065;
  controls.minDistance = maxSize * .8;
  controls.maxDistance = maxSize * 4.5;
  controls.target.set(0, 0, 0);
  controls.update();

  const grid = new THREE.GridHelper(Math.max(30, Math.ceil(maxSize * 2.4)), Math.max(12, Math.ceil(maxSize * 1.2)), 0x8c877e, 0xc8c0b4);
  grid.position.y = -size.y / 2 - .56;
  scene.add(grid);

  const resize = new ResizeObserver(() => {
    if (!container.clientWidth || !container.clientHeight) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
  resize.observe(container);
  const record = { scene, camera, renderer, controls, frame: 0, resize, group, initial: camera.position.clone(), disposed: false };
  const loop = () => {
    if (record.disposed) return;
    controls.update();
    renderer.render(scene, camera);
    record.frame = requestAnimationFrame(loop);
  };
  state.renderers[kind] = record;
  loop();
}

function addVoxels(group, voxels) {
  // Leave a narrow, physical gap between cells. The warm canvas shows through
  // as a crisp seam without tinting or darkening the voxel's supplied colour.
  const geometry = new THREE.BoxGeometry(.9, .9, .9);
  const material = createVoxelMaterial();
  const mesh = new THREE.InstancedMesh(geometry, material, voxels.length);
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  voxels.forEach((voxel, index) => {
    matrix.makeTranslation(Number(voxel.x) || 0, Number(voxel.z) || 0, Number(voxel.y) || 0);
    mesh.setMatrixAt(index, matrix);
    color.set(cssColor(voxel.color || voxel.hex || (index % 7 === 0 ? "#e8b843" : "#c84832")));
    mesh.setColorAt(index, color);
  });
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  group.add(mesh);
}

function createVoxelMaterial() {
  // InstancedMesh supplies each voxel's tint through instanceColor. Keep the
  // material's base multiplier white and do not request a missing per-vertex
  // geometry colour attribute, which can multiply valid instance colours down
  // to black on some Three.js/WebGL code paths.
  return new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
}

function addBricks(group, bricks) {
  const groups = new Map();
  bricks.forEach((brick) => {
    const w = Number(brick.w || brick.width || 1);
    const d = Number(brick.d || brick.depth || 1);
    const h = Number(brick.h || brick.height || 1);
    const key = `${w},${d},${h}`;
    if (!groups.has(key)) groups.set(key, { w, d, h, bricks: [] });
    groups.get(key).bricks.push(brick);
  });
  for (const { w, d, h, bricks: same } of groups.values()) {
    const geometry = new THREE.BoxGeometry(w * .97, h * .72, d * .97);
    const material = new THREE.MeshStandardMaterial({ roughness: .48, metalness: .02 });
    const mesh = new THREE.InstancedMesh(geometry, material, same.length);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    same.forEach((brick, index) => {
      const x = Number(brick.x) || 0;
      const y = Number(brick.y) || 0;
      const z = Number(brick.z) || 0;
      matrix.makeTranslation(x + (w - 1) / 2, z * .72, y + (d - 1) / 2);
      mesh.setMatrixAt(index, matrix);
      color.set(cssColor(brick.color || brick.hex || "#c84832"));
      mesh.setColorAt(index, color);
    });
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    group.add(mesh);

    const studGeometry = new THREE.CylinderGeometry(.29, .29, .16, 16);
    const studs = new THREE.InstancedMesh(studGeometry, material, same.length * w * d);
    let studIndex = 0;
    same.forEach((brick) => {
      const bx = Number(brick.x) || 0;
      const by = Number(brick.y) || 0;
      const bz = Number(brick.z) || 0;
      for (let ix = 0; ix < w; ix++) for (let iy = 0; iy < d; iy++) {
        matrix.makeTranslation(bx + ix, bz * .72 + h * .36 + .08, by + iy);
        studs.setMatrixAt(studIndex, matrix);
        color.set(cssColor(brick.color || brick.hex || "#c84832"));
        studs.setColorAt(studIndex++, color);
      }
    });
    studs.castShadow = true;
    if (studs.instanceColor) studs.instanceColor.needsUpdate = true;
    group.add(studs);
  }
}

function disposeViewer(kind) {
  const record = state.renderers[kind];
  if (!record) return;
  record.disposed = true;
  cancelAnimationFrame(record.frame);
  record.resize.disconnect();
  const controls = Array.isArray(record.controls) ? record.controls : [record.controls];
  controls.forEach((control, index) => {
    if (record.syncHandlers) control.removeEventListener("change", record.syncHandlers[index]);
    control.dispose();
  });
  const scenes = record.scenes || [record.scene];
  scenes.forEach(disposeObject);
  const renderers = record.renderers || [record.renderer];
  renderers.forEach((renderer) => {
    renderer.dispose();
    renderer.domElement.remove();
  });
  delete state.renderers[kind];
}

function disposeObject(root) {
  root?.traverse?.((item) => {
    item.geometry?.dispose?.();
    const materials = Array.isArray(item.material) ? item.material : item.material ? [item.material] : [];
    materials.forEach((material) => {
      Object.values(material).forEach((value) => value?.isTexture && value.dispose());
      material.dispose();
    });
  });
}

function resetCamera(kind) {
  const record = state.renderers[kind];
  if (!record) return;
  const cameras = record.cameras || [record.camera];
  const controls = Array.isArray(record.controls) ? record.controls : [record.controls];
  cameras.forEach((camera) => camera.position.copy(record.initial));
  controls.forEach((control) => {
    control.target.set(0, 0, 0);
    control.update();
  });
}

async function recoverJob() {
  const id = sessionStorage.getItem(STORAGE_KEY);
  if (!id) return;
  showProcessing("queued", 2, "Reopening your workshop…");
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    applyJob(job);
    if (ACTIVE_STAGES.has(job.stage)) startPolling();
  } catch (error) {
    if (isMissingJob(error)) {
      state.job = null;
      state.lastAction = null;
      sessionStorage.removeItem(STORAGE_KEY);
      showView("landing");
      return;
    }
    state.lastAction = "recover";
    fail(error.message);
  }
}

document.querySelector("#home-button").addEventListener("click", () => showView(state.job ? (stageToView[state.job.stage] || "landing") : "landing"));
document.querySelector("#start-button").addEventListener("click", () => showView("upload"));
document.querySelector("#image-input").addEventListener("change", (event) => inspectImage(event.target.files[0]));
document.querySelector("#upload-form").addEventListener("submit", uploadImage);
document.querySelector("#legolize-button").addEventListener("click", startLegolize);
document.querySelector("#manual-button").addEventListener("click", generateManual);
document.querySelector("#retry-button").addEventListener("click", retry);
document.querySelectorAll('[data-action="restart"]').forEach((button) => button.addEventListener("click", resetBuild));
document.querySelector('[data-action="back-to-voxel"]').addEventListener("click", () => { showView("voxel"); populateVoxel(state.job); });
document.querySelectorAll("[data-reset-viewer]").forEach((button) => button.addEventListener("click", () => resetCamera(button.dataset.resetViewer)));

const dropZone = document.querySelector("#drop-zone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); }));
dropZone.addEventListener("drop", (event) => inspectImage(event.dataTransfer.files[0]));
window.addEventListener("beforeunload", () => { disposeViewer("voxel"); disposeViewer("lego"); });

showView("landing");
recoverJob();
