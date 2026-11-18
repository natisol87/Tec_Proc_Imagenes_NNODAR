"use strict";

/*
 * Pastiche frontend.
 *
 * The server does not keep information: each step is a POST to /api/steps/{name} with the
 * factors and the layers that the step needs. This file keeps the state in the browser:
 * the photo, the result of each SAVED step (its layers as PNG blobs) and the
 * factors selected by the user. The state is also saved in IndexedDB, so the work
 * is not lost when the page is reloaded.
 */

const MAX_SIDE = 1600;          // photos are reduced before the upload (Vercel accepts up to 4.5 MB)
const AUTO_APPLY_DELAY = 350;   // ms to wait after a change in a factor before applying the step
const SCALARS = ["face"];  // results sent as values, not as images

const state = {
  steps: [],      // from GET /api/steps
  palettes: [],   // from GET /api/palettes
  colors: [],     // from GET /api/colors: the inks that the user can select for a zone
  photo: null,    // Blob (JPEG)
  saved: {},      // name of the step → {factors, preview, layers, face, meta}
  drafts: {},     // name of the step → factors in edition
  current: null,  // name of the step visible on the screen
  result: null,   // last result of the current step: applied but NOT saved
  busy: false,
  requestId: 0,
};

const $ = (selector) => document.querySelector(selector);
const urls = new WeakMap();  // Blob → object URL (only one URL for each blob)
const blobUrl = (blob) => {
  if (!urls.has(blob)) urls.set(blob, URL.createObjectURL(blob));
  return urls.get(blob);
};

// ===========================================================================
// Start
// ===========================================================================

async function init() {
  bindSources();
  bindWorkbench();
  try {
    const [steps, palettes, colors] = await Promise.all([api("/api/steps"), api("/api/palettes"), api("/api/colors")]);
    state.steps = steps;
    state.palettes = palettes;
    state.colors = colors;
  } catch (error) {
    toast(`No se pudo conectar con el servidor: ${error.message}`, "error");
    return;
  }
  await restoreSession();
  state.current ||= state.steps[0].name;
  render();
  if (state.photo && !state.saved[state.current]) applyCurrent();
}

// ===========================================================================
// Origin of the photo: file, camera, paste, link
// ===========================================================================

function bindSources() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
  loadExamples();

  const drop = $("#drop");
  $("#file-input").addEventListener("change", (e) => e.target.files[0] && loadPhoto(e.target.files[0]));
  ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (e) => {
    e.preventDefault();
    drop.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (e) => {
    e.preventDefault();
    drop.classList.remove("dragging");
  }));
  // Drop on any part of the page: a file, or an image from another tab (it arrives as a link).
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    handleDataTransfer(e.dataTransfer);
  });

  $("#camera-start").addEventListener("click", startCamera);
  $("#camera-shoot").addEventListener("click", takePicture);
  $("#camera-flip").addEventListener("click", flipCamera);
  $("#camera-input").addEventListener("change", (e) => e.target.files[0] && loadPhoto(e.target.files[0]));

  document.addEventListener("paste", (e) => {
    // The user can paste a link in the URL field (the target is `document` when no element has the focus).
    if (e.target instanceof Element && e.target.matches("input[type=url]")) return;
    e.preventDefault();
    handleDataTransfer(e.clipboardData);
    $("#paste-box").classList.add("flash");
    setTimeout(() => $("#paste-box").classList.remove("flash"), 300);
  });
  $("#paste-button").addEventListener("click", pasteFromButton);

  $("#url-form").addEventListener("submit", (e) => {
    e.preventDefault();
    loadFromUrl($("#url-input").value);
  });

  $("#reset").addEventListener("click", resetAll);
}

/** Example photos included in the application (frontend/examples/examples.json), to test it without a photo. */
async function loadExamples() {
  const grid = $("#examples");
  try {
    const examples = await (await fetch("examples/examples.json")).json();
    grid.innerHTML = examples.map((e, i) => `<button class="example" data-i="${i}" title="${e.credit}">
      <img src="examples/${e.file}" alt="${e.label}" loading="lazy"><span>${e.label}</span></button>`).join("");
    grid.querySelectorAll(".example").forEach((b) => b.addEventListener("click", async () => {
      const example = examples[Number(b.dataset.i)];
      const blob = await (await fetch(`examples/${example.file}`)).blob();
      loadPhoto(new File([blob], example.file, { type: blob.type }));
    }));
  } catch {
    grid.innerHTML = `<p class="hint">No se pudieron cargar los ejemplos.</p>`;
  }
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
    t.setAttribute("aria-selected", t.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== name));
  if (name !== "camera") stopCamera();
  if (name === "paste") $("#paste-box").focus();
}

function handleDataTransfer(data) {
  if (!data) return;
  const file = [...data.files].find((f) => f.type.startsWith("image/"));
  if (file) return loadPhoto(file);
  const imageItem = [...(data.items || [])].find((i) => i.kind === "file" && i.type.startsWith("image/"));
  if (imageItem) return loadPhoto(imageItem.getAsFile());

  const html = data.getData("text/html");
  const fromHtml = html && new DOMParser().parseFromString(html, "text/html").querySelector("img")?.src;
  const link = fromHtml || data.getData("text/uri-list") || data.getData("text/plain");
  if (link && /^https?:\/\//i.test(link.trim())) return loadFromUrl(link.trim());
  toast("Eso no parece una imagen. Copiá la imagen (no la página) o su dirección.", "error");
}

async function pasteFromButton() {
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (type) return loadPhoto(await item.getType(type));
      if (item.types.includes("text/plain")) {
        const text = (await (await item.getType("text/plain")).text()).trim();
        if (/^https?:\/\//i.test(text)) return loadFromUrl(text);
      }
    }
    toast("El portapapeles no tiene ninguna imagen.", "error");
  } catch {
    toast("Tu navegador no deja leer el portapapeles con el botón: hacé clic en el recuadro y usá Ctrl + V.", "error");
  }
}

async function loadFromUrl(url) {
  if (!url) return;
  toast("Descargando la imagen…");
  try {
    const data = await api("/api/photo/from-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    await loadPhoto(await dataUrlToBlob(data.image));
  } catch (error) {
    toast(error.message, "error");
  }
}

// ----- Camera -----
let stream = null;
let facingMode = "user";

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    return toast("Este navegador no permite usar la cámara. Probá con «Abrir la cámara del teléfono».", "error");
  }
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode, width: { ideal: 1280 } }, audio: false });
  } catch {
    return toast("No se pudo encender la cámara. Revisá los permisos del navegador.", "error");
  }
  const video = $("#camera-video");
  video.srcObject = stream;
  video.classList.toggle("back", facingMode === "environment");
  await video.play();
  $("#camera-empty").hidden = true;
  $("#camera-shoot").disabled = false;
  $("#camera-flip").disabled = false;
}

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $("#camera-empty").hidden = false;
  $("#camera-shoot").disabled = true;
  $("#camera-flip").disabled = true;
}

function flipCamera() {
  facingMode = facingMode === "user" ? "environment" : "user";
  startCamera();
}

function takePicture() {
  const video = $("#camera-video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    stopCamera();
    loadPhoto(blob);
  }, "image/jpeg", 0.92);
}

// ----- Common entry for all the origins -----
async function loadPhoto(file) {
  if (!file) return;
  if (Object.keys(state.saved).length && !confirm("Vas a empezar un cuadro nuevo. Los pasos guardados se borran (tus factores se mantienen). ¿Seguimos?")) {
    return;
  }
  let blob;
  try {
    blob = await shrink(file);
  } catch {
    return toast("No se pudo leer esa imagen. Probá con JPG, PNG o WEBP.", "error");
  }
  state.photo = blob;
  state.saved = {};
  if (state.drafts.frame) state.drafts.frame.face = 0;  // the faces of the previous photo are not valid now
  state.result = null;
  state.current = state.steps[0].name;
  await saveSession();
  render();
  toast("¡Foto cargada! Ajustá los factores de cada paso y guardalo.", "ok");
  applyCurrent();
}

async function shrink(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";  // transparent PNGs go over a white background
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject()), "image/jpeg", 0.92));
}

// ===========================================================================
// Steps: inputs, application, saving
// ===========================================================================

const stepByName = (name) => state.steps.find((s) => s.name === name);
const stepIndex = (name) => state.steps.findIndex((s) => s.name === name);

function defaultFactors(step) {
  return Object.fromEntries(step.factors.map((f) => [f.name, structuredClone(f.schema.default)]));
}

function factorsOf(name) {
  // First the default values, so the factors added in a new version of the application have a value, also when the
  // browser restored factors saved by an old version.
  // The same object is completed directly: the controls on the screen use a reference to it.
  const defaults = defaultFactors(stepByName(name));
  const values = (state.drafts[name] ||= {});
  for (const key of Object.keys(values)) if (!(key in defaults)) delete values[key];
  for (const [key, value] of Object.entries(defaults)) if (!(key in values)) values[key] = value;
  return values;
}

/** For each layer that the step needs, finds the most recent previous SAVED step that produced it. */
function inputsFor(step) {
  const inputs = {};
  const missing = [];
  const earlier = state.steps.slice(0, stepIndex(step.name)).reverse();
  for (const field of [...step.requires, ...step.optional]) {
    // Step 5 only uses the photo when the zones are transparent: if not, it is not sent (smaller request).
    if (step.name === "compose" && field === "image" && (factorsOf(step.name).opacity ?? 1) >= 1) continue;
    if (field === "photo") {
      if (state.photo) inputs.photo = state.photo;
      else missing.push("la foto");
      continue;
    }
    const source = earlier.find((s) => s.produces.includes(field) && state.saved[s.name]);
    const saved = source && state.saved[source.name];
    const value = saved && (SCALARS.includes(field) ? saved[field] : saved.layers[field]);
    if (value !== undefined && value !== null) inputs[field] = value;
    else if (step.requires.includes(field)) {
      const producer = earlier.find((s) => s.produces.includes(field));
      missing.push(producer ? `el paso ${producer.number} (${producer.label})` : field);
    }
  }
  return { inputs, missing: [...new Set(missing)] };
}

function status(step) {
  if (state.saved[step.name]) return "saved";
  return inputsFor(step).missing.length ? "locked" : "available";
}

async function applyCurrent() {
  const step = stepByName(state.current);
  const { inputs, missing } = inputsFor(step);
  if (missing.length) return;

  const form = new FormData();
  form.append("params", JSON.stringify(factorsOf(step.name)));
  for (const [field, value] of Object.entries(inputs)) {
    if (field === "face") form.append("face", JSON.stringify(value));
    else if (SCALARS.includes(field)) form.append(field, value);
    else form.append(field, value, `${field}.png`);
  }

  const id = ++state.requestId;  // only the response to the LAST request is shown
  setBusy(true);
  try {
    const data = await api(`/api/steps/${step.name}`, { method: "POST", body: form });
    if (id !== state.requestId) return;
    const layers = {};
    for (const [field, url] of Object.entries(data.layers)) layers[field] = await dataUrlToBlob(url);
    state.result = {
      factors: structuredClone(factorsOf(step.name)),
      preview: await dataUrlToBlob(data.preview ?? data.layers[data.preview_layer]),
      layers,
      face: data.face ?? null,
      meta: { elapsed: data.elapsed_ms, colors: data.colors, width: data.width, height: data.height, face: data.face,
              faces: data.faces ?? null },
    };
  } catch (error) {
    if (id !== state.requestId) return;
    state.result = null;
    toast(error.message, "error");
  } finally {
    if (id === state.requestId) setBusy(false);
  }
  renderStage();
  renderActions();
  const choice = $("#face-choice");
  if (choice && step.name === "frame") fillFaceChoice(choice, factorsOf("frame"));
}

async function saveCurrent({ advance = true } = {}) {
  const step = stepByName(state.current);
  if (!state.result) return;
  state.saved[step.name] = state.result;
  state.result = null;
  // The next steps were created from the previous version of this step: they must be executed again.
  const dropped = state.steps.slice(stepIndex(step.name) + 1).filter((s) => state.saved[s.name]);
  dropped.forEach((s) => delete state.saved[s.name]);
  if (dropped.length) toast(`Guardado. Rehacé ${dropped.length === 1 ? "el paso siguiente" : `los ${dropped.length} pasos siguientes`}: dependían de este.`, "ok");

  const next = state.steps[stepIndex(step.name) + 1];
  if (advance && next) state.current = next.name;
  await saveSession();
  render();
  if (advance && next && !state.saved[next.name]) await applyCurrent();
  if (!next && advance) toast("¡Tu cuadro está listo! Descargalo desde la vista previa.", "ok");
}

async function completeRemaining() {
  for (const step of state.steps) {
    if (state.saved[step.name]) continue;
    state.current = step.name;
    render();
    await applyCurrent();
    if (!state.result) return;  // a step failed: stop, so the user can correct its factors
    await saveCurrent({ advance: false });
  }
  state.current = state.steps.at(-1).name;
  render();
  toast("¡Listo! Podés volver a cualquier paso para retocarlo.", "ok");
}

function selectStep(name) {
  const step = stepByName(name);
  const { missing } = inputsFor(step);
  if (missing.length && !state.saved[name]) {
    return toast(`Para este paso primero guardá ${missing.join(" y ")}.`, "error");
  }
  state.current = name;
  state.result = null;
  saveSession();
  render();
  if (!state.saved[name]) applyCurrent();
}

// ===========================================================================
// Drawing of the interface
// ===========================================================================

function render() {
  $("#reset").hidden = !state.photo;
  renderPhoto();
  renderStepper();
  renderFactors();
  renderStage();
  renderActions();
  renderSaved();
}

function renderPhoto() {
  const thumb = $("#photo-thumb");
  thumb.innerHTML = state.photo
    ? `<img src="${blobUrl(state.photo)}" alt="Foto original">`
    : `<div class="empty-thumb">Todavía no hay foto</div>`;
}

function renderStepper() {
  $("#stepper").innerHTML = state.steps.map((step) => {
    const st = status(step);
    const label = { saved: "guardado ✓", available: "listo para hacer", locked: "bloqueado" }[st];
    const classes = ["step-chip", st, step.name === state.current ? "current" : ""].join(" ");
    return `<button class="${classes}" data-step="${step.name}">
      <span class="num">${step.number}</span>
      <span class="text"><span>${step.label}</span><span class="state">${step.name === state.current ? "en pantalla" : label}</span></span>
    </button>`;
  }).join("");
  document.querySelectorAll(".step-chip").forEach((chip) => chip.addEventListener("click", () => selectStep(chip.dataset.step)));
}

function renderFactors() {
  const step = stepByName(state.current);
  $("#step-title").textContent = `Paso ${step.number} · ${step.label}`;
  $("#step-desc").textContent = step.description;
  const values = factorsOf(step.name);
  const list = $("#factor-list");
  list.innerHTML = "";
  for (const factor of step.factors) {
    if (factor.schema.kind === "custom") continue;  // edited inside the palette selector
    list.append(factorControl(step, factor, values));
  }
}

function factorControl(step, factor, values) {
  const s = factor.schema;
  const box = document.createElement("div");
  box.className = "factor";
  const id = `f-${step.name}-${factor.name}`;
  const help = s.description ? `<small>${s.description}</small>` : "";

  if (s.kind === "face") {
    box.id = "face-choice";
    box.innerHTML = `<div class="factor-title"><span>${s.title}</span></div>${help}<div class="faces"></div>`;
    fillFaceChoice(box, values);
  } else if (s.kind === "palettes") {
    box.innerHTML = `<div class="factor-title"><span>${s.title}</span></div>${help}`;
    box.append(palettePicker(values));
  } else if (s.enum) {
    box.innerHTML = `<div class="factor-title"><span>${s.title}</span></div>${help}`;
    const group = document.createElement("div");
    group.className = "segmented";
    for (const option of s.enum) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = s.options?.[option] ?? option;
      b.className = values[factor.name] === option ? "on" : "";
      b.addEventListener("click", () => {
        values[factor.name] = option;
        group.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        factorChanged();
      });
      group.append(b);
    }
    box.append(group);
  } else if (s.type === "boolean") {
    box.innerHTML = `<label class="switch" for="${id}"><span>${s.title}</span><input type="checkbox" id="${id}" ${values[factor.name] ? "checked" : ""}></label>${help}`;
    box.querySelector("input").addEventListener("change", (e) => {
      values[factor.name] = e.target.checked;
      factorChanged();
    });
  } else {
    const step_ = s.step ?? (s.type === "integer" ? 1 : 0.1);
    const decimals = String(step_).split(".")[1]?.length ?? 0;
    const unit = s.unit ? ` ${s.unit}` : "";
    const show = (v) => `${Number(v).toFixed(decimals)}${unit}`;
    box.innerHTML = `<label for="${id}"><span>${s.title}</span><span class="value">${show(values[factor.name])}</span></label>
      <input type="range" id="${id}" min="${s.minimum}" max="${s.maximum}" step="${step_}" value="${values[factor.name]}">${help}`;
    const input = box.querySelector("input");
    input.addEventListener("input", () => {
      values[factor.name] = Number(input.value);
      box.querySelector(".value").textContent = show(input.value);
      factorChanged();
      if (factor.name === "rows" || factor.name === "cols") refreshPalettePicker(values);
    });
  }
  return box;
}

// ---------------------------------------------------------------------------
// Selection of the face when the photo has several faces
// ---------------------------------------------------------------------------

function photoFaces() {
  const entry = state.current === "frame" && state.result ? state.result : state.saved.frame;
  return entry?.meta?.faces ?? [];
}

/** Buttons "La más grande", 1, 2… each one with a small image of that face. Not visible when there is only one face. */
function fillFaceChoice(box, values) {
  const faces = photoFaces();
  box.hidden = faces.length < 2;
  const group = box.querySelector(".faces");
  group.innerHTML = "";
  if (faces.length < 2) return;
  const options = [{ value: 0, label: "La más grande" }, ...faces.map((f, i) => ({ value: i + 1, label: `${i + 1}`, face: f }))];
  for (const option of options) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `face-option${values.face === option.value ? " on" : ""}`;
    b.innerHTML = option.face ? `<canvas width="56" height="56"></canvas><span>${option.label}</span>` : `<span>${option.label}</span>`;
    if (option.face) drawFace(b.querySelector("canvas"), option.face);
    b.addEventListener("click", () => {
      values.face = option.value;
      group.querySelectorAll(".face-option").forEach((x) => x.classList.toggle("on", x === b));
      factorChanged();
    });
    group.append(b);
  }
}

const photoBitmaps = new WeakMap();  // Blob of the photo → ImageBitmap (decoded only once)
async function drawFace(canvas, face) {
  if (!photoBitmaps.has(state.photo)) photoBitmaps.set(state.photo, createImageBitmap(state.photo));
  const bitmap = await photoBitmaps.get(state.photo);
  // The rectangle of the face arrives as fractions of the photo; a small margin is added around it.
  const side = Math.max(face.width * bitmap.width, face.height * bitmap.height) * 1.3;
  const cx = (face.x + face.width / 2) * bitmap.width;
  const cy = (face.y + face.height / 2) * bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, cx - side / 2, cy - side / 2, side, side, 0, 0, canvas.width, canvas.height);
}

function palettePicker(values) {
  const wrap = document.createElement("div");
  wrap.className = "palette-picker";
  wrap.append(paletteGrid(values));
  const shuffle = document.createElement("button");
  shuffle.type = "button";
  shuffle.className = "ghost small";
  shuffle.textContent = "Mezclar paletas";
  shuffle.style.marginTop = "8px";
  shuffle.addEventListener("click", () => {
    values.palettes = [...state.palettes.map((p) => p.name)].sort(() => Math.random() - 0.5);
    values.custom = [];  // new palettes: their own colors again
    refreshPalettePicker(values);
    factorChanged();
  });
  wrap.append(shuffle);
  return wrap;
}

// Zones where the color can be selected in each portrait, with their position in palette.zones.
const CUSTOM_ZONES = [
  { key: "hair", label: "Pelo", index: 0 },
  { key: "skin", label: "Piel", index: 1 },
  { key: "lips", label: "Labios", index: 3 },
  { key: "eyes", label: "Ojos", index: 4 },
];

function paletteGrid(values) {
  const cells = values.rows * values.cols;
  const names = state.palettes.map((p) => p.name);
  while (values.palettes.length < cells) values.palettes.push(names[values.palettes.length % names.length]);
  values.custom ||= [];
  while (values.custom.length < cells) values.custom.push({});
  const grid = document.createElement("div");
  grid.className = "palette-grid";
  grid.style.gridTemplateColumns = `repeat(${values.cols}, 1fr)`;
  for (let i = 0; i < cells; i++) {
    const cell = document.createElement("div");
    cell.className = "palette-cell";
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Paleta del cuadro ${i + 1}`);
    select.innerHTML = state.palettes.map((p) => `<option value="${p.name}">${p.label}</option>`).join("");
    select.value = values.palettes[i];
    const swatch = document.createElement("div");
    swatch.className = "swatch";
    const pickers = document.createElement("div");
    pickers.className = "zone-colors";

    // The colors of this portrait: the palette, with the colors selected by the user in place of the original ones.
    const colorsOf = () => {
      const zones = [...state.palettes.find((x) => x.name === select.value).zones];
      for (const z of CUSTOM_ZONES) if (values.custom[i][z.key]) zones[z.index] = values.custom[i][z.key];
      return zones;
    };
    const paint = () => {
      const p = state.palettes.find((x) => x.name === select.value);
      // Background, then hair, skin, clothes, lips and eye shadow.
      swatch.innerHTML = [p.background, ...colorsOf()].map((c) => `<span style="background:${c}"></span>`).join("");
      pickers.querySelectorAll(".zone-chip").forEach((chip) => {
        chip.querySelector(".dot").style.background = colorsOf()[Number(chip.dataset.index)];
      });
    };
    for (const zone of CUSTOM_ZONES) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "zone-chip";
      chip.dataset.index = zone.index;
      chip.setAttribute("aria-label", `Color de ${zone.label.toLowerCase()} del cuadro ${i + 1}`);
      chip.innerHTML = `<span class="dot"></span>${zone.label}`;
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        openSwatchMenu(chip, (color) => {
          values.custom[i][zone.key] = color;
          paint();
          factorChanged();
        });
      });
      pickers.append(chip);
    }
    paint();
    select.addEventListener("change", () => {
      values.palettes[i] = select.value;
      values.custom[i] = {};  // a new palette has its own colors
      paint();
      factorChanged();
    });
    cell.append(swatch, select, pickers);
    grid.append(cell);
  }
  return grid;
}

// A small grid with the predefined inks, below the button that opened it.
function openSwatchMenu(anchor, onPick) {
  closeSwatchMenu();
  const menu = document.createElement("div");
  menu.className = "swatch-menu";
  menu.setAttribute("role", "listbox");
  for (const color of state.colors) {
    const b = document.createElement("button");
    b.type = "button";
    b.title = color.label;
    b.setAttribute("aria-label", color.label);
    b.style.background = color.hex;
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      onPick(color.hex);
      closeSwatchMenu();
    });
    menu.append(b);
  }
  anchor.after(menu);
  menu.querySelector("button")?.focus();
}

function closeSwatchMenu() {
  document.querySelectorAll(".swatch-menu").forEach((m) => m.remove());
}

document.addEventListener("click", closeSwatchMenu);
document.addEventListener("keydown", (e) => e.key === "Escape" && closeSwatchMenu());

function refreshPalettePicker(values) {
  const picker = document.querySelector(".palette-picker");
  if (picker) picker.firstElementChild.replaceWith(paletteGrid(values));
}

let applyTimer = null;
function factorChanged() {
  saveSession();
  clearTimeout(applyTimer);
  applyTimer = setTimeout(applyCurrent, AUTO_APPLY_DELAY);
}

function currentView() {
  const name = state.current;
  if (state.result) return { entry: state.result, kind: "draft" };
  if (state.saved[name]) return { entry: state.saved[name], kind: "saved" };
  return { entry: null, kind: "none" };
}

function beforeImage() {
  const index = stepIndex(state.current);
  if (index === 0) return state.photo;
  return state.saved[state.steps[index - 1].name]?.preview ?? null;
}

function renderStage() {
  const step = stepByName(state.current);
  const { entry, kind } = currentView();
  const frame = $("#stage-frame");
  const badge = $("#stage-badge");
  const { missing } = inputsFor(step);

  if (!state.photo) {
    frame.innerHTML = `<div class="stage-empty">Cargá un retrato para empezar</div>`;
  } else if (!entry) {
    frame.innerHTML = `<div class="stage-empty">${missing.length ? `Primero guardá ${missing.join(" y ")}` : "Aplicando…"}</div>`;
  } else {
    frame.innerHTML = `<img src="${blobUrl(entry.preview)}" alt="Resultado del paso ${step.label}">`;
  }

  badge.className = `badge ${kind === "saved" ? "saved" : kind === "draft" ? "draft" : ""}`;
  badge.textContent = { saved: "guardado", draft: "sin guardar", none: "sin aplicar" }[kind];

  const meta = entry?.meta;
  const faceText = meta?.face ? " · cara detectada" : "";
  $("#stage-meta").textContent = meta ? `${meta.width}×${meta.height} px · ${meta.colors} colores · ${meta.elapsed} ms${faceText}` : "";

  const link = $("#download-step");
  link.hidden = !entry;
  if (entry) {
    const isFinal = step.name === state.steps.at(-1).name;
    link.href = blobUrl(entry.preview);
    const ext = entry.preview.type === "image/jpeg" ? "jpg" : "png";
    link.download = isFinal ? "pastiche-cuadro.png" : `pastiche-paso-${step.number}-${step.name}.${ext}`;
    link.textContent = isFinal ? "Descargar el cuadro" : "Descargar este paso";
  }
  $("#compare").disabled = !entry || !beforeImage();
}

function renderActions() {
  const step = stepByName(state.current);
  const saveButton = $("#save-step");
  saveButton.disabled = !state.result || state.busy;
  saveButton.textContent = state.saved[step.name] && state.result ? "Guardar cambios" : "Guardar paso";
  const remaining = state.steps.filter((s) => !state.saved[s.name]).length;
  $("#auto-all").disabled = !state.photo || state.busy || remaining === 0;
  $("#auto-all").textContent = remaining === 0 ? "Todos los pasos están guardados"
    : remaining === 1 ? "Completar el paso que falta" : `Completar los ${remaining} pasos que faltan`;
  const hint = state.saved[step.name] && !state.result
    ? "Este paso ya está guardado. Si cambiás un factor se vuelve a aplicar y podés guardarlo de nuevo."
    : "Cada cambio se aplica solo. Cuando te guste, guardá el paso para pasar al siguiente.";
  $("#step-hint").textContent = state.photo ? hint : "";
}

function renderSaved() {
  const strip = $("#saved-strip");
  const saved = state.steps.filter((s) => state.saved[s.name]);
  if (!saved.length) {
    strip.innerHTML = `<p class="hint">Cada paso que guardes aparece acá. Podés volver a cualquiera, cambiar sus factores y guardarlo de nuevo.</p>`;
    return;
  }
  strip.innerHTML = saved.map((step) => {
    const url = blobUrl(state.saved[step.name].preview);
    return `<div class="saved-card">
      <button class="thumb" data-step="${step.name}" aria-label="Ver el paso ${step.label}"><img src="${url}" alt=""></button>
      <div class="caption"><span>${step.number} · ${step.label}</span><a href="${url}" download="pastiche-paso-${step.number}-${step.name}.${state.saved[step.name].preview.type === "image/jpeg" ? "jpg" : "png"}">${state.saved[step.name].preview.type === "image/jpeg" ? "JPG" : "PNG"}</a></div>
    </div>`;
  }).join("");
  strip.querySelectorAll("button.thumb").forEach((b) => b.addEventListener("click", () => selectStep(b.dataset.step)));
}

function setBusy(busy) {
  state.busy = busy;
  $("#stage-frame").classList.toggle("busy", busy);
  $("#stage-badge").className = busy ? "badge working" : $("#stage-badge").className;
  if (busy) $("#stage-badge").textContent = "aplicando…";
  renderActions();
}

function bindWorkbench() {
  $("#save-step").addEventListener("click", () => saveCurrent());
  $("#reset-factors").addEventListener("click", () => {
    const step = stepByName(state.current);
    state.drafts[step.name] = defaultFactors(step);
    renderFactors();
    factorChanged();
  });
  $("#auto-all").addEventListener("click", completeRemaining);

  // "Mantener para ver el antes": shows the previous image while the button is pressed.
  const compare = $("#compare");
  const showBefore = () => {
    const img = $("#stage-frame img");
    const before = beforeImage();
    if (img && before) img.src = blobUrl(before);
  };
  const showAfter = () => renderStage();
  compare.addEventListener("pointerdown", showBefore);
  ["pointerup", "pointerleave", "pointercancel"].forEach((t) => compare.addEventListener(t, showAfter));
  compare.addEventListener("keydown", (e) => (e.key === " " || e.key === "Enter") && showBefore());
  compare.addEventListener("keyup", showAfter);
}

async function resetAll() {
  if (!confirm("¿Borrar la foto, los pasos guardados y los factores?")) return;
  state.photo = null;
  state.saved = {};
  state.drafts = {};
  state.result = null;
  state.current = state.steps[0].name;
  await saveSession();
  render();
}

// ===========================================================================
// Auxiliary functions: API, blobs, messages
// ===========================================================================

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    throw new Error(detail?.message || (typeof detail === "string" ? detail : `Error ${response.status}`));
  }
  return data;
}

async function dataUrlToBlob(url) {
  return (await fetch(url)).blob();
}

let toastTimer = null;
function toast(message, kind = "") {
  const box = $("#toast");
  box.textContent = message;
  box.className = `toast ${kind}`;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), kind === "error" ? 6000 : 3500);
}

// ===========================================================================
// Saving the work in the browser (IndexedDB can save blobs, localStorage cannot)
// ===========================================================================

const DB_NAME = "pastiche";
const STORE = "session";

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveSession() {
  try {
    const db = await openDb();
    const snapshot = { photo: state.photo, saved: state.saved, drafts: state.drafts, current: state.current };
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(snapshot, "current");
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    /* private mode or blocked storage: the application works, but the work is lost when the page is reloaded */
  }
}

async function restoreSession() {
  try {
    const db = await openDb();
    const snapshot = await new Promise((resolve, reject) => {
      const request = db.transaction(STORE).objectStore(STORE).get("current");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    if (!snapshot) return;
    Object.assign(state, {
      photo: snapshot.photo ?? null,
      saved: snapshot.saved ?? {},
      drafts: snapshot.drafts ?? {},
      current: stepByName(snapshot.current) ? snapshot.current : null,
    });
    if (state.photo) toast("Recuperamos tu trabajo guardado.", "ok");
  } catch {
    /* nothing saved, or blocked storage */
  }
}

init();
