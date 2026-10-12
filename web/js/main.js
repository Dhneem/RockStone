// Rockstone CT Workbench - main application module.
import { get, post, del, upload } from "./api.js?v=2.1.3";
import { SliceViewer, drawUrl } from "./viewer2d.js?v=2.1.12";
import { VolumeRenderer } from "./viewer3d.js?v=2.1.9";

const $ = (id) => document.getElementById(id);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const S = {
  summary: null,
  win: { lo: null, hi: null },
  slice: { index: 0 },
  viewer: null,
  roiViewer: null,
  vol: null,
  plan: [],
  lastEstimate: null,
  bhApplied: false,
  loaded: false,
};

// ------------------------------------------------------------- utilities
function setStatus(text, cls = "") {
  const el = $("status-text");
  el.textContent = text;
  el.className = cls;
}
function msg(id, text, cls = "") {
  const el = $(id);
  if (el) { el.textContent = text; el.className = `msg ${cls}`.trim(); }
}
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(+v) ? "—" : (+v).toFixed(d));
const signed = (v, d = 2) => (v === null || v === undefined ? "—" : (v >= 0 ? "+" : "") + (+v).toFixed(d));

function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/** Render a nested dict into a table, skipping long arrays. */
function renderKv(el, obj, prefix = "") {
  if (!el) return;
  const rows = [];
  const walk = (o, p) => {
    for (const [k, v] of Object.entries(o || {})) {
      const key = p ? `${p}.${k}` : k;
      if (v && typeof v === "object" && !Array.isArray(v)) walk(v, key);
      else if (Array.isArray(v) && v.length > 12) continue;
      else if (v && typeof v === "object") rows.push([key, JSON.stringify(v)]);
      else rows.push([key, String(v)]);
    }
  };
  walk(obj, prefix);
  el.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") ||
    "<tr><td>—</td><td>—</td></tr>";
}

function drawBars(canvas, labels, values, opts = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!values || !values.length) return;
  const max = Math.max(...values, 1);
  const pad = 6;
  const bw = (w - pad * 2) / values.length;
  ctx.fillStyle = opts.color || "#e8a24a";
  values.forEach((v, i) => {
    const bh = (h - pad * 2) * (v / max);
    ctx.fillRect(pad + i * bw, h - pad - bh, Math.max(bw - 1, 1), bh);
  });
  if (labels) {
    ctx.fillStyle = "#8a7f6f";
    ctx.font = "10px monospace";
    ctx.fillText(String(labels[0]), 2, h - 1);
    const mid = Math.floor(labels.length / 2);
    ctx.fillText(String(labels[mid]), w / 2 - 12, h - 1);
    ctx.fillText(String(labels[labels.length - 1]), w - 34, h - 1);
  }
  if (opts.title) {
    ctx.fillStyle = "#9c9182";
    ctx.font = "11px sans-serif";
    ctx.fillText(opts.title, 8, 13);
  }
}

function drawProfile(canvas, xs, ys, opts = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!xs || !xs.length) return;
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = 30;
  ctx.strokeStyle = "#352e23";
  ctx.strokeRect(pad, 10, w - pad - 10, h - 30);
  ctx.strokeStyle = opts.color || "#62c29a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ys.forEach((y, i) => {
    const px = pad + (w - pad - 10) * (i / Math.max(ys.length - 1, 1));
    const py = 10 + (h - 30) * (1 - (y - lo) / Math.max(hi - lo, 1e-9));
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = "#9c9182";
  ctx.font = "10px monospace";
  ctx.fillText(fmt(hi, 0), 2, 16);
  ctx.fillText(fmt(lo, 0), 2, h - 22);
  if (opts.title) { ctx.font = "11px sans-serif"; ctx.fillText(opts.title, pad + 4, 22); }
  if (opts.unit) { ctx.fillText(opts.unit, w - 60, h - 8); }
}

// ----------------------------------------------------------- summary/header
async function refreshSummary() {
  S.summary = await get("/api/summary");
  renderHeader();
  syncMeasureScale();
  return S.summary;
}

/** Feed the voxel size to the slice viewer so measurements report mm. */
function syncMeasureScale() {
  const v = S.summary?.voxel_mm;
  if (S.viewer) S.viewer.spacing = v ? [v.y, v.x] : null;
}

function renderHeader() {
  const s = S.summary;
  if (!s || !s.loaded) {
    $("dataset-info").textContent = "no dataset loaded";
    return;
  }
  const sp = s.spacing;
  $("dataset-info").textContent =
    `${s.name} · ${s.shape.z}×${s.shape.y}×${s.shape.x} · ` +
    `voxel ${sp.z}/${sp.y}/${sp.x} mm · ${s.stats.min.toFixed(0)}…${s.stats.max.toFixed(0)} ${s.unit}`;
  $("unit-badge").textContent = s.unit;
  $("history-badge").textContent = s.history && s.history.length
    ? `history: ${s.history[s.history.length - 1]} (${s.history.length})`
    : "history: clean";
}

// ------------------------------------------------------------------- tabs
function wireTabs() {
  $("tabs").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-tab]");
    if (!btn) return;
    $$("#tabs button").forEach((b) => b.classList.toggle("active", b === btn));
    $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${btn.dataset.tab}`));
    onTab(btn.dataset.tab);
  });
}

async function onTab(name) {
  try {
    if (name === "viewer") await initViewerTab();
    else if (name === "volume") await initVolumeTab();
    else if (name === "bh") await initBhTab();
    else if (name === "roi") await initRoiTab();
    else if (name === "align") await initAlignTab();
    else if (name === "pipeline") await initPipelineTab();
    else if (name === "analysis") await initAnalysisTab();
    else if (name === "import") await renderDetections();
    else if (name === "cracks") await initCracksTab();
    else if (name === "help") initHelpTab();
  } catch (e) {
    setStatus(String(e.message || e), "error");
  }
}

// ----------------------------------------------------------------- cracks
async function initCracksTab() {
  if (!S.cracksWired) {
    S.cracksWired = true;
    $("btn-cr-detect").onclick = runCrackDetection;
    $("btn-cr-clear").onclick = clearCracks;
    $("btn-cr-demo").onclick = async () => {
      msg("cr-status", "loading fractured demo sample …");
      try {
        const r = await post("/api/sample", { fractures: true });
        S.summary = r;
        S.slice.index = Math.floor((r.shape?.z || 96) / 2);
        $("cr-slice-index").value = S.slice.index;
        $("slice-slider").max = (r.shape?.z || 96) - 1;
        refreshSummary();
        runCrackDetection();
      } catch (e) { msg("cr-status", String(e.message || e), "error"); }
    };
    $("cr-slice-index").onchange = refreshCrackOverlay;
    $("cr-color").oninput = refreshCrackOverlay;  // live re-tint
  }
  if (S.crackResult) { renderCrackResults(S.crackResult); return; }
  if (!S.cracksRestored && S.summary?.loaded) {
    S.cracksRestored = true;
    get("/api/cracks").then((r) => {
      if (r.result) {
        S.crackResult = r.result;
        if (r.color) $("cr-color").value = r.color;
        renderCrackResults(r.result, r.overlay);
      }
    }).catch(() => {});  // no stored result
  }
}

async function runCrackDetection() {
  if (!S.summary?.loaded) { msg("cr-status", "load a volume first", "error"); return; }
  msg("cr-status", "detecting …");
  try {
    const mode = $("cr-threshold-mode").value;
    const body = {
      threshold: mode === "auto" ? "auto" : parseFloat($("cr-threshold-value").value),
      min_voxels: parseInt($("cr-min-voxels").value, 10) || 25,
      min_elongation: parseFloat($("cr-min-elongation").value) || 0.55,
      max_aperture_mm: parseFloat($("cr-max-aperture").value) || 0,
      index: parseInt($("cr-slice-index").value, 10) || 0,
      color: $("cr-color").value,
    };
    const r = await post("/api/cracks/detect", body);
    S.crackResult = r.result;
    if (r.color) $("cr-color").value = r.color;
    renderCrackResults(r.result, r.overlay);
  } catch (e) { msg("cr-status", String(e.message || e), "error"); }
}

function renderCrackResults(res, overlay) {
  if (!res) return;
  const rows = [
    ["cracks found", res.count],
    ["total crack volume", `${fmt(res.volume_mm3, 1)} mm³`],
    ["crack fraction", `${fmt(res.fraction_pct, 3)} % of specimen`],
    ["mean aperture", `${fmt(res.mean_aperture_mm, 3)} mm`],
    ["max aperture", `${fmt(res.max_aperture_mm, 3)} mm`],
    ["dominant orientation", res.dominant_orientation_deg == null ? "—" : `${fmt(res.dominant_orientation_deg, 1)}°`],
    ["threshold", `${fmt(res.threshold, 0)} (${res.threshold_source})`],
    ["specimen voxels", res.specimen_voxels],
  ];
  $("cr-summary-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  $("cr-components-table").querySelector("tbody").innerHTML = (res.components || []).map((c) =>
    `<tr><td>${c.id}</td><td>${c.voxels}</td><td>${fmt(c.volume_mm3, 2)}</td>` +
    `<td>${fmt(c.aperture_mm, 3)}</td><td>${fmt(c.orientation_deg, 1)}°</td>` +
    `<td class="mono small">${c.center.map((v) => fmt(v, 0)).join(", ")}</td></tr>`).join("")
    || "<tr><td colspan=6>no cracks detected</td></tr>";
  const oh = res.orientation_histogram || {};
  if (oh.weights?.length) {
    const labels = oh.weights.map((_, i) => String(Math.round(i * (oh.bin_deg || 15))));
    drawBars($("cr-orient-canvas"), labels, oh.weights, { color: "#e8a24a", title: "orientation (deg)" });
  }
  const ps = res.per_slice_mm2 || [];
  if (ps.length) drawProfile($("cr-slice-canvas"), ps.map((_, i) => i), ps, { title: "crack area per slice (mm²)" });
  if (overlay) $("cr-overlay-img").src = overlay;
  msg("cr-status", res.count ? `${res.count} crack(s), ${fmt(res.volume_mm3, 1)} mm³ total` : "no cracks found");
  setStatus(`cracks: ${res.count}`);
}

async function refreshCrackOverlay() {
  if (!S.crackResult) return;
  const i = parseInt($("cr-slice-index").value, 10) || 0;
  const c = encodeURIComponent($("cr-color").value || "");
  $("cr-overlay-img").src = `/api/slice.png?overlay=cracks&index=${i}&color=${c}`;
}

async function clearCracks() {
  try {
    await del("/api/cracks");
    S.crackResult = null;
    $("cr-summary-table").innerHTML = "";
    $("cr-components-table").querySelector("tbody").innerHTML = "";
    $("cr-overlay-img").removeAttribute("src");
    msg("cr-status", "cleared");
    setStatus("ready");
  } catch (e) { msg("cr-status", String(e.message || e), "error"); }
}

// -------------------------------------------------------------------- pores
function wirePoreSection() {
  if (S.poresWired) return;
  S.poresWired = true;
  $("btn-po-analyse").onclick = runPoreAnalysis;
  $("btn-po-clear").onclick = clearPores;
  $("po-slice-index").onchange = refreshPoreOverlay;
  $("po-color").oninput = refreshPoreOverlay;  // live re-tint
}

async function restorePores() {
  if (S.poreResult || S.poresRestored || !S.summary?.loaded) return;
  S.poresRestored = true;
  try {
    const r = await get("/api/pores");
    if (r.result) {
      S.poreResult = r.result;
      if (r.color) $("po-color").value = r.color;
      renderPoreResults(r.result, r.overlay);
    }
  } catch (e) { /* no stored result */ }
}

async function runPoreAnalysis() {
  if (!S.summary?.loaded) { msg("po-status", "load a volume first", "error"); return; }
  msg("po-status", "analysing pores …");
  try {
    const mode = $("po-threshold-mode").value;
    const body = {
      threshold: mode === "auto" ? "auto" : parseFloat($("po-threshold-value").value),
      min_voxels: parseInt($("po-min-voxels").value, 10) || 8,
      max_elongation: parseFloat($("po-max-elongation").value) || 0.7,
      index: parseInt($("po-slice-index").value, 10) || 0,
      color: $("po-color").value,
    };
    const r = await post("/api/pores", body);
    S.poreResult = r.result;
    if (r.color) $("po-color").value = r.color;
    renderPoreResults(r.result, r.overlay);
  } catch (e) { msg("po-status", String(e.message || e), "error"); }
}

function renderPoreResults(res, overlay, opts = {}) {
  if (!res) return;
  const rows = [
    ["pores resolved", res.count],
    ["resolved pore volume", `${fmt(res.volume_mm3, 2)} mm³ (${fmt(res.volume_cm3, 4)} cm³)`],
    ["resolved porosity", `${fmt(res.porosity_pct, 3)} % of specimen`],
    ["mean Ø (eq. sphere)", `${fmt(res.mean_diameter_mm, 3)} mm`],
    ["median Ø", `${fmt(res.median_diameter_mm, 3)} mm`],
    ["D10 / D50 / D90", `${fmt(res.d10_mm, 3)} / ${fmt(res.d50_mm, 3)} / ${fmt(res.d90_mm, 3)} mm`],
    ["mean local thickness", `${fmt(res.mean_local_thickness_mm, 3)} mm`],
    ["mean sphericity", fmt(res.mean_sphericity, 3)],
    ["threshold", `${fmt(res.threshold, 0)} (${res.threshold_source})`],
  ];
  $("po-summary-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  const d = res.distribution || {};
  if (d.counts?.length) {
    const edges = d.bin_edges_mm || [];
    const labels = edges.length
      ? edges.map((e) => fmt(e, e < 10 ? 1 : 0))
      : d.counts.map((_, i) => String(i));
    drawBars($("po-hist-canvas"), labels, d.counts, { color: "#40d2be",
      title: opts.fromReport ? "pore sizes (auto params)" : "pore sizes (mm)" });
    $("po-hist-note").textContent = `${d.log_spaced ? "log" : "linear"}-spaced bins · ` +
      `bars = pore count; volume per bin available in the JSON export`;
  } else {
    drawBars($("po-hist-canvas"), null, [], {});
    $("po-hist-note").textContent = "no pores above the min-voxel size";
  }

  $("po-components-table").querySelector("tbody").innerHTML = (res.components || []).map((c) =>
    `<tr><td>${c.id}</td><td>${fmt(c.volume_mm3, 3)}</td><td>${fmt(c.diameter_mm, 3)}</td>` +
    `<td>${fmt(2 * (c.radius_mm ?? 0), 3)}</td><td>${fmt(c.sphericity, 3)}</td>` +
    `<td class="mono small">${c.center.map((v) => fmt(v, 0)).join(", ")}</td></tr>`).join("")
    || "<tr><td colspan=6>no pores detected</td></tr>";

  if (overlay) $("po-overlay-img").src = overlay;
  const note = opts.fromReport ? " (from Run analysis, default params)" : "";
  msg("po-status", res.count
    ? `${res.count} pore(s), D50 ${fmt(res.d50_mm, 2)} mm, ${fmt(res.porosity_pct, 2)} % porosity${note}`
    : `no pores found${note}`);
  setStatus(`pores: ${res.count}`);
}

async function refreshPoreOverlay() {
  if (!S.poreResult) return;
  const i = parseInt($("po-slice-index").value, 10) || 0;
  const c = encodeURIComponent($("po-color").value || "");
  $("po-overlay-img").src = `/api/slice.png?overlay=pores&index=${i}&color=${c}`;
}

async function clearPores() {
  try {
    await del("/api/pores");
    S.poreResult = null;
    $("po-summary-table").innerHTML = "";
    $("po-components-table").querySelector("tbody").innerHTML = "";
    drawBars($("po-hist-canvas"), null, [], {});
    $("po-hist-note").textContent = "";
    $("po-overlay-img").removeAttribute("src");
    msg("po-status", "cleared");
  } catch (e) { msg("po-status", String(e.message || e), "error"); }
}

// ------------------------------------------------------------------- help
function initHelpTab() {
  const frame = document.getElementById("help-frame");
  if (frame && !frame.src) frame.src = frame.dataset.src;  // lazy: load manual on first open
}

// ----------------------------------------------------------------- import
async function renderDetections() {
  if (!S.summary || !S.summary.loaded) return;
  let det = {}, meta = {};
  try {
    const d = await get("/api/detections");
    det = d.detections || {};
    meta = d.metadata || {};
  } catch { /* keep empty */ }
  const val = (o) => (o && "value" in o ? o.value : null);
  const src = (o) => (o && o.source ? o.source : "");
  const sst = det.slice_spacing, px = det.pixel_size, dims = det.image_dimensions, sm = det.scanner_metadata;
  $("d-slice").textContent = val(sst) !== null ? `${val(sst)} mm  ·  ${src(sst)}` : "—";
  $("d-pixel").textContent = val(px) ? `${val(px).join(" × ")} mm  ·  ${src(px)}` : "—";
  $("d-dims").textContent = val(dims) ? `${val(dims).join(" × ")} px (x,y,z)  ·  ${src(dims)}` : "—";
  $("d-scanner").textContent = val(sm) || "—";
  $("d-source").textContent = meta.spacing_source
    ? Object.entries(meta.spacing_source).map(([k, v]) => `${k}: ${v}`).join("  |  ")
    : [src(sst), src(px)].filter(Boolean).join("  |  ");

  const acq = meta.acquisition || {};
  const rows = Object.keys(acq).length ? Object.entries(acq) : [
    ["manufacturer", meta.manufacturer || "—"],
    ["model", meta.manufacturer_model || "—"],
    ["kVp", meta.kvp ?? "—"],
    ["kernel", meta.convolution_kernel || "—"],
  ];
  $("acq-body").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

async function loadPath() {
  const path = $("import-path").value.trim();
  if (!path) { msg("import-msg", "enter a path first", "error"); return; }
  const body = { path, kind: $("import-kind").value };
  const dz = parseFloat($("import-dz").value);
  const dxy = parseFloat($("import-dxy").value);
  if (dz > 0 || dxy > 0) body.spacing = [dz || 1, dxy || 1, dxy || 1];
  setStatus(`importing ${path} …`);
  try {
    const r = await post("/api/import", body);
    await refreshSummary();
    msg("import-msg", `imported ${r.summary.shape.z} slices from ${path}`);
    await renderDetections();
    resetViewers();
    setStatus("import complete");
  } catch (e) {
    msg("import-msg", String(e.message || e), "error");
  }
}

async function renderBrowser(path) {
  const b = await get(`/api/browse?path=${encodeURIComponent(path || "data")}`);
  $("browser-path").textContent = b.path;
  const list = $("browser-list");
  list.innerHTML = "";
  const mk = (label, target, isDir) => {
    const el = document.createElement("button");
    el.textContent = label;
    el.className = isDir ? "dir" : "";
    el.onclick = () => {
      if (isDir) renderBrowser(target);
      else { $("import-path").value = target; }
    };
    return el;
  };
  if (b.parent) list.appendChild(mk("↰ " + b.parent, b.parent, true));
  b.dirs.forEach((d) => list.appendChild(mk("📁 " + d, `${b.path}/${d}`.replace(/\\/g, "/"), true)));
  b.files.forEach((f) => list.appendChild(mk("📄 " + f, `${b.path}/${f}`.replace(/\\/g, "/"), false)));
}

function wireImport() {
  $("btn-import").onclick = loadPath;
  $("btn-browse").onclick = () => renderBrowser($("import-path").value || "data");
  $("btn-sample").onclick = async () => {
    setStatus("loading synthetic sample …");
    try {
      await post("/api/sample", {});
      await refreshSummary();
      await renderDetections();
      msg("import-msg", "synthetic rock-core sample loaded");
      resetViewers();
    } catch (e) { msg("import-msg", String(e.message || e), "error"); }
  };
  $("file-input").onchange = async (ev) => {
    if (!ev.target.files.length) return;
    setStatus(`uploading ${ev.target.files.length} files …`);
    try {
      await upload("/api/import/upload", Array.from(ev.target.files));
      await refreshSummary();
      await renderDetections();
      msg("import-msg", `uploaded ${ev.target.files.length} files`);
      resetViewers();
    } catch (e) { msg("import-msg", String(e.message || e), "error"); }
    ev.target.value = "";
  };
  renderBrowser("data");
}

// ------------------------------------------------------------ slice viewer
let viewerInited = false;
async function initViewerTab() {
  if (!viewerInited) {
    viewerInited = true;
    const refreshDebounced = debounce(() => S.viewer.refresh(), 70);
    // Readouts render from viewer state, so they stay correct across tool switches.
    const renderMeasureOut = () => {
      const v = S.viewer.measureValue();
      $("measure-out").textContent = v != null
        ? (v.mm != null ? `${v.mm.toFixed(2)} mm (${v.px.toFixed(1)} px)` : `${v.px.toFixed(1)} px`)
        : (S.viewer.tool === "measure" ? "click two points, or drag" : "—");
    };
    const renderThicknessOut = () => {
      const v = S.viewer.thicknessValue();
      $("thickness-out").textContent = v != null
        ? (v.mm
            ? `${v.n} caliper${v.n > 1 ? "s" : ""} · mean ${v.mm.mean.toFixed(2)} mm · ` +
              `σ ${v.mm.std.toFixed(2)} · ${v.mm.min.toFixed(2)}–${v.mm.max.toFixed(2)}`
            : `${v.n} caliper${v.n > 1 ? "s" : ""} · mean ${v.px.mean.toFixed(1)} px · ` +
              `σ ${v.px.std.toFixed(1)} · ${v.px.min.toFixed(1)}–${v.px.max.toFixed(1)}`)
        : (S.viewer.tool === "thickness" ? "drag a caliper across the crack/layer" : "—");
      $("thickness-row").hidden = v == null && S.viewer.tool !== "thickness";
    };
    S.viewer = new SliceViewer($("slice-canvas"), {
      onWindowChange: (lo, hi) => {
        S.win.lo = lo; S.win.hi = hi;
        $("wl-width").value = Math.round(hi - lo);
        $("wl-level").value = Math.round((hi + lo) / 2);
        refreshDebounced();
      },
      onWheel: (d) => setSlice(S.slice.index + d),
      onDoubleClick: () => autoWindow(),
      onMeasureChange: renderMeasureOut,
      onThicknessChange: renderThicknessOut,
    });
    syncMeasureScale();
    // The two tools are mutually exclusive pointer modes, but their annotations
    // persist across switches: only Clear — or clicking a tool's own button
    // again (toggling it off) — removes them (manual, "Measure" section).
    const setSliceTool = (which) => {
      const on = S.viewer.tool !== which;
      S.viewer.setTool(on ? which : "window");
      const btns = { measure: ["slice-measure", "📏", "Measure"],
                     thickness: ["slice-thickness", "🧱", "Thickness"] };
      for (const key of Object.keys(btns)) {
        const [id, icon, name] = btns[key];
        const active = on && key === which;
        $(id).classList.toggle("primary", active);
        $(id).textContent = active ? `${icon} ${name}…` : `${icon} ${name}`;
      }
      if (!on) {
        if (which === "measure") S.viewer.clearMeasure(); else S.viewer.clearThickness();
      }
      renderMeasureOut();
      renderThicknessOut();
    };
    $("slice-measure").onclick = () => setSliceTool("measure");
    $("slice-thickness").onclick = () => setSliceTool("thickness");
    $("measure-clear").onclick = () => {
      S.viewer.clearMeasure();
      S.viewer.clearThickness();
    };
    $("slice-slider").oninput = (e) => setSlice(+e.target.value);
    $("wl-width").onchange = () => applyWindowInputs();
    $("wl-level").onchange = () => applyWindowInputs();
    $("wl-auto").onclick = autoWindow;
    $("slice-overlay").onchange = () => {
      const mode = $("slice-overlay").value;
      // until the user picks a color, show the anchor of the active default rendering
      if (!S.viewerColorTouched) $("slice-color").value = mode === "segmentation" ? "#50c8ff" : "#ff3c3c";
      S.viewer.configure({ overlay: mode });
    };
    $("slice-color").oninput = () => {
      S.viewerColorTouched = true;
      S.viewer.configure({ color: $("slice-color").value });
    };
    $("slice-invert").onchange = () => S.viewer.configure({ invert: $("slice-invert").checked });
    document.querySelectorAll(".preset").forEach((b) => {
      b.onclick = () => {
        const w = +b.dataset.w, l = +b.dataset.l;
        setWindow(l - w / 2, l + w / 2);
      };
    });
  }
  const s = S.summary;
  if (s && s.loaded) {
    $("slice-slider").max = s.shape.z - 1;
    setSlice(Math.min(S.slice.index, s.shape.z - 1), true);
  }
  await loadHistogram();
}

function setSlice(i, force = false) {
  const z = S.summary?.shape?.z || 1;
  const idx = Math.min(Math.max(0, i), z - 1);
  if (!force && idx === S.slice.index) return;
  S.slice.index = idx;
  $("slice-slider").value = idx;
  $("slice-label").textContent = `#${idx} / ${z - 1}`;
  S.viewer?.configure({ index: idx });
  $("roi-index").value = idx;
  $("bh-index").value = idx;
  updateSliceInfo();
}

function setWindow(lo, hi) {
  S.win.lo = lo; S.win.hi = hi;
  $("wl-width").value = Math.round(hi - lo);
  $("wl-level").value = Math.round((hi + lo) / 2);
  S.viewer?.configure({ lo, hi });
}

function applyWindowInputs() {
  const w = +$("wl-width").value, l = +$("wl-level").value;
  setWindow(l - w / 2, l + w / 2);
}

async function autoWindow() {
  try {
    const info = await get(`/api/slice/info?index=${S.slice.index}`);
    setWindow(info.window.lo, info.window.hi);
    renderSliceInfo(info);
  } catch (e) { setStatus(String(e.message || e), "error"); }
}

async function updateSliceInfo() {
  try {
    const info = await get(`/api/slice/info?index=${S.slice.index}`);
    renderSliceInfo(info);
  } catch { /* ignore */ }
}

function renderSliceInfo(info) {
  if (info.window && (S.win.lo === null || S.win.hi === null)) {
    S.win.lo = info.window.lo;
    S.win.hi = info.window.hi;
    $("wl-width").value = Math.round(info.window.hi - info.window.lo);
    $("wl-level").value = Math.round((info.window.hi + info.window.lo) / 2);
    S.viewer?.configure({ lo: S.win.lo, hi: S.win.hi });
    S.roiViewer?.configure({ lo: S.win.lo, hi: S.win.hi });
    $("vol-width").value = Math.round(info.window.hi - info.window.lo);
    $("vol-level").value = Math.round((info.window.hi + info.window.lo) / 2);
  }
  $("slice-info").innerHTML = [
    ["index", info.index], ["size", `${info.shape[0]} × ${info.shape[1]}`],
    ["min", fmt(info.min, 1)], ["max", fmt(info.max, 1)],
    ["mean", fmt(info.mean, 1)], ["std", fmt(info.std, 1)],
    ["unit", info.unit], ["ROI mask", info.has_mask ? "active" : "—"],
    ["segmentation", info.has_segmentation ? "active" : "—"],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

async function loadHistogram() {
  try {
    const h = await get("/api/histogram?bins=96&stride=3");
    drawBars($("hist-canvas"), [h.lo, (h.lo + h.hi) / 2, h.hi], h.counts,
      { color: "#e8a24a", title: `histogram (stride sample n=${h.sampled})` });
  } catch { /* ignore */ }
}

// ------------------------------------------------------------ 3D viewer
async function initVolumeTab() {
  if (!S.vol) {
    S.vol = new VolumeRenderer($("vol-canvas"));
    $("vol-auto").onchange = (e) => { S.vol.autoRotate = e.target.checked; if (e.target.checked) S.vol.start(); };
    $("vol-reset").onclick = () => S.vol.reset();
    $("vol-reload").onclick = () => loadBrick();
    $("vol-width").onchange = () => volWindow();
    $("vol-level").onchange = () => volWindow();
    $("vol-maxdim").onchange = () => loadBrick();
    $("vol-smooth").onchange = () => loadBrick();
    $("vol-source").onchange = () => loadBrick();
    if (S.summary?.loaded && !S.win.lo && S.summary) {
      $("vol-width").value = Math.round(S.summary.stats.max - S.summary.stats.min);
      $("vol-level").value = Math.round((S.summary.stats.max + S.summary.stats.min) / 2);
    }
    await loadBrick();
    S.vol.start();
  }
}

function volWindow() {
  const w = +$("vol-width").value, l = +$("vol-level").value;
  S.vol.setWindow(l - w / 2, l + w / 2);
}

async function loadBrick() {
  const maxDim = $("vol-maxdim").value;
  const smooth = $("vol-smooth").value || 0;
  const source = $("vol-source").value;
  setStatus("loading volume brick …");
  try {
    const info = await S.vol.load(`/api/volume/brick?max_dim=${maxDim}&smoothing=${smooth}&source=${source}`);
    const [z, y, x] = info.shape;
    $("vol-info").innerHTML = [
      ["brick", `${z} × ${y} × ${x}`], ["source", `${info.shape} @ factor ${info.factor}`],
      ["spacing", info.spacing.map((s) => s.toFixed(3)).join(" × ") + " mm"],
      ["value range", `${fmt(info.lo, 0)} … ${fmt(info.hi, 0)}`],
      ["bytes", `${(z * y * x / 1048576).toFixed(1)} MB (uint8)`],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    if (!$("vol-width").value || !S.volWinSet) {
      if (S.win.lo === null) {
        try {
          const info = await get(`/api/slice/info?index=${S.slice.index}`);
          S.win.lo = info.window.lo;
          S.win.hi = info.window.hi;
        } catch { /* fall back to brick range */ }
      }
      if (S.win.lo !== null) {
        $("vol-width").value = Math.round(S.win.hi - S.win.lo);
        $("vol-level").value = Math.round((S.win.hi + S.win.lo) / 2);
      } else {
        $("vol-width").value = Math.round(info.hi - info.lo);
        $("vol-level").value = Math.round((info.hi + info.lo) / 2);
      }
      S.volWinSet = true;
    }
    volWindow();
    setStatus("3D volume ready");
  } catch (e) {
    setStatus(String(e.message || e), "error");
  }
}

// ------------------------------------------------------ beam hardening tab
function bhParams() {
  const method = $("bh-method").value;
  const params = {};
  if (method === "polynomial") params.degree = +$("bh-degree").value;
  if (method === "spectral") {
    params.kvp = +$("bh-kvp").value;
    params.inherent_al_mm = +$("bh-inherent").value;
    params.added_filter = $("bh-filter").value === "none" ? "" : $("bh-filter").value;
    params.added_mm = $("bh-filter").value === "none" ? 0 : +$("bh-filtermm").value;
    params.material = $("bh-material").value;
  }
  return { method, strength: +$("bh-strength").value, params };
}

function updateBhFields() {
  const m = $("bh-method").value;
  $("bh-polynomial-row").hidden = m !== "polynomial";
  document.querySelectorAll(".spectral-params").forEach((el) => { el.style.display = m === "spectral" ? "" : "none"; });
}

function step(n) {
  document.querySelectorAll(".stepper span").forEach((s) => {
    if (+s.dataset.step <= n) s.classList.add("on");
  });
}

async function initBhTab() {
  $("bh-index").value = S.slice.index;
  $("bh-method").onchange = updateBhFields;
  updateBhFields();
  $("bh-strength").oninput = (e) => { $("bh-strength-val").textContent = (+e.target.value).toFixed(2); };
  $("btn-bh-detect").onclick = bhDetect;
  $("btn-bh-preview").onclick = bhPreview;
  $("btn-bh-apply").onclick = bhApply;
  wireWipe();
  if (!$("bh-profile").dataset.done) await bhDetect();
  try { await initRoiTab(); } catch { /* no volume yet */ }
  // populate the before/after images on first open (they render as broken icons without a src)
  if (S.summary?.loaded) {
    if (!$("bh-img-before").getAttribute("src")) await bhPreview();
    if (!$("roi-img-before").getAttribute("src")) await roiPreview();
  }
}

async function bhDetect() {
  setStatus("running beam-hardening detection …");
  try {
    const r = await post("/api/bh/detect", { index: +$("bh-index").value });
    const d = r.detect;
    $("bh-sev").textContent = signed(d.cupping_severity, 4);
    $("bh-verdict").textContent = d.verdict || "—";
    $("bh-core").textContent = fmt(d.core_mean, 0);
    $("bh-shell").textContent = fmt(d.shell_mean, 0);
    $("bh-streak").textContent = `streaks ${(d.dark_streak_fraction ?? 0).toFixed(3)}`;
    const p = r.profile || {};
    if (p.radius && p.mean) {
      drawProfile($("bh-profile"), p.radius, p.mean,
        { title: "radial cupping profile", unit: r.unit });
      $("bh-profile").dataset.done = "1";
    }
    step(2);
    setStatus("detection complete");
  } catch (e) { setStatus(String(e.message || e), "error"); }
}

async function bhPreview() {
  const b = bhParams();
  msg("bh-status", "computing preview …");
  try {
    const r = await post("/api/bh/preview", { ...b, index: +$("bh-index").value });
    $("bh-img-before").src = r.before;
    $("bh-img-after").src = r.after;
    $("bh-sev-before").textContent = signed(r.detect_before.cupping_severity, 4);
    $("bh-sev-after").textContent = signed(r.detect_after.cupping_severity, 4);
    $("bh-wipe").style.setProperty("--split", "50%");
    msg("bh-status", `preview ready · core ${fmt(r.detect_before.core_mean, 0)} → ${fmt(r.detect_after.core_mean, 0)} HU`);
    step(3);
  } catch (e) { msg("bh-status", String(e.message || e), "error"); }
}

async function bhApply() {
  const b = bhParams();
  msg("bh-status", "applying correction to the whole volume …");
  setStatus("beam-hardening correction running …");
  try {
    const r = await post("/api/bh/apply", b);
    $("bh-sev-before").textContent = signed(r.detect_before.cupping_severity, 4);
    $("bh-sev-after").textContent = signed(r.detect_after.cupping_severity, 4);
    $("bh-sev").textContent = signed(r.detect_after.cupping_severity, 4);
    $("bh-verdict").textContent = r.detect_after.verdict || "—";
    $("bh-core").textContent = fmt(r.detect_after.core_mean, 0);
    $("bh-shell").textContent = fmt(r.detect_after.shell_mean, 0);
    await refreshSummary();
    msg("bh-status", `applied (${r.info.method || b.method}) · severity ${signed(r.detect_before.cupping_severity, 4)} → ${signed(r.detect_after.cupping_severity, 4)}`);
    step(3);
    setStatus("correction applied - use Undo to revert");
    S.viewer?.refresh();
    loadHistogram();
  } catch (e) { msg("bh-status", String(e.message || e), "error"); }
}

function wireWipe() {
  const wipe = $("bh-wipe");
  const setFromEvent = (ev) => {
    const r = wipe.getBoundingClientRect();
    const pct = Math.min(Math.max(((ev.clientX - r.left) / r.width) * 100, 0), 100);
    wipe.style.setProperty("--split", `${pct}%`);
    $("bh-wipe-handle").style.left = `${pct}%`;
  };
  let dragging = false;
  wipe.addEventListener("pointerdown", (ev) => { dragging = true; wipe.setPointerCapture(ev.pointerId); setFromEvent(ev); });
  wipe.addEventListener("pointermove", (ev) => { if (dragging) setFromEvent(ev); });
  wipe.addEventListener("pointerup", () => { dragging = false; });
}

// --------------------------------------------------------------- ROI tab
let roiInited = false;
async function initRoiTab() {
  if (!roiInited) {
    roiInited = true;
    S.roiViewer = new SliceViewer($("roi-canvas"), {
      onWheel: (d) => { setSlice(S.slice.index + d); syncRoiIndex(); },
      onShapeChange: () => {},
      onDrawFinish: () => {},
    });
    S.roiViewer.state.lo = S.win.lo; S.roiViewer.state.hi = S.win.hi;
    $("roi-type").onchange = () => {
      const t = $("roi-type").value;
      $("roi-artifact-row").hidden = t !== "artifact";
      S.roiViewer.setTool(t === "rect" || t === "ellipse" || t === "polygon" ? t : "window");
      if (t === "polygon") S.roiViewer.setShape(null);
      if (t !== "polygon" && !S.roiViewer.shape) defaultRoiShape();
    };
    $("roi-index").onchange = () => { setSlice(+$("roi-index").value); syncRoiIndex(); };
    $("btn-roi-preview").onclick = roiPreview;
    $("btn-roi-area").onclick = roiArea;
    $("btn-roi-apply").onclick = roiApply;
    $("btn-roi-clear").onclick = roiClear;
    $("btn-roi-clean").onclick = roiClean;
    $("btn-roi-air").onclick = roiRemoveAir;
  }
  $("roi-index").value = S.slice.index;
  const t = $("roi-type").value;
  S.roiViewer.setTool(t === "rect" || t === "ellipse" || t === "polygon" ? t : "window");
  // keep an existing (possibly moved) shape across tab switches; seed a default only when none
  if (!S.roiViewer.shape && t !== "polygon") defaultRoiShape();
  await S.roiViewer.configure({ index: S.slice.index, lo: S.win.lo, hi: S.win.hi });
}

function syncRoiIndex() { $("roi-index").value = S.slice.index; S.roiViewer?.configure({ index: S.slice.index }); }

function defaultRoiShape() {
  const w = S.summary?.shape?.x || 256, h = S.summary?.shape?.y || 256;
  S.roiViewer?.setShape({ type: "ellipse", cx: w / 2, cy: h / 2, rx: w * 0.42, ry: h * 0.42 });
}

function roiSpec() {
  const t = $("roi-type").value;
  if (t === "auto" || t === "remove_air") return { type: t };
  if (t === "artifact") return { type: "artifact", mode: $("roi-artifact-mode").value };
  const sh = S.roiViewer.shape;
  if (!sh) throw new Error("draw a ROI on the slice first");
  if (sh.type === "polygon") {
    if (sh.points.length < 3) throw new Error("freehand ROI needs at least 3 points");
    return { type: "polygon", points: sh.points.map(([x, y]) => [Math.round(x), Math.round(y)]) };
  }
  if (sh.type === "rect") return { type: "rect", x: sh.x, y: sh.y, w: sh.w, h: sh.h };
  return { type: "ellipse", cx: sh.cx, cy: sh.cy, rx: sh.rx, ry: sh.ry };
}

async function roiArea() {
  try {
    const spec = roiSpec();
    const r = await post("/api/roi/area", { spec, index: +$("roi-index").value });
    const px2 = r.area_px.toLocaleString();
    $("roi-area-out").textContent = r.spacing_mm
      ? `${fmt(r.area_mm2, 1)} mm² · ${px2} px · ${(r.area_fraction * 100).toFixed(1)} % of slice`
      : `${px2} px · ${(r.area_fraction * 100).toFixed(1)} % of slice (no spacing metadata)`;
    msg("roi-status", `ROI area on slice ${r.index}: ${fmt(r.area_mm2, 1)} mm²`);
  } catch (e) { msg("roi-status", String(e.message || e), "error"); }
}

async function roiPreview() {
  try {
    const spec = roiSpec();
    msg("roi-status", "previewing …");
    const r = await post("/api/roi/preview", { spec, index: +$("roi-index").value });
    $("roi-img-before").src = r.before;
    $("roi-img-after").src = r.after;
    $("roi-info").innerHTML = [
      ["area fraction", r.area_fraction.toFixed(4)],
      ["area", `${r.area_px} px`],
      ["fill value", fmt(r.fill, 1)],
      ["threshold / detail", JSON.stringify(r.info)],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    msg("roi-status", `preview: ${(r.area_fraction * 100).toFixed(1)} % kept`);
  } catch (e) { msg("roi-status", String(e.message || e), "error"); }
}

async function roiApply() {
  try {
    const spec = roiSpec();
    const r = await post("/api/roi/apply", { spec, scope: "slice", index: +$("roi-index").value });
    await refreshSummary();
    msg("roi-status", `mask applied to all slices · ${(r.info.area_fraction * 100).toFixed(1)} % kept`);
    step(4);
    S.viewer?.configure({ overlay: $("slice-overlay").value || "mask" });
  } catch (e) { msg("roi-status", String(e.message || e), "error"); }
}

async function roiClear() {
  try {
    await del("/api/roi/mask");
    await refreshSummary();
    msg("roi-status", "mask cleared");
  } catch (e) { msg("roi-status", String(e.message || e), "error"); }
}

async function roiRemoveAir() {
  try {
    msg("roi-status", "removing background air …");
    const r = await post("/api/roi/remove-air", { index: +$("roi-index").value });
    await refreshSummary();
    $("roi-img-before").src = r.before;
    $("roi-img-after").src = r.after;
    msg("roi-status",
        `air removed · slice ${r.index}: threshold ${fmt(r.threshold, 0)} · ` +
        `${(r.area_fraction * 100).toFixed(1)} % kept`);
    step(5);
    setStatus("background air removed - Undo restores the previous state");
    S.viewer?.configure({ overlay: $("slice-overlay").value || "mask" });
    S.viewer?.refresh();
    loadHistogram();
  } catch (e) { msg("roi-status", String(e.message || e), "error"); }
}

async function roiClean() {
  try {
    msg("roi-status", "cleaning volume …");
    const r = await post("/api/roi/clean", { fill: "auto" });
    await refreshSummary();
    $("roi-img-after").removeAttribute("src");
    msg("roi-status", `clean volume ready · background set to ${fmt(r.fill, 0)}`);
    step(5);
    setStatus("volume cleaned - Undo restores the previous state");
    S.viewer?.refresh();
    loadHistogram();
  } catch (e) { msg("roi-status", String(e.message || e), "error"); }
}

// ------------------------------------------------------------ alignment tab
function readout(dx, dy, rot) {
  $("align-readout").innerHTML =
    `X: <b>${signed(dx)} px</b> &nbsp;&nbsp; Y: <b>${signed(dy)} px</b> ` +
    `&nbsp;&nbsp; Rotation: <b>${signed(rot)}°</b>`;
}

function currentAlignMethod() {
  return document.querySelector("input[name=amethod]:checked")?.value || "phase";
}

async function loadAlignPanel() {
  const index = +$("align-cur").value;
  const ref = +$("align-ref").value;
  const method = currentAlignMethod();
  setStatus(`registering slice #${index} onto #${ref} (${method}) …`);
  const r = await get(`/api/align/panel?index=${index}&ref=${ref}&method=${method}`);
  await Promise.all([
    drawUrl($("align-ref-canvas"), r.ref_png),
    drawUrl($("align-cur-canvas"), r.cur_png),
    drawUrl($("align-diff-canvas"), r.diff),
    drawUrl($("align-aligned"), r.aligned_png),
    drawUrl($("align-diff-after"), r.diff_aligned),
  ]);
  $("align-ref-label").textContent = `#${r.reference}`;
  $("align-cur-label").textContent = `#${r.index}`;
  readout(r.estimate.dx, r.estimate.dy, r.estimate.rotation);
  S.lastEstimate = r.estimate;
  $("align-est-meta").textContent =
    `method=${r.estimate.method} · NCC ${r.estimate.ncc_before} → ${r.estimate.ncc_after} · ` +
    `${r.estimate.seconds}s · scale=${r.estimate.scale}`;
  $("m-dx").value = r.estimate.dx; $("m-dx-v").textContent = signed(r.estimate.dx);
  $("m-dy").value = r.estimate.dy; $("m-dy-v").textContent = signed(r.estimate.dy);
  $("m-rot").value = r.estimate.rotation; $("m-rot-v").textContent = signed(r.estimate.rotation, 2);
  setStatus("alignment estimate ready");
}

async function initAlignTab() {
  const s = S.summary;
  if (s?.loaded) {
    $("align-cur").max = s.shape.z - 1;
    $("align-ref").max = s.shape.z - 1;
    $("align-cur").value = Math.min(S.slice.index + 1, s.shape.z - 1);
    $("align-ref").value = S.slice.index;
  }
  $("btn-align-auto").onclick = () => loadAlignPanel().catch((e) => setStatus(String(e.message || e), "error"));
  document.querySelectorAll("input[name=amethod]").forEach((r) => {
    r.onchange = () => loadAlignPanel().catch(() => {});
  });
  ["align-cur", "align-ref"].forEach((id) => {
    $(id).onchange = () => loadAlignPanel().catch(() => {});
  });
  $("btn-align-manual").onclick = () => {
    const p = $("manual-panel");
    p.hidden = !p.hidden;
    if (!p.hidden) manualPreview();
  };
  $("btn-align-apply-all").onclick = alignApplyAll;
  $("btn-manual-preview").onclick = manualPreview;
  $("btn-manual-apply").onclick = manualApplySingle;
  for (const [id, key, digits] of [["m-dx", "dx", 1], ["m-dy", "dy", 1], ["m-rot", "rot", 2]]) {
    $(id).oninput = (e) => { $(`${id}-v`).textContent = signed(+e.target.value, digits); };
  }
  ["m-dx", "m-dy", "m-rot", "m-scale"].forEach((id) => {
    $(id).addEventListener("input", () => {
      if (!$("manual-panel").hidden) manualPreviewDebounced();
    });
  });
  $("m-scale-enable").onchange = (e) => { $("m-scale").disabled = !e.target.checked; };
  $("m-scale").oninput = (e) => { $("m-scale-v").textContent = (+e.target.value).toFixed(3); };
  if (!S.lastEstimate) await loadAlignPanel();
}

function manualTransform() {
  return {
    dx: +$("m-dx").value, dy: +$("m-dy").value, rotation: +$("m-rot").value,
    scale: $("m-scale-enable").checked ? +$("m-scale").value : 1.0,
  };
}

const manualPreviewDebounced = debounce(() => manualPreview(), 120);
async function manualPreview() {
  try {
    const t = manualTransform();
    const r = await post("/api/align/preview", {
      index: +$("align-cur").value, ref_index: +$("align-ref").value, ...t,
    });
    await drawUrl($("align-aligned"), r.aligned_png);
    await drawUrl($("align-diff-after"), r.diff);
    readout(t.dx, t.dy, t.rotation);
  } catch (e) { setStatus(String(e.message || e), "error"); }
}

async function manualApplySingle() {
  try {
    const t = manualTransform();
    const r = await post("/api/align/apply", {
      mode: "single", index: +$("align-cur").value, ref_index: +$("align-ref").value, manual: t,
    });
    await refreshSummary();
    msg("align-status", `slice #${r.info.index} aligned (dx ${signed(t.dx)}, dy ${signed(t.dy)}, rot ${signed(t.rotation)}°)`);
    await loadAlignPanel();
  } catch (e) { msg("align-status", String(e.message || e), "error"); }
}

async function alignApplyAll() {
  try {
    msg("align-status", "aligning every slice …");
    setStatus("applying alignment to all slices …");
    const r = await post("/api/align/apply", {
      mode: "all", ref_index: +$("align-ref").value, method: currentAlignMethod(),
      scope: "all", manual: $("manual-panel").hidden ? null : manualTransform(),
    });
    await refreshSummary();
    const info = r.info || {};
    msg("align-status", `${info.slices_transformed} slices registered · max |dx| ${fmt(info.max_abs_dx, 2)} px`);
    setStatus("alignment applied to all slices");
    await loadAlignPanel();
    S.viewer?.refresh();
  } catch (e) { msg("align-status", String(e.message || e), "error"); }
}

// -------------------------------------------------------------- pipeline tab
const ENUMS = {
  beam_hardening: { method: [["polynomial", "Polynomial"], ["water", "Water calibration"], ["thickness", "Chord-length"], ["spectral", "Spectral (Cu/Al)"]] },
  align: { method: [["phase", "Phase correlation"], ["cross", "Cross-correlation"], ["feature", "Feature-based"], ["mutualinfo", "Mutual information"]],
           scope: [["all", "All slices"], ["range", "Range"], ["single", "Single slice"]] },
  denoise: { method: [["gaussian", "Gaussian"], ["median", "Median"], ["bilateral", "Bilateral"]] },
  segmentation: { method: [["otsu", "Otsu"], ["multi_otsu", "Multi-Otsu"], ["manual", "Manual threshold"]] },
  calibration: { target: [["density", "Density (g/cm³)"], ["HU", "HU"], ["mu", "Linear attenuation (1/mm)"]] },
  export: { format: [["tiff", "TIFF stack"], ["png_zip", "PNG zip"], ["raw", "RAW"], ["nrrd", "NRRD"]] },
  crop: { mode: [["shape", "Shape ROI"], ["auto", "Auto specimen boundary"], ["clear", "Clear mask"]] },
  reconstruction: {},
};

async function initPipelineTab() {
  const { stages, plan } = await get("/api/pipeline/stages");
  S.stageSpecs = stages;
  if (!S.plan.length) S.plan = JSON.parse(JSON.stringify(plan));
  renderStages();
  $("btn-pipeline-run").onclick = runPipeline;
  $("btn-pipeline-default").onclick = async () => {
    const p = await get("/api/pipeline/stages");
    S.plan = JSON.parse(JSON.stringify(p.default_plan || p.plan));
    renderStages();
    savePlan();
  };
}

function renderStages() {
  const box = $("pipeline-stages");
  box.innerHTML = "";
  const byId = Object.fromEntries((S.stageSpecs || []).map((s) => [s.id, s]));
  S.plan.forEach((item, i) => {
    const spec = byId[item.id] || { name: item.id, description: "", params: {} };
    const row = document.createElement("div");
    row.className = "stage" + (item.enabled ? "" : " off");
    row.dataset.id = item.id;

    const params = { ...spec.params, ...(item.params || {}) };
    const paramHtml = Object.entries(params).map(([k, v]) => paramControl(i, item.id, k, v)).join("");

    row.innerHTML = `
      <div class="order">${i + 1}</div>
      <input type="checkbox" ${item.enabled ? "checked" : ""} title="enable stage" />
      <div class="name">${spec.name || item.id}</div>
      <div class="tools">
        <span class="stage-status" data-status="${item.id}"></span>
        <button class="icon-btn" data-move="up" ${i === 0 ? "disabled" : ""}>↑</button>
        <button class="icon-btn" data-move="down" ${i === S.plan.length - 1 ? "disabled" : ""}>↓</button>
      </div>
      <div class="desc">${spec.description || ""}</div>
      <details><summary>parameters</summary><div class="params">${paramHtml}</div></details>`;

    row.querySelector("input[type=checkbox]").onchange = (e) => {
      item.enabled = e.target.checked;
      row.classList.toggle("off", !item.enabled);
      savePlan();
    };
    row.querySelector('[data-move="up"]').onclick = () => moveStage(i, -1);
    row.querySelector('[data-move="down"]').onclick = () => moveStage(i, 1);
    row.querySelectorAll("[data-param]").forEach((el) => {
      el.onchange = () => {
        const key = el.dataset.param;
        let v = el.type === "checkbox" ? el.checked
          : el.type === "number" ? (el.value === "" ? null : +el.value)
          : el.value;
        if (el.tagName === "TEXTAREA") {
          try { v = JSON.parse(el.value); } catch { el.style.outline = "1px solid #ff6b5e"; return; }
          el.style.outline = "";
        }
        item.params = { ...(item.params || {}), [key]: v };
        savePlan();
      };
    });
    box.appendChild(row);
  });
}

function paramControl(stageIdx, stageId, key, val) {
  const id = `p-${stageIdx}-${key}`;
  const opts = ENUMS[stageId]?.[key];
  if (opts) {
    const sel = opts.map(([v, label]) =>
      `<option value="${v}" ${String(val) === v ? "selected" : ""}>${label}</option>`).join("");
    return `<label>${key}<select id="${id}" data-param="${key}">${sel}</select></label>`;
  }
  if (typeof val === "boolean") {
    return `<label>${key}<input type="checkbox" id="${id}" data-param="${key}" ${val ? "checked" : ""} /></label>`;
  }
  if (typeof val === "number") {
    return `<label>${key}<input type="number" id="${id}" data-param="${key}" value="${val}" step="any" /></label>`;
  }
  if (val && typeof val === "object") {
    return `<label>${key}<textarea id="${id}" data-param="${key}" rows="2" cols="26">${JSON.stringify(val)}</textarea></label>`;
  }
  return `<label>${key}<input type="text" id="${id}" data-param="${key}" value="${val ?? ""}" /></label>`;
}

function moveStage(i, delta) {
  const j = i + delta;
  if (j < 0 || j >= S.plan.length) return;
  const [it] = S.plan.splice(i, 1);
  S.plan.splice(j, 0, it);
  renderStages();
  savePlan();
}

async function savePlan() {
  try { await post("/api/pipeline/plan", { plan: S.plan }); } catch { /* ignore */ }
}

let pollTimer = null;
async function runPipeline() {
  try {
    await savePlan();
    $("pipeline-log").innerHTML = "";
    $("pipeline-bar").style.width = "0%";
    msg("pipeline-status", "running …");
    $("btn-pipeline-run").disabled = true;
    await post("/api/pipeline/run", { plan: S.plan });
    pollTimer = setInterval(pollPipeline, 600);
  } catch (e) {
    msg("pipeline-status", String(e.message || e), "error");
    $("btn-pipeline-run").disabled = false;
  }
}

async function pollPipeline() {
  try {
    const j = await get("/api/pipeline/progress");
    const p = j.progress || {};
    const pct = p.n ? Math.round((p.i / p.n) * 100) : 0;
    $("pipeline-bar").style.width = `${pct}%`;
    msg("pipeline-status", p.stage ? `${p.stage} (${p.i}/${p.n}) · ${p.status}` : (p.state || ""));
    const log = $("pipeline-log");
    for (const r of j.results || []) {
      const key = `data-log-${r.id}`;
      if (log.querySelector(`[data-log2="${r.id}"]`)) continue;
      const div = document.createElement("div");
      div.dataset.log2 = r.id;
      div.className = r.ok ? "ok" : "err";
      div.textContent = r.ok
        ? `✔ ${r.name || r.id}  ${r.seconds}s  ${JSON.stringify(r.info || {}).slice(0, 220)}`
        : `✘ ${r.name || r.id}  ${r.error} ${(r.trace || "").split("\n").slice(0, 2).join(" ⏎ ")}`;
      log.appendChild(div);
      const st = document.querySelector(`[data-status="${r.id}"]`);
      if (st) { st.textContent = r.ok ? "✔" : "✘"; st.className = `stage-status ${r.ok ? "ok" : "err"}`; }
    }
    if (!j.running && (p.state === "done" || p.state === "error")) {
      clearInterval(pollTimer);
      pollTimer = null;
      $("btn-pipeline-run").disabled = false;
      $("pipeline-bar").style.width = "100%";
      const bad = (j.results || []).filter((r) => !r.ok);
      msg("pipeline-status", bad.length
        ? `finished with ${bad.length} failed stage(s)`
        : `finished in ${p.seconds ?? "?"}s · ${(j.results || []).length} stages`,
        bad.length ? "error" : "");
      if (j.error) {
        const div = document.createElement("div");
        div.className = "err";
        div.textContent = j.error;
        log.appendChild(div);
      }
      await refreshSummary();
      setStatus("pipeline complete");
      if (j.reconstruction) $("vol-source").value = "reconstruction";
    }
  } catch (e) {
    clearInterval(pollTimer);
    msg("pipeline-status", String(e.message || e), "error");
    $("btn-pipeline-run").disabled = false;
  }
}

// ----------------------------------------------------------- analysis tab
async function initAnalysisTab() {
  $("btn-analysis").onclick = runAnalysis;
  document.querySelectorAll("[data-export]").forEach((a) => {
    a.setAttribute("href", `/api/export/${a.dataset.export}`);
    a.setAttribute("download", "");
  });
  wirePoreSection();
  runAnalysis(); // show current results immediately; button re-runs
  restorePores(); // re-show a stored pore-size result after refresh
}

async function runAnalysis() {
  msg("analysis-status", "analysing …");
  try {
    const r = await get("/api/analysis");
    const rep = r.report || {};
    const rows = [];
    const walk = (o, p) => {
      for (const [k, v] of Object.entries(o || {})) {
        const key = p ? `${p}.${k}` : k;
        if (v && typeof v === "object" && !Array.isArray(v)) walk(v, key);
        else if (Array.isArray(v)) continue;
        else rows.push([key, typeof v === "number" ? fmt(v, v % 1 ? 3 : 1) : String(v)]);
      }
    };
    walk({ unit: rep.unit, spacing_mm: rep.spacing_mm, voxel_volume_mm3: rep.voxel_volume_mm3,
           global: rep.global, roi: rep.roi }, "");
    $("stats-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

    const cls = r.segmentation_classes || [];
    $("classes-table").querySelector("tbody").innerHTML = cls.map((c) =>
      `<tr><td>${c.label ?? c.class ?? c.name ?? "-"}</td><td>${c.voxels ?? "-"}</td>` +
      `<td>${fmt(c.volume_cm3 ?? c.volume_mm3, 3)}</td><td>${fmt(c.mean ?? c.mean_value, 1)}</td></tr>`).join("")
      || "<tr><td colspan=4>run the segmentation stage first</td></tr>";

    if (rep.histogram?.counts) {
      drawBars($("hist-canvas"), null, rep.histogram.counts, { color: "#62c29a" });
    }
    const rp = r.radial?.profile || r.radial;
    if (rp?.radius?.length && rp?.mean?.length) {
      drawProfile($("radial-canvas"), rp.radius, rp.mean, { title: "radial profile" });
    }
    // pore-size distribution from the report (shown until a manual run overrides it)
    if (!S.poreResult && rep.pore_size?.distribution?.counts) {
      renderPoreResults(rep.pore_size, null, { fromReport: true });
    }
    msg("analysis-status", "analysis ready");
    setStatus("analysis ready");
  } catch (e) { msg("analysis-status", String(e.message || e), "error"); }
}

// --------------------------------------------------------------- lifecycle
function resetViewers() {
  if (S.viewer) { S.win = { lo: null, hi: null }; autoWindow().then(() => S.viewer.refresh()); }
  if (S.vol) loadBrick();
  loadHistogram();
}

async function boot() {
  wireTabs();
  wireImport();
  $("btn-undo").onclick = async () => {
    try {
      const r = await post("/api/undo");
      await refreshSummary();
      setStatus(r.ok ? "undone" : "nothing to undo");
      S.viewer?.refresh();
      S.roiViewer?.refresh();
      loadHistogram();
    } catch (e) { setStatus(String(e.message || e), "error"); }
  };
  await refreshSummary();
  if (S.summary?.loaded) {
    S.slice.index = Math.floor(S.summary.shape.z / 2);
    $("slice-slider").max = S.summary.shape.z - 1;
    $("slice-label").textContent = `#${S.slice.index} / ${S.summary.shape.z - 1}`;
  }
  await renderDetections();
  setStatus("ready");
  $("status-right").textContent = S.summary?.loaded
    ? `${S.summary.shape.z}×${S.summary.shape.y}×${S.summary.shape.x}` : "";
  // deep-link: #viewer #volume #bh #align #pipeline #analysis open that tab
  const hash = (location.hash || "").replace(/^#/, "");
  const deep = $("tabs")?.querySelector(`button[data-tab="${hash}"]`);
  if (deep) deep.click();
}

boot().catch((e) => setStatus(String(e.message || e), "error"));
