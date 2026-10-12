"""FastAPI backend for the Rockstone CT image-processing workbench.

JSON/binary API under ``/api/*``; the browser UI in ``web/`` is mounted at ``/``.

Run:  .venv/Scripts/python -m uvicorn server.app:app --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from server.io import exporters
from server.io.importers import import_path
from server import report as report_mod
from server.processing import analysis as analysis_mod
from server.processing import beamhardening as bh
from server.processing import calibration, cracks as cracks_mod, filters, pores as pores_mod, roi as roi_mod, segment
from server.processing.align import align_volume, apply_transform, estimate
from server.processing.pipeline import (
    PipelineContext,
    _jsonable,
    default_plan,
    preview_stage,
    run_pipeline,
    stage_specs,
)
from server.store import WORKSPACE, Volume
from server.synth import load_or_generate

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"
UPLOAD_DIR = ROOT / "data" / "uploads"
SAMPLE_SHAPE = (96, 256, 256)

ws = WORKSPACE

APP: dict[str, Any] = {
    "version": 0,            # bumped whenever voxels change (invalidates caches)
    "plan": default_plan(),
    "running": False,
    "progress": {"state": "idle", "i": 0, "n": 0, "stage": None, "status": None},
    "results": [],
    "reconstruction": {},
    "exports": {},
    "report": {},
    "error": None,
    "_win": None,            # cached default window/level
}


# --------------------------------------------------------------- helpers
def _bump() -> None:
    APP["version"] += 1
    APP["_win"] = None


def _volume() -> Volume:
    if ws.volume is None:
        raise HTTPException(400, "no volume loaded - import a dataset first")
    return ws.volume


def _clamp(index: int | None) -> int:
    z = _volume().shape[0]
    if index is None or index < 0:
        return z // 2
    return int(min(max(index, 0), z - 1))


def _window(lo: float | None = None, hi: float | None = None,
            w: float | None = None, l: float | None = None) -> tuple[float, float]:
    """Window/level: explicit w/l > explicit lo/hi > cached volume percentiles."""
    if w is not None and l is not None:
        return float(l - w / 2.0), float(l + w / 2.0)
    if lo is not None and hi is not None and hi > lo:
        return float(lo), float(hi)
    cached = APP.get("_win")
    if cached and cached[0] == APP["version"]:
        return cached[1], cached[2]
    data = _volume().data
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        p1, p99 = 0.0, 1.0
    else:
        p1, p99 = (float(v) for v in np.percentile(finite, [1.0, 99.0]))
    if p99 - p1 < 1e-6:
        p99 = p1 + 1.0
    APP["_win"] = (APP["version"], p1, p99)
    return p1, p99


def _gray_png(arr: np.ndarray, lo: float, hi: float, invert: bool = False) -> bytes:
    from PIL import Image

    a = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=lo, posinf=hi, neginf=lo)
    q = np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if invert:
        q = 1.0 - q
    img = Image.fromarray((q * 255.0).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _url(arr: np.ndarray, lo: float, hi: float, invert: bool = False) -> str:
    return "data:image/png;base64," + base64.b64encode(_gray_png(arr, lo, hi, invert)).decode()


def _parse_color(value: Any) -> tuple[int, int, int] | None:
    """'#rrggbb' / 'rrggbb' / 'r,g,b' → (r, g, b); anything else → None."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip().lstrip("#")
    try:
        if "," in s:
            parts = [int(p.strip()) for p in s.split(",")]
            if len(parts) != 3:
                return None
            return tuple(max(0, min(255, p)) for p in parts)  # type: ignore[return-value]
        if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None
    return None


_CRACK_TINT_DEFAULT = (255, 150, 60)  # orange
_PORE_TINT_DEFAULT = (64, 210, 190)  # teal


def _color_hex(color: tuple[int, int, int] | None,
               default: tuple[int, int, int] | None = None) -> str:
    r, g, b = color or default or _CRACK_TINT_DEFAULT
    return f"#{r:02x}{g:02x}{b:02x}"


def _overlay_url(arr: np.ndarray, lo: float, hi: float,
                 mask: np.ndarray | None = None, labels: np.ndarray | None = None,
                 cracks: np.ndarray | None = None,
                 crack_color: tuple[int, int, int] | None = None,
                 overlay_color: tuple[int, int, int] | None = None) -> str:
    """Grayscale slice with an optional label tint and red mask outline."""
    from PIL import Image

    a = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=lo)
    q = np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    rgb = np.repeat((q * 255.0).astype(np.uint8)[:, :, None], 3, axis=2)
    if labels is not None:
        lab = np.asarray(labels)
        palette = np.array([[80, 200, 255], [255, 170, 60], [120, 255, 140],
                            [255, 100, 200], [200, 200, 90], [150, 140, 255]], dtype=np.uint8)
        if overlay_color is not None:
            # hue-rotate the palette so class 1 matches the chosen color
            # (saturation/value preserved → classes stay distinguishable)
            import colorsys

            def _hsv(px: Any) -> tuple[float, float, float]:
                return colorsys.rgb_to_hsv(*(int(c) / 255 for c in px))

            delta = (_hsv(overlay_color)[0] - _hsv(palette[0])[0]) % 1.0
            rotated = []
            for px in palette:
                h, s, v = _hsv(px)
                rotated.append([round(c * 255) for c in
                                colorsys.hsv_to_rgb((h + delta) % 1.0, s, v)])
            palette = np.array(rotated, dtype=np.uint8)
        sel = lab > 0
        if sel.any():
            tint = palette[(lab[sel] - 1) % len(palette)]
            rgb[sel] = (0.55 * rgb[sel] + 0.45 * tint).astype(np.uint8)
    if cracks is not None:
        ck = np.asarray(cracks, dtype=bool)
        if ck.any():
            from scipy import ndimage
            rim = ck ^ ndimage.binary_erosion(ck, np.ones((3, 3), bool))
            if crack_color is not None:
                tint = np.array(crack_color, dtype=np.uint8)
                edge = (0.55 * tint.astype(np.float32)).astype(np.uint8)
            else:
                tint = np.array(_CRACK_TINT_DEFAULT, dtype=np.uint8)
                edge = np.array([255, 90, 20], dtype=np.uint8)
            rgb[ck] = tint
            rgb[rim] = edge
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        from scipy import ndimage
        edge = m ^ ndimage.binary_erosion(m, np.ones((3, 3), bool))
        rgb[edge] = (np.array(overlay_color, dtype=np.uint8)
                     if overlay_color is not None else np.array([255, 60, 60], dtype=np.uint8))
        rgb[~m & ~edge] = (0.6 * rgb[~m & ~edge]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _diff_url(a: np.ndarray, b: np.ndarray) -> str:
    """Signed difference b-a as a blue/white/red diverging PNG."""
    from PIL import Image

    d = np.nan_to_num(np.asarray(b, np.float32) - np.asarray(a, np.float32))
    span = float(np.percentile(np.abs(d), 98.0)) or 1.0
    t = np.clip(d / max(span, 1e-9), -1.0, 1.0)
    # diverging: white = no difference, red = positive, blue = negative
    r = 1.0 - np.clip(-t, 0.0, 1.0)
    bl = 1.0 - np.clip(t, 0.0, 1.0)
    g = 1.0 - np.abs(t)
    rgb = (np.stack([r, g, bl], axis=-1) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _commit(data: np.ndarray, label: str) -> None:
    """Snapshot history *then* swap the voxels (so undo returns to the old state)."""
    ws.push_history(label)
    ws.replace_data(np.asarray(data, dtype=np.float32))
    ws.segmentation = None
    _bump()


def _load_volume(vol: Volume) -> dict[str, Any]:
    ws.set_volume(vol)
    for key in ("reconstruction", "exports", "report"):
        APP[key] = {}
    APP["results"] = []
    APP["progress"] = {"state": "idle", "i": 0, "n": 0, "stage": None, "status": None}
    _bump()
    return ws.summary()


def _ensure_volume() -> None:
    if ws.volume is None:
        try:
            ws.set_volume(load_or_generate(SAMPLE_SHAPE))
        except Exception:  # noqa: BLE001 - startup must never crash the server
            pass
        _bump()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _ensure_volume()
    yield


app = FastAPI(title="Rockstone CT Workbench", version="1.0", lifespan=_lifespan)


# --------------------------------------------------------------- workspace
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "loaded": ws.volume is not None, "version": APP["version"]}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    out = ws.summary()
    out["version"] = APP["version"]
    out["report"] = APP.get("report") or None
    out["running"] = APP["running"]
    return out


@app.post("/api/sample")
def load_sample(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    shape = tuple(body.get("shape") or SAMPLE_SHAPE)
    vol = load_or_generate(shape, seed=int(body.get("seed", 7)),
                           fractures=bool(body.get("fractures", False)))
    return _load_volume(vol)


@app.post("/api/undo")
def undo() -> dict[str, Any]:
    ok = ws.undo()
    if ok:
        _bump()
    return {"ok": ok, "summary": ws.summary()}


# --------------------------------------------------------------- import
@app.get("/api/browse")
def browse(path: str | None = None) -> dict[str, Any]:
    p = Path(path).expanduser() if path else ROOT
    if not p.exists():
        raise HTTPException(404, f"path not found: {p}")
    if p.is_file():
        p = p.parent
    dirs, files = [], []
    try:
        for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
            (dirs if child.is_dir() else files).append(child.name)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs[:400], "files": files[:400]}


@app.post("/api/import")
def import_path_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    path = body.get("path")
    if not path:
        raise HTTPException(400, "path is required")
    spacing = body.get("spacing")
    try:
        vol = import_path(str(path), kind=body.get("kind", "auto"),
                          raw_params=body.get("raw_params"),
                          spacing=tuple(spacing) if spacing else None)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.get("spacing"):  # user override wins over detected values
        vol.spacing = tuple(float(s) for s in body["spacing"])
        vol.metadata.setdefault("spacing_source", {})
        for axis in ("z", "y", "x"):
            vol.metadata["spacing_source"][axis] = "user"
        det = vol.metadata.get("detections", {})
        det["slice_spacing"] = {"value": vol.spacing[0], "source": "user"}
        det["pixel_size"] = {"value": (vol.spacing[1], vol.spacing[2]), "source": "user"}
    return {"summary": _load_volume(vol), "detections": vol.metadata.get("detections", {})}


@app.post("/api/import/upload")
async def import_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(400, "no files uploaded")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = UPLOAD_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "upload.bin").name
        (out_dir / name).write_bytes(await f.read())
    try:
        vol = import_path(str(out_dir), kind="auto")
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"summary": _load_volume(vol), "detections": vol.metadata.get("detections", {})}


@app.get("/api/detections")
def detections() -> dict[str, Any]:
    return {"detections": _volume().metadata.get("detections", {}),
            "metadata": _volume().metadata, "unit": _volume().unit}


# --------------------------------------------------------------- 2-D viewer
@app.get("/api/slice.png")
def slice_png(index: int = -1, w: float | None = None, l: float | None = None,
              lo: float | None = None, hi: float | None = None, invert: int = 0,
              overlay: str | None = None, color: str | None = None) -> Response:
    data = _volume().data
    i = _clamp(index)
    v_lo, v_hi = _window(lo, hi, w, l)
    sl = data[i]
    mask = None
    if overlay == "mask":
        mask = ws.mask2d if ws.mask2d is not None else (
            ws.mask3d[i] if ws.mask3d is not None else None)
    if overlay == "cracks":
        ck = ws.crack_labels[i] if ws.crack_labels is not None else None
        tint = _parse_color(color) or ws.crack_color
        png = base64.b64decode(
            _overlay_url(sl, v_lo, v_hi, cracks=ck, crack_color=tint).split(",", 1)[1])
    elif overlay == "pores":
        pr = ws.pore_labels[i] if ws.pore_labels is not None else None
        tint = _parse_color(color) or ws.pore_color
        png = base64.b64decode(
            _overlay_url(sl, v_lo, v_hi, cracks=pr,
                         crack_color=tint or _PORE_TINT_DEFAULT).split(",", 1)[1])
    elif overlay in ("mask", "segmentation") or mask is not None:
        labels = ws.segmentation[i] if (overlay == "segmentation" and
                                        ws.segmentation is not None) else None
        png = base64.b64decode(_overlay_url(sl, v_lo, v_hi, mask=mask, labels=labels,
                                            overlay_color=_parse_color(color)).split(",", 1)[1])
    else:
        png = _gray_png(sl, v_lo, v_hi, invert=bool(invert))
    return Response(png, media_type="image/png",
                    headers={"X-Window-Lo": str(v_lo), "X-Window-Hi": str(v_hi)})


@app.get("/api/histogram")
def histogram(bins: int = 128, stride: int = 4) -> dict[str, Any]:
    """Coarse value histogram of the volume (strided sample -> cheap)."""
    data = _volume().data
    flat = data[::max(1, int(stride)), ::max(1, int(stride)), ::max(1, int(stride))].ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"bins": [], "counts": [], "lo": 0.0, "hi": 1.0, "unit": _volume().unit}
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    counts, edges = np.histogram(finite, bins=max(8, int(bins)), range=(lo, hi))
    return {"bins": [float(e) for e in edges], "counts": [int(c) for c in counts],
            "lo": lo, "hi": hi, "unit": _volume().unit, "sampled": int(finite.size)}


@app.get("/api/slice/info")
def slice_info(index: int = -1) -> dict[str, Any]:
    data = _volume().data
    i = _clamp(index)
    sl = data[i]
    finite = sl[np.isfinite(sl)]
    lo, hi = _window()
    return {"index": i, "shape": [int(sl.shape[0]), int(sl.shape[1])],
            "min": float(finite.min()), "max": float(finite.max()),
            "mean": float(finite.mean()), "std": float(finite.std()),
            "window": {"lo": lo, "hi": hi}, "unit": _volume().unit,
            "has_mask": ws.mask2d is not None or ws.mask3d is not None,
            "has_segmentation": ws.segmentation is not None}


# --------------------------------------------------------------- 3-D viewer
@app.get("/api/volume/brick")
def volume_brick(max_dim: int = 192, smoothing: float = 0.0,
                 source: str = "volume") -> Response:
    from scipy import ndimage

    rec = APP.get("reconstruction") or {}
    if source == "reconstruction" and rec.get("brick") is not None:
        q = np.ascontiguousarray(rec["brick"], dtype=np.uint8)
        lo, hi = rec.get("value_range", [0.0, 1.0])
        factor = rec.get("factor", 1)
        shape = list(q.shape)
    else:
        data = _volume().data
        max_dim = min(max(int(max_dim), 16), 2048)  # clamp: sane floor/ceiling
        factor = max(1, int(np.ceil(max(max(data.shape) / max_dim, 1))))
        brick = data[::factor, ::factor, ::factor]
        if smoothing and float(smoothing) > 0:
            brick = ndimage.gaussian_filter(brick, float(smoothing))
        lo = float(np.nanmin(brick))
        hi = float(np.nanmax(brick))
        q = np.clip((brick - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
        q = np.ascontiguousarray(q)
        shape = list(q.shape)
    spacing = [s * factor for s in _volume().spacing]
    return Response(q.tobytes(), media_type="application/octet-stream", headers={
        "X-Shape": ",".join(str(s) for s in shape),
        "X-Lo": str(lo), "X-Hi": str(hi), "X-Factor": str(factor),
        "X-Spacing": ",".join(str(s) for s in spacing),
        "X-Unit": _volume().unit,
    })


# --------------------------------------------------------------- beam hardening
@app.post("/api/bh/detect")
def bh_detect(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    vol = _volume()
    data = vol.data
    mask = ws.effective_mask()
    if body.get("index") is not None:
        i = _clamp(int(body["index"]))
        sl = data[i]
        m2 = mask[i] if (mask is not None and mask.ndim == 3) else mask
        if m2 is None:
            m2, _info = roi_mod.auto_boundary(sl)
        det = bh.detect(sl[None, ...], m2, vol.unit, vol.spacing)
        profile = bh.radial_profile(sl, m2)
    else:
        i = data.shape[0] // 2
        m2 = _mid_mask(mask)
        if m2 is None:
            m2, _info = roi_mod.auto_boundary(data[i])
        det = bh.detect(data, mask, vol.unit, vol.spacing)
        profile = bh.radial_profile(data[i], m2)
    det = {k: v for k, v in det.items() if k != "mask"}
    profile.pop("mask", None)
    return {"detect": det, "profile": profile, "index": i, "unit": vol.unit,
            "acquisition": vol.metadata.get("acquisition", {})}


def _mid_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return mask[mask.shape[0] // 2] if mask.ndim == 3 else mask


def _bh_params(body: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    """method/params/strength with acquisition defaults for the spectral model."""
    method = body.get("method", "thickness")
    params = dict(body.get("params") or {})
    strength = float(body.get("strength", 1.0))
    if method == "spectral":
        acq = _volume().metadata.get("acquisition") or {}
        for k in ("kvp", "inherent_al_mm", "added_filter", "added_mm", "material"):
            if params.get(k) in (None, "") and acq.get(k) is not None:
                params[k] = acq[k]
        if params.get("mu_water_reference") is None:
            params["mu_water_reference"] = _volume().metadata.get("mu_water_reference")
    return method, params, strength


@app.post("/api/bh/preview")
def bh_preview(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    method, params, strength = _bh_params(body)
    i = _clamp(body.get("index"))
    prev = preview_stage(vol.data, "beam_hardening",
                         {**params, "method": method, "strength": strength},
                         vol.unit, vol.spacing, ws.effective_mask(), index=i,
                         metadata=vol.metadata)
    lo, hi = _window()
    before, after = prev["before"], prev["after"]
    mask = _mid_mask(ws.effective_mask())
    return {
        "index": i,
        "before": _url(before, lo, hi),
        "after": _url(after, lo, hi),
        "diff": _diff_url(before, after),
        "info": prev["info"],
        "detect_before": {k: v for k, v in bh.detect(before[None, ...], mask, vol.unit,
                                                     vol.spacing).items() if k != "mask"},
        "detect_after": {k: v for k, v in bh.detect(after[None, ...], mask, vol.unit,
                                                    vol.spacing).items() if k != "mask"},
        "window": {"lo": lo, "hi": hi},
    }


@app.post("/api/bh/apply")
def bh_apply(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    method, params, strength = _bh_params(body)
    mask = ws.effective_mask()
    detect_before = {k: v for k, v in bh.detect(vol.data, mask, vol.unit,
                                                vol.spacing).items() if k != "mask"}
    try:
        out, info = bh.correct(vol.data, method, params, strength=strength, unit=vol.unit,
                               mask=mask, spacing=vol.spacing,
                               mu_water=vol.metadata.get("mu_water_reference"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"{method} failed: {exc}") from exc
    _commit(out, f"beam-hardening correction ({method})")
    detect_after = {k: v for k, v in bh.detect(out, mask, vol.unit, vol.spacing).items()
                    if k != "mask"}
    return {"info": info, "detect_before": detect_before, "detect_after": detect_after,
            "summary": ws.summary()}


@app.post("/api/bh/standards/measure")
def bh_measure_standard(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    spec = dict(body.get("spec") or {"auto": "water"})
    try:
        stats = bh.measure_standard(vol.data, spec, ws.effective_mask())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    stats["true"] = spec.get("true")
    vol.metadata.setdefault("standards", []).append(stats)
    return {"standard": stats, "standards": vol.metadata["standards"]}


# --------------------------------------------------------------- ROI / mask
def _roi_mask(slice2d: np.ndarray, body: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    spec = dict(body.get("spec") or {"type": "frame"})
    params = dict(body.get("params") or {})
    t = spec.get("type", "frame")
    if t == "auto":
        return roi_mod.auto_boundary(slice2d, {**params, **spec})
    if t == "remove_air":
        m = roi_mod.remove_air(slice2d, spec.get("threshold"),
                               float(spec.get("margin_sigma", 2.0)))
        return m, {"threshold": spec.get("threshold", "auto"), "area_fraction": float(m.mean())}
    if t == "artifact":
        m = roi_mod.artifact_mask(slice2d, {**params, **spec})
        return m, {"mode": spec.get("mode", "residual"), "area_fraction": float(m.mean()),
                   "keep": "outside"}
    m = roi_mod.shape_mask(slice2d.shape, spec)
    return m, {"area_fraction": float(m.mean())}


@app.post("/api/roi/preview")
def roi_preview(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    i = _clamp(body.get("index"))
    sl = vol.data[i]
    mask, info = _roi_mask(sl, body)
    fill = float(body.get("fill", -1000.0 if vol.unit == "HU" else 0.0))
    after = sl.copy()
    after[~mask] = fill
    lo, hi = _window()
    return {"index": i, "before": _url(sl, lo, hi),
            "after": _url(after, lo, hi),
            "overlay": _overlay_url(sl, lo, hi, mask=mask),
            "info": info, "area_px": int(mask.sum()),
            "area_fraction": float(mask.mean()), "fill": fill,
            "window": {"lo": lo, "hi": hi}}


@app.post("/api/roi/area")
def roi_area(body: dict[str, Any]) -> dict[str, Any]:
    """Area of the current ROI spec on one slice, in px² and mm² (read-only)."""
    vol = _volume()
    i = _clamp(body.get("index"))
    mask, info = _roi_mask(vol.data[i], body)
    dy, dx = float(vol.spacing[1]), float(vol.spacing[2])
    area_px = int(mask.sum())
    return {"index": i, "area_px": area_px,
            "area_mm2": area_px * dy * dx,
            "area_fraction": float(mask.mean()),
            "spacing_mm": [dy, dx], "info": info}


@app.post("/api/roi/apply")
def roi_apply(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    scope = body.get("scope", "slice")
    if scope == "volume":
        mask3d, info = roi_mod.auto_boundary_3d(vol.data, body.get("params") or {})
        ws.mask3d = mask3d.astype(bool)
        ws.mask2d = None
        info["scope"] = "volume"
    else:
        i = _clamp(body.get("index"))
        mask, info = _roi_mask(vol.data[i], body)
        ws.mask2d = mask.astype(bool)
        ws.mask3d = None
        info["scope"] = f"slice {i} (applied to every slice)"
    ws.push_history("ROI mask applied")
    return {"info": info, "summary": ws.summary()}


@app.post("/api/roi/clean")
def roi_clean(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    vol = _volume()
    mask = ws.effective_mask()
    if mask is None:
        raise HTTPException(400, "no mask - preview/apply an ROI first")
    fill = body.get("fill", "auto")
    if fill in (None, "auto"):
        border = np.concatenate([vol.data[..., 0].ravel(), vol.data[..., -1].ravel(),
                                 vol.data[:, 0].ravel(), vol.data[:, -1].ravel()])
        border = border[np.isfinite(border)]
        fill_val = float(np.median(border))
    else:
        fill_val = float(fill)
    out = np.where(mask, vol.data, fill_val).astype(np.float32)
    _commit(out, "ROI / clean volume")
    return {"fill": fill_val, "summary": ws.summary()}


@app.post("/api/roi/remove-air")
def roi_remove_air(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """One click: threshold air from the volume border, mask it out, fill, single undo step."""
    body = body or {}
    vol = _volume()
    data = vol.data
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise HTTPException(400, "volume has no finite voxels")
    # air level from the 4 lateral border faces (z-faces are usually specimen,
    # same convention as roi_clean's auto fill)
    border = np.concatenate([data[..., 0].ravel(), data[..., -1].ravel(),
                             data[:, 0].ravel(), data[:, -1].ravel()])
    border = border[np.isfinite(border)]
    air = float(np.median(border))
    thr = body.get("threshold")
    if thr is None:
        thr = air + float(body.get("margin_sigma", 2.0)) * (float(border.std()) + 1e-6) * 3.0
        # a sane auto threshold must keep a real specimen, not just its densest flecks
        keep = float((finite > thr).mean())
        if keep > 0.98 or keep < 0.02:
            thr = roi_mod.otsu_threshold(finite)
    thr = float(thr)
    mask = data > thr
    if not mask.any():
        raise HTTPException(400, f"threshold {thr:.1f} removes every voxel - no air detected?")
    fill_val = air if body.get("fill") in (None, "auto") else float(body.get("fill"))
    i = _clamp(body.get("index"))
    lo, hi = _window()
    before = _url(data[i], lo, hi)
    ws.mask3d = mask.astype(bool)
    ws.mask2d = None
    out = np.where(mask, data, fill_val).astype(np.float32)
    after = _url(out[i], lo, hi)
    _commit(out, "remove background air")
    return {"threshold": thr, "air": air, "fill": fill_val, "index": i,
            "before": before, "after": after,
            "area_fraction": float(mask.mean()), "summary": ws.summary()}


@app.delete("/api/roi/mask")
def roi_clear() -> dict[str, Any]:
    ws.mask2d = ws.mask3d = None
    return {"ok": True, "summary": ws.summary()}


# --------------------------------------------------------------- alignment
@app.get("/api/align/panel")
def align_panel(index: int = -1, ref: int = -1, method: str = "phase",
                params: str | None = None) -> dict[str, Any]:
    vol = _volume()
    data = vol.data
    i = _clamp(index if index >= 0 else 1)
    r = _clamp(ref if ref >= 0 else 0)
    if i == r:
        i = _clamp(r + 1)
    est_params = json.loads(params) if params else {}
    est = estimate(data[r], data[i], method, est_params)
    lo, hi = _window()
    aligned = apply_transform(data[i], est["dx"], est["dy"], est["rotation"], est["scale"],
                              cval=float(np.median(data[i])))
    return {"index": i, "reference": r, "method": method, "estimate": est,
            "readout": {"dx": est["dx"], "dy": est["dy"], "rotation": est["rotation"]},
            "ref_png": _url(data[r], lo, hi), "cur_png": _url(data[i], lo, hi),
            "aligned_png": _url(aligned, lo, hi),
            "diff": _diff_url(data[r], data[i]),
            "diff_aligned": _diff_url(data[r], aligned),
            "window": {"lo": lo, "hi": hi}}


@app.post("/api/align/estimate")
def align_estimate(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    data = vol.data
    i = _clamp(body.get("index"))
    r = _clamp(body.get("ref_index", 0))
    method = body.get("method", "phase")
    try:
        est = estimate(data[r], data[i], method, body.get("params") or {})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"estimate": est, "index": i, "reference": r}


@app.post("/api/align/preview")
def align_preview(body: dict[str, Any]) -> dict[str, Any]:
    """Manual-align feedback: warp the current slice by the dialled transform."""
    vol = _volume()
    data = vol.data
    i = _clamp(body.get("index"))
    r = _clamp(body.get("ref_index", 0))
    dx = float(body.get("dx", 0.0))
    dy = float(body.get("dy", 0.0))
    rot = float(body.get("rotation", 0.0))
    scale = float(body.get("scale", 1.0))
    moved = apply_transform(data[i], dx, dy, rot, scale, cval=float(np.median(data[i])))
    lo, hi = _window()
    return {"aligned_png": _url(moved, lo, hi), "diff": _diff_url(data[r], moved),
            "index": i, "reference": r}


@app.post("/api/align/apply")
def align_apply(body: dict[str, Any]) -> dict[str, Any]:
    vol = _volume()
    data = vol.data
    mode = body.get("mode", "all")
    method = body.get("method", "phase")
    params = dict(body.get("params") or {})
    manual = body.get("manual")
    ref_index = _clamp(body.get("ref_index", 0))
    ws.push_history("slice alignment")
    if mode == "single":
        i = _clamp(body.get("index"))
        if manual:
            tr = {k: float(v) for k, v in manual.items()}
        else:
            est = estimate(data[ref_index], data[i], method, params)
            tr = {"dx": est["dx"], "dy": est["dy"], "rotation": est["rotation"],
                  "scale": est["scale"]}
        out = data.copy()
        out[i] = apply_transform(data[i], tr.get("dx", 0.0), tr.get("dy", 0.0),
                                 tr.get("rotation", 0.0), tr.get("scale", 1.0),
                                 cval=float(np.median(data[i])))
        info = {"mode": "single", "index": i, **tr}
        transforms = {i: tr}
    else:
        scope = body.get("scope", "all")
        out, info, transforms = align_volume(
            data, ref_index=ref_index, method=method, params=params, scope=scope,
            manual=manual, sequential=bool(body.get("sequential", False)))
    ws.replace_data(np.asarray(out, dtype=np.float32))
    ws.transforms = {int(k): dict(v) for k, v in transforms.items()}
    ws.segmentation = None
    _bump()
    info["n_transforms"] = len(ws.transforms)
    return {"info": _jsonable(info),
            "transforms": {str(k): v for k, v in list(ws.transforms.items())[:64]},
            "summary": ws.summary()}


# --------------------------------------------------------------- filters / segment
@app.post("/api/stage/preview")
def stage_preview(body: dict[str, Any]) -> dict[str, Any]:
    """Fast single-slice preview for any pipeline stage (ring, denoise, crop, ...)."""
    vol = _volume()
    stage_id = body.get("stage_id")
    try:
        prev = preview_stage(vol.data, stage_id, body.get("params") or {},
                             vol.unit, vol.spacing, ws.effective_mask(),
                             index=body.get("index"), metadata=vol.metadata)
    except KeyError as exc:
        raise HTTPException(404, f"unknown stage: {stage_id}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"{stage_id}: {exc}") from exc
    if "before" in prev:
        lo, hi = _window()
        return {"index": prev["index"], "before": _url(prev["before"], lo, hi),
                "after": _url(prev["after"], lo, hi),
                "diff": _diff_url(prev["before"], prev["after"]),
                "info": prev["info"], "window": {"lo": lo, "hi": hi}}
    return prev


@app.post("/api/segment")
def segment_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    vol = _volume()
    method = body.get("method", "otsu")
    labels, info = segment.segment_volume(vol.data, method, body.get("params") or {},
                                          mask=ws.effective_mask())
    ws.push_history(f"segmentation ({method})")
    ws.segmentation = labels
    classes = segment.class_stats(labels, vol.voxel_volume_mm3)
    i = _clamp(body.get("index"))
    lo, hi = _window()
    m3 = ws.effective_mask()  # 3-D; the overlay needs the current 2-D slice
    return {"info": _jsonable(info), "classes": classes, "index": i,
            "overlay": _overlay_url(vol.data[i], lo, hi,
                                    labels=labels[i],
                                    mask=m3[i] if m3 is not None else None),
            "summary": ws.summary()}


@app.get("/api/cracks")
def cracks_get_endpoint(index: int | None = None) -> dict[str, Any]:
    """Stored crack result, so a browser refresh can restore the tab."""
    if ws.crack_result is None or ws.crack_labels is None or ws.volume is None:
        return {"result": None}
    i = _clamp(index)
    lo, hi = _window()
    return {"result": _jsonable(ws.crack_result), "index": i,
            "overlay": _overlay_url(ws.volume.data[i], lo, hi,
                                    cracks=ws.crack_labels[i],
                                    crack_color=ws.crack_color),
            "color": _color_hex(ws.crack_color), "summary": ws.summary()}


@app.post("/api/cracks/detect")
def cracks_detect_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Find dark elongated voids (cracks/fractures) inside the specimen."""
    body = body or {}
    vol = _volume()
    thr = body.get("threshold", "auto")
    labels, result = cracks_mod.detect_cracks(
        vol.data, vol.spacing, ws.effective_mask(),
        threshold=thr if isinstance(thr, str) else float(thr),
        min_voxels=int(body.get("min_voxels", 25)),
        min_elongation=float(body.get("min_elongation", 0.55)),
        max_aperture_mm=float(body.get("max_aperture_mm", 4.0) or 0) or None,
    )
    ws.crack_labels = labels
    ws.crack_result = result
    if "color" in body:
        ws.crack_color = _parse_color(body.get("color"))
    i = _clamp(body.get("index"))
    lo, hi = _window()
    return {"result": _jsonable(result), "index": i,
            "overlay": _overlay_url(vol.data[i], lo, hi, cracks=labels[i],
                                    crack_color=ws.crack_color),
            "color": _color_hex(ws.crack_color), "summary": ws.summary()}


@app.delete("/api/cracks")
def cracks_clear_endpoint() -> dict[str, Any]:
    ws.crack_labels = None
    ws.crack_result = None
    return {"ok": True, "summary": ws.summary()}


@app.get("/api/pores")
def pores_get_endpoint(index: int | None = None) -> dict[str, Any]:
    """Stored pore-size result, so a browser refresh can restore the tab."""
    if ws.pore_result is None or ws.pore_labels is None or ws.volume is None:
        return {"result": None}
    i = _clamp(index)
    lo, hi = _window()
    return {"result": _jsonable(ws.pore_result), "index": i,
            "overlay": _overlay_url(ws.volume.data[i], lo, hi,
                                    cracks=ws.pore_labels[i],
                                    crack_color=ws.pore_color or _PORE_TINT_DEFAULT),
            "color": _color_hex(ws.pore_color, _PORE_TINT_DEFAULT),
            "summary": ws.summary()}


@app.post("/api/pores")
def pores_analyze_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect pores inside the specimen and compute the size distribution."""
    body = body or {}
    vol = _volume()
    thr = body.get("threshold", "auto")
    labels, result = pores_mod.detect_pores(
        vol.data, vol.spacing, ws.effective_mask(),
        threshold=thr if isinstance(thr, str) else float(thr),
        min_voxels=int(body.get("min_voxels", 8)),
        max_elongation=(float(body["max_elongation"])
                        if body.get("max_elongation") not in (None, "") else 0.7),
        bins=int(body.get("bins", 24)),
    )
    ws.pore_labels = labels
    ws.pore_result = result
    if "color" in body:
        ws.pore_color = _parse_color(body.get("color"))
    i = _clamp(body.get("index"))
    lo, hi = _window()
    return {"result": _jsonable(result), "index": i,
            "overlay": _overlay_url(vol.data[i], lo, hi, cracks=labels[i],
                                    crack_color=ws.pore_color or _PORE_TINT_DEFAULT),
            "color": _color_hex(ws.pore_color, _PORE_TINT_DEFAULT),
            "summary": ws.summary()}


@app.delete("/api/pores")
def pores_clear_endpoint() -> dict[str, Any]:
    ws.pore_labels = None
    ws.pore_result = None
    return {"ok": True, "summary": ws.summary()}


@app.post("/api/rings")
def rings_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    vol = _volume()
    out, info = filters.remove_rings_volume(vol.data, body or {})
    _commit(out, "ring artifact removal")
    return {"info": _jsonable(info), "summary": ws.summary()}


@app.post("/api/denoise")
def denoise_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    vol = _volume()
    out, info = filters.denoise_volume(vol.data, body.get("method", "gaussian"),
                                       body.get("params") or {})
    _commit(out, f"denoise ({body.get('method', 'gaussian')})")
    return {"info": _jsonable(info), "summary": ws.summary()}


@app.post("/api/calibrate")
def calibrate_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    vol = _volume()
    try:
        out, info = calibration.calibrate(
            vol.data, standards=body.get("standards"),
            target=body.get("target", "density"), degree=int(body.get("degree", 1)),
            unit=vol.unit, mask=ws.effective_mask())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _commit(out, f"calibration ({info['unit_to']})")
    vol.unit = info["unit_to"]
    vol.metadata["calibration"] = info
    return {"info": info, "summary": ws.summary()}


# --------------------------------------------------------------- pipeline
@app.get("/api/pipeline/stages")
def pipeline_stages() -> dict[str, Any]:
    return {"stages": stage_specs(), "plan": APP["plan"], "default_plan": default_plan()}


@app.post("/api/pipeline/plan")
def pipeline_plan(body: dict[str, Any]) -> dict[str, Any]:
    plan = body.get("plan")
    if not isinstance(plan, list):
        raise HTTPException(400, "plan must be a list of {id, enabled, params}")
    APP["plan"] = plan
    return {"ok": True, "plan": APP["plan"]}


@app.post("/api/pipeline/run")
def pipeline_run(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    if APP["running"]:
        raise HTTPException(409, "pipeline already running")
    _volume()
    plan = body.get("plan") or APP["plan"]
    APP["plan"] = plan
    APP["running"] = True
    APP["error"] = None
    APP["results"] = []
    APP["progress"] = {"state": "running", "i": 0, "n": sum(
        1 for p in plan if p.get("enabled", True)), "stage": None, "status": "starting"}
    thread = threading.Thread(target=_run_pipeline, args=(plan,), daemon=True)
    thread.start()
    return {"started": True, "plan": plan}


def _run_pipeline(plan: list[dict[str, Any]]) -> None:
    t0 = time.time()
    try:
        vol = _volume()
        ws.push_history("pipeline run")
        ctx = PipelineContext(data=vol.data.copy(), unit=vol.unit, spacing=vol.spacing,
                              mask=ws.effective_mask(), name=vol.name,
                              metadata=dict(vol.metadata))

        def progress(i: int, n: int, stage_id: str, status: str) -> None:
            APP["progress"] = {"state": "running" if status == "running" else "finishing",
                               "i": i, "n": n, "stage": stage_id, "status": status}

        res = run_pipeline(ctx, plan, progress=progress)
        # commit results
        ws.replace_data(np.asarray(ctx.data, dtype=np.float32))
        ws.segmentation = ctx.segmentation
        ws.transforms = dict(ctx.transforms)
        ws.pipeline_log = res["results"]
        ws.notes.append({"kind": "pipeline", "seconds": res["seconds"], "ok": res["ok"]})
        APP["report"] = ctx.report
        APP["reconstruction"] = ctx.reconstruction
        APP["exports"] = ctx.exports
        APP["results"] = res["results"]
        APP["progress"] = {"state": "done", "i": len(res["results"]),
                           "n": len(res["results"]), "stage": None,
                           "status": "done" if res["ok"] else "error",
                           "seconds": res["seconds"]}
        _bump()
    except Exception as exc:  # noqa: BLE001
        APP["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        APP["progress"] = {"state": "error", "i": 0, "n": 0, "stage": None, "status": "error"}
    finally:
        APP["running"] = False
        APP.setdefault("wall_seconds", round(time.time() - t0, 3))


@app.get("/api/pipeline/progress")
def pipeline_progress() -> dict[str, Any]:
    return {"running": APP["running"], "progress": APP["progress"],
            "results": APP["results"], "error": APP["error"],
            "report": APP.get("report") or None,
            "reconstruction": {k: v for k, v in (APP.get("reconstruction") or {}).items()
                               if k != "brick"},
            "exports": {k: (len(v) if isinstance(v, (bytes, bytearray)) else v)
                        for k, v in (APP.get("exports") or {}).items()}}


# --------------------------------------------------------------- analysis / export
@app.get("/api/analysis")
def analysis_endpoint(bins: int = 256) -> dict[str, Any]:
    vol = _volume()
    rep = analysis_mod.analyze(vol.data, spacing=vol.spacing, mask=ws.effective_mask(),
                               unit=vol.unit, segmentation=ws.segmentation, bins=bins)
    APP["report"] = rep
    mask = ws.effective_mask()
    radial = None
    mid = mask[mask.shape[0] // 2] if (mask is not None and mask.ndim == 3) else mask
    if mid is None or not mid.any():
        # no ROI mask yet -> detect the specimen boundary on the middle slice
        try:
            mid, _ = roi_mod.auto_boundary(vol.data[vol.shape[0] // 2])
        except Exception:
            mid = None
    if mid is not None and np.any(mid):
        radial = analysis_mod.radial_density_profile(vol.data[vol.shape[0] // 2], mid)
    # report["classes"] is class_stats enriched with mean_value/std_value
    classes = rep.get("classes")
    if classes is None and ws.segmentation is not None:
        classes = segment.class_stats(ws.segmentation, vol.voxel_volume_mm3)
    return {"report": rep, "radial": radial, "unit": vol.unit,
            "segmentation_classes": classes}


@app.get("/api/report/plate.png")
def report_plate(depth_levels: int = 20, bins: int = 120, grid: int = 18,
                 cols: int = 3) -> Response:
    """Core-CT report plate (PNG): depth traces, histogram map, slabs, slice grid."""
    vol = _volume()
    lo, hi = _window()
    png = report_mod.render_plate(vol.data, spacing=vol.spacing, unit=vol.unit,
                                  name=vol.name, lo=lo, hi=hi,
                                  depth_levels=depth_levels, bins=bins, grid=grid,
                                  cols=cols)
    return Response(png, media_type="image/png")


@app.get("/api/export/{fmt}")
def export_endpoint(fmt: str, dtype: str = "float32",
                    lo: float | None = None, hi: float | None = None) -> Response:
    vol = _volume()
    exports = APP.get("exports") or {}
    name = vol.name or "volume"
    if fmt in exports and isinstance(exports[fmt], (bytes, bytearray)) and lo is None and hi is None:
        payload = bytes(exports[fmt])
    elif fmt == "tiff":
        payload = exporters.export_tiff_stack(vol, (lo, hi) if lo is not None and hi is not None else None)
    elif fmt in ("png", "png_zip"):
        payload = exporters.export_png_zip(vol, (lo, hi) if lo is not None and hi is not None else None)
    elif fmt == "raw":
        payload = exporters.export_raw(vol, dtype)
    elif fmt == "nrrd":
        payload = exporters.export_nrrd(vol, dtype if dtype in ("float32", "int16") else "float32")
    elif fmt in ("csv", "stats"):
        report = APP.get("report") or analysis_mod.analyze(vol.data, spacing=vol.spacing,
                                                           mask=ws.effective_mask(),
                                                           unit=vol.unit)
        payload = exporters.export_stats_csv(report).encode()
    elif fmt == "json":
        payload = exporters.export_json({"summary": ws.summary(), "report": APP.get("report"),
                                         "history": [h.label for h in ws.history]}).encode()
    else:
        raise HTTPException(404, f"unknown export format '{fmt}'")
    ext = {"tiff": "tif", "png": "zip", "png_zip": "zip", "raw": "raw",
           "nrrd": "nrrd", "csv": "csv", "stats": "csv", "json": "json"}.get(fmt, "bin")
    media = {"tiff": "image/tiff", "png": "application/zip", "png_zip": "application/zip",
             "nrrd": "application/octet-stream", "csv": "text/csv", "stats": "text/csv",
             "json": "application/json"}.get(fmt, "application/octet-stream")
    return Response(payload, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{name}.{ext}"',
        "Content-Length": str(len(payload)),
    })


# --------------------------------------------------------------- static UI
# User manual (manual.html + screenshots/ + printable PDF) for the in-app Help tab.
if DOCS_DIR.is_dir():
    app.mount("/manual", StaticFiles(directory=str(DOCS_DIR), html=True), name="manual")
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
