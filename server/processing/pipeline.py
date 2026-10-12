"""The CT processing pipeline.

    Import -> Inspect -> Crop -> Beam-Hardening Correction -> Ring Artifact
    Removal -> Noise Reduction -> Slice Alignment -> Calibration ->
    Segmentation -> 3D Reconstruction -> Quantitative Analysis -> Export

Each stage is a dict with an id, a human name, a parameter schema (rendered by
the UI) and a function ``(ctx, params) -> info``.  Stages run in order on a
:class:`PipelineContext`; stages that do not modify voxels (inspect,
segmentation, analysis, export) simply annotate the context.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np

from server.processing import analysis as analysis_mod
from server.processing import beamhardening, calibration, filters, segment
from server.processing.align import align_volume
from server.processing.roi import apply_mask_all_slices, auto_boundary_3d, shape_mask


class PipelineContext:
    """Mutable state threaded through the pipeline stages."""

    def __init__(self, data: np.ndarray, unit: str = "intensity",
                 spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 mask: np.ndarray | None = None, name: str = "volume",
                 metadata: dict[str, Any] | None = None):
        self.data = np.asarray(data, dtype=np.float32)
        self.unit = unit
        self.spacing = spacing
        self.mask = mask  # (y, x) or (z, y, x) bool
        self.name = name
        self.metadata = dict(metadata or {})
        self.segmentation: np.ndarray | None = None
        self.cracks: dict[str, Any] | None = None
        self.transforms: dict[int, dict[str, float]] = {}
        self.report: dict[str, Any] = {}
        self.reconstruction: dict[str, Any] = {}
        self.exports: dict[str, Any] = {}


# --------------------------------------------------------------- helpers
def _mask3d(ctx: PipelineContext) -> np.ndarray | None:
    if ctx.mask is None:
        return None
    m = ctx.mask.astype(bool)
    if m.ndim == 2 and ctx.data.ndim == 3:
        m = np.broadcast_to(m, ctx.data.shape)
    return m


def _fill_value(ctx: PipelineContext, params: dict[str, Any]) -> float:
    fill = params.get("fill", "auto")
    if fill in (None, "auto"):
        border = np.concatenate([ctx.data[..., 0].ravel(), ctx.data[..., -1].ravel(),
                                 ctx.data[:, 0].ravel(), ctx.data[:, -1].ravel()])
        return float(np.median(border[np.isfinite(border)]))
    try:
        return float(fill)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- stages
def stage_inspect(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    data = ctx.data
    stats = {"min": float(np.nanmin(data)), "max": float(np.nanmax(data)),
             "mean": float(np.nanmean(data)), "std": float(np.nanstd(data))}
    m = _mask3d(ctx)
    bh = beamhardening.detect(data, m, ctx.unit, ctx.spacing) if bool(p.get("beam_hardening", True)) else None
    return {"stats": stats, "beam_hardening": bh, "unit": ctx.unit,
            "shape": list(data.shape), "spacing": list(ctx.spacing)}


def stage_crop(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    """ROI crop / mask applied to every slice (tool B of the beam-hardening step)."""
    mode = p.get("mode", "shape")  # 'shape' | 'auto' | 'clear'
    fill = _fill_value(ctx, p)
    if mode == "clear":
        ctx.mask = None
        return {"mode": "clear", "note": "mask removed"}
    if mode == "auto":
        params = {k: p.get(k) for k in ("threshold_mode", "percentile", "threshold", "close_size",
                                        "largest_component", "open") if k in p}
        m3, info = auto_boundary_3d(ctx.data, params)
        ctx.mask = m3  # keep the full 3-D mask: slices differ
        out = ctx.data.copy()
        out[~m3] = fill
        ctx.data = out
        return {"mode": "auto", **info, "fill": fill, "kept_voxels": int(m3.sum()),
                "applied_to": "all slices"}
    spec = p.get("shape") or {"type": "frame"}
    h, w = ctx.data.shape[-2:]
    m2 = shape_mask((h, w), spec)
    ctx.mask = m2
    out = ctx.data.copy()
    if ctx.data.ndim == 3:
        out[:, ~m2] = fill
    else:
        out[~m2] = fill
    ctx.data = out
    return {"mode": "shape", "shape": spec, "area_fraction": float(m2.mean()),
            "fill": fill, "applied_to": "all slices"}


def stage_beam_hardening(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    m = _mask3d(ctx)
    params = dict(p)
    if params.get("method", "thickness") == "spectral":
        # fill unknown acquisition details from the volume metadata so the
        # spectrum/filter model matches how the data were actually acquired
        acq = ctx.metadata.get("acquisition") or {}
        for k in ("kvp", "inherent_al_mm", "added_filter", "added_mm", "material"):
            if params.get(k) in (None, "") and acq.get(k) is not None:
                params[k] = acq[k]
        if params.get("mu_water_reference") is None:
            params["mu_water_reference"] = ctx.metadata.get("mu_water_reference")
    out, info = beamhardening.correct(
        ctx.data,
        method=p.get("method", "thickness"),
        params=params,
        strength=float(p.get("strength", 1.0)),
        unit=ctx.unit,
        mask=m,
        spacing=ctx.spacing,
        mu_water=p.get("mu_water_reference") or ctx.metadata.get("mu_water_reference"),
    )
    ctx.data = out
    return info


def stage_ring_removal(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    out, info = filters.remove_rings_volume(ctx.data, p)
    ctx.data = out
    return info


def stage_denoise(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    method = p.get("method", "gaussian")
    params = {k: v for k, v in p.items() if k not in ("method", "enabled")}
    out, info = filters.denoise_volume(ctx.data, method, params)
    ctx.data = out
    return info


def stage_align(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    out, info, transforms = align_volume(
        ctx.data,
        ref_index=p.get("reference"),
        method=p.get("method", "phase"),
        params=p,
        scope=p.get("scope", "all"),
        manual=p.get("manual"),
        sequential=bool(p.get("sequential", False)),
    )
    ctx.data = out
    ctx.transforms.update(transforms)
    return info


def stage_calibration(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    m = _mask3d(ctx)
    out, info = calibration.calibrate(
        ctx.data,
        standards=p.get("standards"),
        target=p.get("target", "density"),
        degree=int(p.get("degree", 1)),
        unit=ctx.unit,
        mask=m,
    )
    ctx.data = out
    ctx.unit = info["unit_to"]
    return info


def stage_segmentation(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    m = _mask3d(ctx)
    labels, info = segment.segment_volume(ctx.data, p.get("method", "otsu"), p, mask=m)
    ctx.segmentation = labels
    info["classes_stats"] = segment.class_stats(labels, float(np.prod(ctx.spacing)))
    return info


def stage_crack_detection(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    """Non-destructive: measures fractures, stores labels for the overlay."""
    from server.processing.cracks import detect_cracks

    m = _mask3d(ctx)
    labels, result = detect_cracks(
        ctx.data, ctx.spacing, m,
        threshold=p.get("threshold", "auto"),
        min_voxels=int(p.get("min_voxels", 25)),
        min_elongation=float(p.get("min_elongation", 0.55)),
        max_aperture_mm=float(p.get("max_aperture_mm", 4.0) or 0) or None,
    )
    ctx.cracks = {"labels": labels, "result": result}
    return result


def stage_reconstruction(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    """Assemble the aligned stack into the renderable 3-D volume.

    Slices are already in a stack after alignment; this stage validates
    ordering/spacing, optionally smooths the volume and builds the downsampled
    brick used by the WebGL volume viewer.
    """
    from scipy import ndimage

    t0 = time.time()
    data = ctx.data
    max_dim = int(p.get("max_dim", 192))
    smooth = float(p.get("smoothing", 0.0))
    if smooth > 0:
        data = ndimage.gaussian_filter(data, smooth)
        ctx.data = data.astype(np.float32)
    z, y, x = data.shape
    factor = max(1, int(np.ceil(max(max(z, y, x) / max_dim, 1))))
    brick = data[::factor, ::factor, ::factor]
    lo = float(np.nanmin(brick))
    hi = float(np.nanmax(brick))
    q = np.clip((brick - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
    ctx.reconstruction = {
        "brick": q,
        "shape": list(q.shape),
        "source_shape": [z, y, x],
        "factor": factor,
        "value_range": [lo, hi],
        "spacing": [s * factor for s in ctx.spacing],
        "voxel_volume_mm3": float(np.prod(ctx.spacing)),
        "total_volume_cm3": float(z * y * x * np.prod(ctx.spacing) / 1000.0),
        "seconds": round(time.time() - t0, 3),
    }
    return {k: v for k, v in ctx.reconstruction.items() if k != "brick"}


def stage_analysis(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    rep = analysis_mod.analyze(
        ctx.data,
        spacing=ctx.spacing,
        mask=_mask3d(ctx),
        unit=ctx.unit,
        segmentation=ctx.segmentation,
        bins=int(p.get("bins", 256)),
        porosity_threshold=p.get("porosity_threshold"),
    )
    ctx.report = rep
    return rep


def stage_export(ctx: PipelineContext, p: dict[str, Any]) -> dict[str, Any]:
    from server.io import exporters

    fmt = p.get("format", "tiff")
    if fmt == "tiff":
        payload = exporters.export_tiff_stack(_as_volume(ctx))
        ctx.exports["tiff"] = payload
        result: dict[str, Any] = {"format": "tiff", "bytes": len(payload), "ready": True}
    elif fmt == "png_zip":
        payload = exporters.export_png_zip(_as_volume(ctx))
        ctx.exports["png_zip"] = payload
        result = {"format": "png_zip", "bytes": len(payload), "ready": True}
    elif fmt == "raw":
        payload = exporters.export_raw(_as_volume(ctx), p.get("dtype", "float32"))
        ctx.exports["raw"] = payload
        result = {"format": "raw", "bytes": len(payload), "ready": True}
    elif fmt == "nrrd":
        payload = exporters.export_nrrd(_as_volume(ctx))
        ctx.exports["nrrd"] = payload
        result = {"format": "nrrd", "bytes": len(payload), "ready": True}
    else:
        result = {"format": fmt, "note": "format handled by /api/export"}
    if ctx.report:
        csv = exporters.export_stats_csv(ctx.report)
        ctx.exports["stats_csv"] = csv.encode()
        result["stats_csv_bytes"] = len(csv)
    return result


def _as_volume(ctx: PipelineContext):
    from server.store import Volume

    return Volume(data=ctx.data, spacing=ctx.spacing, unit=ctx.unit, name=ctx.name)


# --------------------------------------------------------------- registry
STAGES: list[dict[str, Any]] = [
    {"id": "inspect", "name": "Inspect", "enabled": True, "params": {"beam_hardening": True},
     "fn": stage_inspect, "modifies": False,
     "description": "Dataset statistics + automatic beam-hardening assessment."},
    {"id": "crop", "name": "Crop / ROI", "enabled": True,
     "params": {"mode": "shape", "shape": {"type": "frame"}, "fill": "auto"},
     "fn": stage_crop, "modifies": True,
     "description": "Rectangular / elliptical / freehand ROI, automatic specimen boundary, remove air."},
    {"id": "beam_hardening", "name": "Beam-Hardening Correction", "enabled": True,
     "params": {"method": "thickness", "strength": 1.0, "degree": 2},
     "fn": stage_beam_hardening, "modifies": True,
     "description": "Polynomial / water calibration / thickness / spectral Cu-Al filter correction."},
    {"id": "ring_removal", "name": "Ring Artifact Removal", "enabled": True,
     "params": {"center": "auto", "ring_window": 9, "strength": 1.0},
     "fn": stage_ring_removal, "modifies": True,
     "description": "Polar-domain removal of concentric detector rings."},
    {"id": "denoise", "name": "Noise Reduction", "enabled": True,
     "params": {"method": "gaussian", "sigma": 0.8, "sigma_z": 0.6},
     "fn": stage_denoise, "modifies": True,
     "description": "Gaussian / median / bilateral filtering."},
    {"id": "align", "name": "Slice Alignment", "enabled": True,
     "params": {"method": "phase", "reference": None, "scope": "all", "sequential": False},
     "fn": stage_align, "modifies": True,
     "description": "Phase / cross-correlation / feature / mutual-information registration."},
    {"id": "calibration", "name": "Calibration", "enabled": False,
     "params": {"target": "density", "degree": 1},
     "fn": stage_calibration, "modifies": True,
     "description": "Two-point standards -> density / HU / linear attenuation."},
    {"id": "segmentation", "name": "Segmentation", "enabled": True,
     "params": {"method": "otsu", "cleanup": True},
     "fn": stage_segmentation, "modifies": False,
     "description": "Otsu / multi-Otsu / manual thresholding with component cleanup."},
    {"id": "crack_detection", "name": "Crack / Fracture Detection", "enabled": False,
     "params": {"threshold": "auto", "min_voxels": 25, "min_elongation": 0.55,
                "max_aperture_mm": 4.0},
     "fn": stage_crack_detection, "modifies": False,
     "description": "Dark elongated voids inside the matrix: count, volume, aperture, orientation."},
    {"id": "reconstruction", "name": "3D Reconstruction", "enabled": True,
     "params": {"max_dim": 192, "smoothing": 0.0},
     "fn": stage_reconstruction, "modifies": True,
     "description": "Assemble the aligned stack, smooth, build the renderable volume."},
    {"id": "analysis", "name": "Quantitative Analysis", "enabled": True,
     "params": {"bins": 256},
     "fn": stage_analysis, "modifies": False,
     "description": "Stats, porosity, class volumes, radial profile."},
    {"id": "export", "name": "Export", "enabled": True, "params": {"format": "tiff"},
     "fn": stage_export, "modifies": False,
     "description": "TIFF / PNG / RAW / NRRD volume plus CSV statistics."},
]

STAGE_INDEX = {s["id"]: i for i, s in enumerate(STAGES)}


def stage_specs(include_impl: bool = False) -> list[dict[str, Any]]:
    """Serializable stage list for the UI (parameter schemas + defaults)."""
    out = []
    for s in STAGES:
        item = {k: v for k, v in s.items() if k in ("id", "name", "enabled", "params", "description", "modifies")}
        item["index"] = STAGE_INDEX[s["id"]]
        out.append(item)
    if include_impl:
        for item, s in zip(out, STAGES):
            item["fn"] = s["fn"]
    return out


def default_plan() -> list[dict[str, Any]]:
    return [{"id": s["id"], "enabled": s["enabled"], "params": dict(s["params"])} for s in STAGES]


def run_pipeline(
    ctx: PipelineContext,
    plan: list[dict[str, Any]],
    progress: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Run the enabled stages in the given order.

    Returns {'results': [...], 'seconds': ..., 'ok': bool}.
    """
    by_id = {s["id"]: s for s in STAGES}
    results: list[dict[str, Any]] = []
    t0 = time.time()
    enabled = [p for p in plan if p.get("enabled", True)]
    ok = True
    for i, item in enumerate(enabled):
        spec = by_id.get(item["id"])
        if spec is None:
            results.append({"id": item["id"], "ok": False, "error": "unknown stage"})
            ok = False
            continue
        if progress:
            progress(i, len(enabled), spec["id"], "running")
        st = time.time()
        try:
            params = {**spec["params"], **(item.get("params") or {})}
            info = spec["fn"](ctx, params)
            entry = {"id": spec["id"], "name": spec["name"], "ok": True,
                     "seconds": round(time.time() - st, 3), "info": _jsonable(info)}
        except Exception as exc:  # keep going: one bad stage must not kill the run
            entry = {"id": spec["id"], "name": spec["name"], "ok": False,
                     "seconds": round(time.time() - st, 3),
                     "error": f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc(limit=4)}
            ok = False
        results.append(entry)
        if progress:
            progress(i + 1, len(enabled), spec["id"], "done" if entry["ok"] else "error")
        if not entry["ok"] and bool(item.get("params", {}).get("stop_on_error", False)):
            break
    return {"results": results, "seconds": round(time.time() - t0, 3), "ok": ok,
            "unit": ctx.unit, "shape": list(ctx.data.shape)}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "brick"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 2000:
            return {"shape": list(obj.shape), "dtype": str(obj.dtype), "truncated": True}
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def preview_stage(data: np.ndarray, stage_id: str, params: dict[str, Any],
                  unit: str, spacing: tuple[float, float, float],
                  mask: np.ndarray | None, index: int | None = None,
                  metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one stage against a single slice (fast parameter preview)."""
    by_id = {s["id"]: s for s in STAGES}
    spec = by_id.get(stage_id)
    if spec is None:
        raise KeyError(stage_id)
    idx = data.shape[0] // 2 if index is None else int(index)
    sl = data[idx]
    m2 = None
    if mask is not None:
        m2 = mask[idx] if mask.ndim == 3 else mask
    tmp = PipelineContext(sl, unit=unit, spacing=spacing, mask=m2,
                          metadata=dict(metadata or {}))
    full = {**spec["params"], **(params or {})}
    if stage_id == "align":
        return {"note": "preview not applicable to alignment", "index": idx}
    info = spec["fn"](tmp, full)
    return {"index": idx, "before": sl, "after": tmp.data, "info": _jsonable(info)}
