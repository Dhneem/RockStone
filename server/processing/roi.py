"""ROI construction: rectangles, ellipses, freehand polygons, auto boundary.

Masks are (y, x) boolean arrays unless stated otherwise.  Everything here is
pure numpy/scipy so it can be reused by the beam-hardening, calibration and
analysis stages.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from server.processing.segment import fill_holes, largest_component, morph_close, otsu_threshold


def shape_mask(shape: tuple[int, int], spec: dict[str, Any]) -> np.ndarray:
    """Rasterise a shape description into a boolean (y, x) mask.

    spec types:
      rect    {type:'rect', x, y, w, h}                 - top-left corner based
      ellipse {type:'ellipse', cx, cy, rx, ry}
      circle  {type:'circle', cx, cy, r}
      polygon {type:'polygon', points:[[x,y], ...]}     - freehand ROI
      frame   {type:'frame'}                           - whole image
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    t = spec.get("type", "frame")
    if t == "frame":
        return np.ones(shape, dtype=bool)
    if t == "rect":
        x, y = float(spec["x"]), float(spec["y"])
        w_, h_ = float(spec["w"]), float(spec["h"])
        return (xx >= x) & (xx < x + w_) & (yy >= y) & (yy < y + h_)
    if t in ("ellipse", "circle"):
        cx, cy = float(spec["cx"]), float(spec["cy"])
        rx = float(spec.get("rx", spec.get("r", 1.0)))
        ry = float(spec.get("ry", spec.get("r", 1.0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            val = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2
        return np.nan_to_num(val, nan=1e9) <= 1.0
    if t == "polygon":
        pts = spec.get("points") or []
        if len(pts) < 3:
            raise ValueError("polygon needs at least 3 points")
        return polygon_mask(shape, [(float(p[0]), float(p[1])) for p in pts])
    raise ValueError(f"unknown shape type '{t}'")


def polygon_mask(shape: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    """Scanline point-in-polygon fill (freehand ROI)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    n = len(points)
    for row in range(h):
        ys = float(row) + 0.5
        xs: list[float] = []
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            if (y1 <= ys < y2) or (y2 <= ys < y1):
                xs.append(x1 + (ys - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            a = max(0, int(np.floor(xs[i])))
            b = min(w, int(np.ceil(xs[i + 1])))
            if b > a:
                mask[row, a:b] = True
    return mask


def auto_boundary(slice2d: np.ndarray, params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Automatic specimen boundary detection.

    Otsu (or a percentile) threshold -> largest connected component -> fill
    holes -> morphological close.  Works on a single slice; `auto_boundary_3d`
    propagates the result through the stack.
    """
    params = dict(params or {})
    data = np.asarray(slice2d, dtype=np.float32)
    thr_mode = params.get("threshold_mode", "otsu")
    if thr_mode == "percentile":
        finite = data[np.isfinite(data)]
        thr = float(np.percentile(finite, float(params.get("percentile", 60))))
    elif thr_mode == "manual":
        thr = float(params.get("threshold", 0.0))
    else:
        thr = otsu_threshold(data)
    mask = data > thr
    if not mask.any():
        return np.zeros(data.shape, dtype=bool), {"threshold": thr, "empty": True}
    mask = morph_close(mask, int(params.get("close_size", 5)))
    mask = fill_holes(mask)
    if bool(params.get("largest_component", True)):
        mask = largest_component(mask)
    if bool(params.get("open", True)):
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = fill_holes(mask)
        if not mask.any():
            mask = fill_holes(largest_component(data > thr))
    info = {
        "threshold": float(thr),
        "threshold_mode": thr_mode,
        "area_px": int(mask.sum()),
        "area_fraction": float(mask.mean()),
    }
    return mask, info


def auto_boundary_3d(data: np.ndarray, params: dict[str, Any] | None = None,
                     ref_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Specimen mask for the whole stack.

    Uses the middle slice by default, then keeps every slice component that
    overlaps the reference so partially visible slices are not lost.
    """
    z, h, w = data.shape
    if ref_mask is None:
        ref_mask, info = auto_boundary(data[z // 2], params)
    else:
        info = {"threshold": None, "from": "reference"}
    out = np.zeros((z, h, w), dtype=bool)
    thr = info.get("threshold")
    for i in range(z):
        if thr is None:
            m, _ = auto_boundary(data[i], params)
        else:
            m = data[i] > float(thr)
            m = fill_holes(m)
        if ref_mask is not None and m.any():
            overlap = (m & ref_mask).sum()
            if overlap < max(4, 0.02 * ref_mask.sum()):
                # slice barely overlaps the reference: fall back to component overlap
                cand = largest_component(m) if m.any() else m
                if not (cand & ref_mask).any():
                    m = np.zeros_like(m)
                else:
                    m = cand
        out[i] = m
    info["slices_filled"] = int(out.any(axis=(1, 2)).sum())
    return out, info


def remove_air(slice2d: np.ndarray, threshold: float | None = None,
               margin_sigma: float = 2.0) -> np.ndarray:
    """Drop background/air voxels: threshold above (air peak + margin)."""
    data = np.asarray(slice2d, dtype=np.float32)
    if threshold is None:
        # air mode from the border ring, which is virtually always background
        border = np.concatenate([data[0].ravel(), data[-1].ravel(), data[:, 0], data[:, -1]])
        air = float(np.median(border))
        thr = air + margin_sigma * (float(border.std()) + 1e-6) * 3.0
        # if the border is already specimen, fall back to Otsu
        if (data > thr).mean() > 0.98:
            thr = otsu_threshold(data)
    else:
        thr = float(threshold)
    return data > thr


def artifact_mask(slice2d: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Mask voxels affected by severe streak/cupping artifacts.

    Modes:
      residual - |value - heavy-blur| above a sigma threshold (streaks, dark bands)
      low      - value below core_mean * factor (air pockets / shadowed regions)
      gradient - strong local gradient rings
    """
    params = dict(params or {})
    data = np.asarray(slice2d, dtype=np.float32)
    mode = params.get("mode", "residual")
    if mode == "low":
        finite = data[np.isfinite(data)]
        core = float(np.percentile(finite, 70))
        factor = float(params.get("factor", 0.6))
        return data < core * factor
    if mode == "gradient":
        gy, gx = np.gradient(ndimage.gaussian_filter(data, 1.0))
        mag = np.hypot(gx, gy)
        thr = float(np.percentile(mag, float(params.get("percentile", 97.0))))
        return mag > thr
    # residual: deviation from a heavily smoothed version
    sigma = float(params.get("sigma", 6.0))
    smooth = ndimage.gaussian_filter(data, sigma)
    resid = data - smooth
    std = float(np.nanstd(resid)) + 1e-9
    k = float(params.get("sigma_k", 3.0))
    return np.abs(resid) > k * std


def sample_stats(slice2d: np.ndarray, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mean/std/size of the pixels inside an ROI shape."""
    data = np.asarray(slice2d, dtype=np.float32)
    m = np.ones(data.shape, dtype=bool) if spec is None else shape_mask(data.shape, spec)
    vals = data[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    return {"n": int(vals.size), "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max())}


def apply_mask_all_slices(data: np.ndarray, mask2d: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Apply one 2-D mask to every slice of a 3-D volume ("apply to all slices")."""
    out = data.copy()
    m = mask2d.astype(bool)
    out[:, ~m] = fill
    return out
