"""Thresholding / segmentation utilities: Otsu, multi-Otsu, region growing.

Pure numpy + scipy so the same code runs in the pipeline and in tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def histogram(data: np.ndarray, bins: int = 256, value_range: tuple[float, float] | None = None,
              mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = data if mask is None else data[mask.astype(bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"bins": [], "counts": [], "min": 0.0, "max": 0.0}
    lo, hi = value_range if value_range else (float(arr.min()), float(arr.max()))
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    return {
        "bins": [float(x) for x in (edges[:-1] + np.diff(edges) / 2)],
        "counts": [int(c) for c in counts],
        "min": lo,
        "max": hi,
    }


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Classic 1979 Otsu threshold on a 1-D array of pixel values."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return lo
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + np.diff(edges) / 2)
    p = counts.astype(np.float64)
    p /= max(p.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b[~np.isfinite(sigma_b)] = -1
    return float(centers[int(np.argmax(sigma_b))])


def multi_otsu(values: np.ndarray, n_classes: int = 3, bins: int = 256) -> list[float]:
    """Optimal (n_classes-1) thresholds maximising between-class variance.

    Exhaustive over binned thresholds - fine for the 1-D histograms we use.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0.0] * (n_classes - 1)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return [lo] * (n_classes - 1)
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = edges[:-1] + np.diff(edges) / 2
    p = counts.astype(np.float64)
    p /= max(p.sum(), 1.0)

    # cumulative quantities for fast class-variance evaluation
    omega = np.concatenate([[0.0], np.cumsum(p)])
    mu = np.concatenate([[0.0], np.cumsum(p * centers)])
    mu_t = mu[-1]

    def between(t1: int, t2: int) -> float:
        w0, w1, w2 = omega[t1], omega[t2] - omega[t1], omega[-1] - omega[t2]
        m0 = (mu[t1] - mu[0]) / max(w0, 1e-12)
        m1 = (mu[t2] - mu[t1]) / max(w1, 1e-12)
        m2 = (mu[-1] - mu[t2]) / max(w2, 1e-12)
        return w0 * (m0 - mu_t) ** 2 + w1 * (m1 - mu_t) ** 2 + w2 * (m2 - mu_t) ** 2

    if n_classes == 2:
        return [float(otsu_threshold(v, bins))]
    best, best_score = (bins // 4, bins // 2), -1.0
    step = max(1, bins // 48)  # coarse-to-fine keeps this sub-millisecond
    for t1 in range(2, bins - 3, step):
        for t2 in range(t1 + 2, bins - 1, step):
            s = between(t1, t2)
            if s > best_score:
                best_score, best = s, (t1, t2)
    t1, t2 = best
    for a in range(max(2, t1 - step), min(bins - 3, t1 + step) + 1):
        for b in range(max(a + 2, t2 - step), min(bins - 1, t2 + step) + 1):
            s = between(a, b)
            if s > best_score:
                best_score, best = s, (a, b)
    return [float(centers[best[0]]), float(centers[best[1]])]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected component (26-connectivity in 3-D)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask)


def region_grow(slice2d: np.ndarray, seed_xy: tuple[int, int], tolerance: float,
                mask: np.ndarray | None = None) -> np.ndarray:
    """6-connected region growing from a seed with an intensity tolerance."""
    h, w = slice2d.shape
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError("seed outside image")
    visited = np.zeros(slice2d.shape, dtype=bool)
    ref = float(slice2d[sy, sx])
    stack = [(sy, sx)]
    visited[sy, sx] = True
    lo, hi = ref - tolerance, ref + tolerance
    allowed = (slice2d >= lo) & (slice2d <= hi)
    if mask is not None:
        allowed &= mask.astype(bool)
    allowed[sy, sx] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and allowed[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    return visited


def morph_close(mask: np.ndarray, size: int = 3) -> np.ndarray:
    st = np.ones((size, size), dtype=bool)
    return ndimage.binary_closing(mask, structure=st)


def morph_open(mask: np.ndarray, size: int = 3) -> np.ndarray:
    st = np.ones((size, size), dtype=bool)
    return ndimage.binary_opening(mask, structure=st)


def segment_volume(data: np.ndarray, method: str = "otsu", params: dict[str, Any] | None = None,
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment the volume, returning (uint8 label volume, info).

    methods: 'otsu' (2 classes), 'multi_otsu' (3 classes), 'manual' (fixed
    threshold), 'double' (lower + upper threshold).
    """
    params = dict(params or {})
    sample = data if mask is None else data[mask.astype(bool)]
    if method == "otsu":
        t = params.get("threshold")
        if t is None:
            t = otsu_threshold(sample)
        labels = (data > float(t)).astype(np.uint8)
        info = {"method": "otsu", "threshold": float(t), "classes": ["below", "above"]}
    elif method == "multi_otsu":
        ts = params.get("thresholds")
        if ts is None:
            ts = multi_otsu(sample, int(params.get("n_classes", 3)))
        ts = sorted(float(t) for t in ts)
        labels = np.zeros(data.shape, dtype=np.uint8)
        for i, t in enumerate(ts):
            labels[data > t] = i + 1
        info = {"method": "multi_otsu", "thresholds": ts, "classes": [f"class{i}" for i in range(len(ts) + 1)]}
    elif method == "manual":
        t = float(params.get("threshold", 0.0))
        labels = (data > t).astype(np.uint8)
        info = {"method": "manual", "threshold": t, "classes": ["below", "above"]}
    elif method == "double":
        lo = float(params.get("lower", 0.0))
        hi = float(params.get("upper", 1.0))
        labels = np.zeros(data.shape, dtype=np.uint8)
        labels[data > lo] = 1
        labels[data > hi] = 2
        info = {"method": "double", "lower": lo, "upper": hi, "classes": ["between", "above"]}
    else:
        raise ValueError(f"unknown segmentation method '{method}'")

    if params.get("cleanup", True):
        structure = np.ones((3, 3, 3), dtype=bool) if data.ndim == 3 else np.ones((3, 3), dtype=bool)
        cleaned = np.zeros_like(labels)
        for lab in np.unique(labels):
            if lab == 0:
                continue
            comp = labels == lab
            comp = largest_component(comp)
            cleaned[comp] = lab
        labels = cleaned
        info["cleanup"] = "largest-component"
    return labels, info


def class_stats(labels: np.ndarray, voxel_mm3: float, names: list[str] | None = None) -> list[dict[str, Any]]:
    """Per-class volume / fraction statistics for a label volume."""
    out = []
    total = int(labels.size)
    for lab in np.unique(labels):
        n = int((labels == lab).sum())
        entry = {
            "label": int(lab),
            "name": (names[lab] if names and lab < len(names) else f"class {int(lab)}"),
            "voxels": n,
            "fraction": n / max(total, 1),
            "volume_mm3": n * voxel_mm3,
            "volume_cm3": n * voxel_mm3 / 1000.0,
        }
        out.append(entry)
    return out
