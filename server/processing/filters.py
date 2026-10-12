"""Ring-artifact removal and noise reduction.

Ring artifacts come from individual detector elements with a gain error: after
reconstruction they appear as circles concentric with the rotation centre, i.e.
as a *theta-independent* stripe in polar coordinates.  Removing them is then a
1-D outlier problem along the radius, which is what :func:`remove_rings` does.

Noise reduction offers gaussian / median / bilateral filters, all slice-wise.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import ndimage


def auto_center(slice2d: np.ndarray, search: int = 4, step: int = 8) -> tuple[float, float]:
    """Find the rotation centre by maximising the sharpness of the radial profile."""
    h, w = slice2d.shape
    base_cx, base_cy = w / 2.0, h / 2.0
    best, best_score = (base_cx, base_cy), -1.0
    for dy in range(-search, search + 1, max(1, step // 8)):
        for dx in range(-search, search + 1, max(1, step // 8)):
            cx, cy = base_cx + dx * step / 2.0, base_cy + dy * step / 2.0
            prof = _theta_median_profile(slice2d, cx, cy)
            if prof is None or prof.size < 16:
                continue
            hp = prof - ndimage.median_filter(prof, size=max(5, prof.size // 8))
            score = float(np.std(hp))
            if score > best_score:
                best_score, best = score, (cx, cy)
    return best


def _polar(slice2d: np.ndarray, cx: float, cy: float, n_theta: int = 180) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample to polar coordinates.  Returns (image[n_r, n_theta], (r_max, cx, cy))."""
    h, w = slice2d.shape
    r_max = float(min(cx, cy, w - cx, h - cy))
    n_r = int(max(16, round(r_max)))
    r = np.linspace(0, r_max, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    yy = cy + rr * np.sin(tt)
    xx = cx + rr * np.cos(tt)
    coords = np.array([yy, xx])
    polar = ndimage.map_coordinates(slice2d, coords, order=1, mode="nearest")
    return polar, (r_max, cx, cy)


def _theta_median_profile(slice2d: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    polar, _ = _polar(slice2d, cx, cy, n_theta=90)
    if polar.size == 0 or polar.shape[0] < 16:
        return None
    return np.median(polar, axis=1)


def remove_rings(slice2d: np.ndarray, params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract the theta-independent radial component (rings) from one slice.

    params: center (x,y) | 'auto', ring_window (px along radius), strength,
    theta_outlier (k) for additionally removing theta-localized spokes.
    """
    params = dict(params or {})
    data = np.asarray(slice2d, dtype=np.float32)
    center = params.get("center", "auto")
    if center in ("auto", None, [None, None]):
        cx, cy = auto_center(data)
        center_src = "auto"
    else:
        cx, cy = float(center[0]), float(center[1])
        center_src = "given"
    window = int(params.get("ring_window", 9))
    strength = float(np.clip(params.get("strength", 1.0), 0.0, 2.0))

    polar, (r_max, cx, cy) = _polar(data, cx, cy, n_theta=int(params.get("n_theta", 180)))
    if polar.shape[0] < window + 2:
        return data.copy(), {"skipped": True, "reason": "image too small for polar transform"}
    theta_med = np.median(polar, axis=1)
    smooth = ndimage.median_filter(theta_med, size=min(window if window % 2 else window + 1,
                                                       max(3, theta_med.size // 4 | 1)))
    ring_profile = theta_med - smooth
    ring_profile = ring_profile - ring_profile.mean()
    # theta outliers (spokes / partial rings)
    k = float(params.get("theta_outlier", 0.0))
    if k > 0:
        dev = polar - theta_med[:, None]
        std = np.std(dev, axis=1, keepdims=True) + 1e-9
        polar = np.where(np.abs(dev) > k * std, theta_med[:, None] - strength * 0.5 * dev, polar)

    n_theta = polar.shape[1]
    corrected_polar = polar - strength * ring_profile[:, None]
    # resample back: reconstruct with a simple inverse mapping over the grid
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx) % (2 * np.pi)
    ri = np.clip(r / max(r_max, 1e-6) * (polar.shape[0] - 1), 0, polar.shape[0] - 1)
    ti = (th / (2 * np.pi) * n_theta) % n_theta
    corr_field = ndimage.map_coordinates(corrected_polar, np.array([ri, ti]), order=1, mode="nearest")
    orig_field = ndimage.map_coordinates(polar, np.array([ri, ti]), order=1, mode="nearest")
    delta = data - orig_field
    out = (corr_field + delta).astype(np.float32)
    return out, {
        "center": [float(cx), float(cy)],
        "center_source": center_src,
        "ring_window": window,
        "strength": strength,
        "ring_rms": float(np.std(ring_profile)),
        "n_rings_detected": int(np.sum(np.abs(ring_profile) > 2.0 * (np.std(ring_profile) + 1e-9))),
    }


def remove_rings_volume(data: np.ndarray, params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Ring removal over a whole stack (shared centre, per-slice profiles)."""
    params = dict(params or {})
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return remove_rings(arr, params)
    center = params.get("center", "auto")
    if center in ("auto", None):
        cx, cy = auto_center(arr[arr.shape[0] // 2])
        params = {**params, "center": [float(cx), float(cy)]}
    out = np.empty_like(arr)
    info: dict[str, Any] = {}
    t0 = time.time()
    for i in range(arr.shape[0]):
        res, info = remove_rings(arr[i], params)
        out[i] = res
    info = dict(info or {})
    info.update({"method": "polar_median", "slices": int(arr.shape[0]),
                 "seconds": round(time.time() - t0, 3), "shared_center": True})
    return out, info


# ---------------------------------------------------------------- denoise
def denoise_slice(slice2d: np.ndarray, method: str = "gaussian",
                  params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    params = dict(params or {})
    data = np.asarray(slice2d, dtype=np.float32)
    if method == "gaussian":
        sigma = float(params.get("sigma", 1.0))
        out = ndimage.gaussian_filter(data, sigma)
        return out, {"method": "gaussian", "sigma": sigma}
    if method == "median":
        size = int(params.get("size", 3))
        out = ndimage.median_filter(data, size=max(3, size | 1))
        return out, {"method": "median", "size": size}
    if method == "bilateral":
        return _bilateral(data, params)
    raise ValueError(f"unknown denoise method '{method}'")


def _bilateral(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Separable-ish bilateral filter: window loop with range weights."""
    sigma_s = float(params.get("sigma_s", 2.0))
    sigma_r = float(params.get("sigma_r", 0.05))
    radius = int(params.get("radius", 3))
    # estimate the data range so sigma_r can be given as a fraction
    vals = data[np.isfinite(data)]
    span = float(np.percentile(vals, 99) - np.percentile(vals, 1)) or 1.0
    sr = sigma_r * span
    pad = np.pad(data, radius, mode="reflect")
    num = np.zeros_like(data)
    den = np.zeros_like(data, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = pad[radius + dy: radius + dy + data.shape[0], radius + dx: radius + dx + data.shape[1]]
            gs = np.exp(-0.5 * (dx * dx + dy * dy) / max(sigma_s ** 2, 1e-6))
            gr = np.exp(-0.5 * ((shifted - data) ** 2) / max(sr * sr, 1e-12))
            w = gs * gr
            num += w * shifted
            den += w
    out = num / np.maximum(den, 1e-12)
    return out.astype(np.float32), {"method": "bilateral", "sigma_s": sigma_s, "sigma_r": sigma_r,
                                    "radius": radius, "range_sigma_units": sr}


def denoise_volume(data: np.ndarray, method: str = "gaussian",
                   params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(data, dtype=np.float32)
    t0 = time.time()
    if arr.ndim == 2:
        out, info = denoise_slice(arr, method, params)
        return out, info
    # 3-D gaussian is both faster and better behaved across slices
    if method == "gaussian":
        params = dict(params or {})
        sz = float(params.get("sigma_z", params.get("sigma", 1.0)))
        out = ndimage.gaussian_filter(arr, sigma=(sz, float(params.get("sigma", 1.0)),
                                                  float(params.get("sigma", 1.0))))
        return out.astype(np.float32), {"method": "gaussian-3d", "sigma": [sz, params.get("sigma", 1.0),
                                                                          params.get("sigma", 1.0)],
                                        "seconds": round(time.time() - t0, 3)}
    out = np.empty_like(arr)
    info: dict[str, Any] = {}
    for i in range(arr.shape[0]):
        out[i], info = denoise_slice(arr[i], method, params)
    info.update({"slices": int(arr.shape[0]), "seconds": round(time.time() - t0, 3)})
    return out, info
