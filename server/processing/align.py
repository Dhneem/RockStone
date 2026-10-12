"""Slice alignment: translation / rotation / optional scaling.

Estimation methods (all subpixel):

* ``phase``        - FFT phase correlation with upsampled DFT refinement
* ``cross``        - zero-mean normalised cross-correlation in the Fourier domain
* ``feature``      - Shi-Tomasi corners + patch descriptors + rigid Procrustes fit
* ``mutualinfo``   - Powell optimisation of the mutual information

Every estimator returns the transform that must be *applied to the moving
slice* to register it onto the reference: content moves by (dx, dy) px and
rotates by `rotation` degrees counter-clockwise about the slice centre.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import ndimage, optimize


# ------------------------------------------------------------------ utils
def _prep(img: np.ndarray, smooth: float = 1.0) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    a = np.nan_to_num(a, nan=float(np.nanmean(a)) if np.isfinite(a).any() else 0.0)
    a = a - float(a.mean())
    if smooth > 0:
        a = ndimage.gaussian_filter(a, smooth)
    return a


def _window(shape: tuple[int, int]) -> np.ndarray:
    wy = np.hanning(shape[0])[:, None]
    wx = np.hanning(shape[1])[None, :]
    return wy * wx


def _parabolic_subpixel(dx: float, dy: float, peak_y: int, peak_x: int, m: np.ndarray,
                        w: int, h: int) -> tuple[float, float]:
    """Quadratic refinement of a correlation peak (wrapped neighbours)."""
    def axis(ym1: float, y0: float, yp1: float) -> float:
        denom = (ym1 - 2 * y0 + yp1)
        return 0.0 if abs(denom) < 1e-12 else 0.5 * (ym1 - yp1) / denom
    cx = axis(m[peak_y, (peak_x - 1) % w], m[peak_y, peak_x], m[peak_y, (peak_x + 1) % w])
    cy = axis(m[(peak_y - 1) % h, peak_x], m[peak_y, peak_x], m[(peak_y + 1) % h, peak_x])
    return dx + float(cx), dy + float(cy)


def _upsampled_peak(cross: np.ndarray, peak: tuple[int, int], upsample: int,
                    full_shape: tuple[int, int]) -> tuple[float, float]:
    """Subpixel refinement by evaluating the IDFT on a small grid around the peak.

    ``cross`` is rfft2-shaped; it is hermitian-completed along x first so the
    evaluated double sum equals the real correlation surface, then sampled on a
    (3 x 3)-cell grid of step 1/upsample around the peak.  Results are wrapped
    into the signed shift range like the integer path does.
    """
    H, W = int(full_shape[0]), int(full_shape[1])
    py, px = peak
    if upsample <= 1:
        return float(px - (W if px > W // 2 else 0)), float(py - (H if py > H // 2 else 0))
    # hermitian completion along the rfft axis (exactly what irfft2 assumes)
    wl = cross.shape[1]
    F = np.zeros((cross.shape[0], W), dtype=complex)
    F[:, :wl] = cross
    F[:, wl:] = np.conj(cross[:, wl - 2:0:-1]) if W % 2 == 0 else np.conj(cross[:, wl - 1:0:-1])

    y0, x0 = py - 1, px - 1
    ys = y0 + np.arange(upsample * 3) / upsample
    xs = x0 + np.arange(upsample * 3) / upsample
    jy = np.fft.fftfreq(H) * H
    ix = np.arange(W)
    wy = np.exp(2j * np.pi * np.outer(ys, jy) / H)   # (ny, H)
    wx = np.exp(2j * np.pi * np.outer(xs, ix) / W)   # (nx, W)
    vals = np.real(wy @ (F @ wx.T))                  # (ny, nx)
    idx = np.unravel_index(int(np.argmax(vals)), vals.shape)
    sub_y = float(ys[idx[0]])
    sub_x = float(xs[idx[1]])
    if sub_y > H / 2:
        sub_y -= H
    if sub_x > W / 2:
        sub_x -= W
    return sub_x, sub_y


def _ramp(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    fy = (np.arange(h) - h * (np.arange(h) >= h // 2)) / h
    fx = (np.arange(w) - w * (np.arange(w) >= w // 2)) / w
    return np.meshgrid(fy, fx, indexing="ij")


# -------------------------------------------------------------- estimators
def phase_correlation(ref: np.ndarray, cur: np.ndarray, upsample: int = 10,
                      smooth: float = 1.0) -> tuple[float, float, float]:
    """Returns (dx, dy, peak_height) needed to move `cur` onto `ref`."""
    a = _prep(ref, smooth)
    b = _prep(cur, smooth)
    win = _window(a.shape)
    fa = np.fft.rfft2(a * win)
    fb = np.fft.rfft2(b * win)
    r = fa * np.conj(fb)
    mag = np.abs(r)
    r = r / np.maximum(mag, 1e-12)
    corr = np.fft.irfft2(r, s=a.shape)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    py, px = int(peak[0]), int(peak[1])
    dx, dy = _upsampled_peak(r, (py, px), upsample, full_shape=a.shape)
    return float(dx), float(dy), float(corr.max())


def cross_correlation(ref: np.ndarray, cur: np.ndarray, smooth: float = 1.0) -> tuple[float, float, float]:
    """Zero-mean normalised cross-correlation via FFT, subpixel by parabola."""
    a = _prep(ref, smooth)
    b = _prep(cur, smooth)
    win = _window(a.shape)
    fa = np.fft.rfft2(a * win)
    fb = np.fft.rfft2(b * win)
    r = fa * np.conj(fb)
    corr = np.fft.irfft2(r, s=a.shape)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    py, px = int(peak[0]), int(peak[1])
    dx = px - w if px > w // 2 else px
    dy = py - h if py > h // 2 else py
    dx, dy = _parabolic_subpixel(dx, dy, py, px, corr, w, h)
    return float(dx), float(dy), float(corr.max())


def _corners(img: np.ndarray, n: int = 300, min_dist: int = 6, border: int = 12,
             frac: float = 0.02) -> np.ndarray:
    """Shi-Tomasi minimum-eigenvalue corners with non-maximum suppression."""
    g = _prep(img, 1.0)
    gy, gx = np.gradient(g)
    gxx = ndimage.gaussian_filter(gx * gx, 2.0)
    gyy = ndimage.gaussian_filter(gy * gy, 2.0)
    gxy = ndimage.gaussian_filter(gx * gy, 2.0)
    trace = gxx + gyy
    det = gxx * gyy - gxy * gxy
    disc = np.maximum(trace * trace - 4.0 * det, 0.0)
    lam_min = 0.5 * (trace - np.sqrt(disc))
    response = -lam_min
    thr = float(np.percentile(response, 100.0 * (1.0 - frac)))
    mx = ndimage.maximum_filter(response, size=min_dist)
    mask = (response >= mx) & (response > thr)
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((0, 2))
    order = np.argsort(response[ys, xs])[::-1][:n]
    return np.stack([xs[order], ys[order]], axis=1)  # (x, y)


def _descriptors(img: np.ndarray, pts: np.ndarray, half: int = 3) -> np.ndarray:
    """Zero-mean intensity patch descriptors around each point."""
    p = _prep(img, 0.7)
    h, w = p.shape
    out = np.zeros((pts.shape[0], (2 * half + 1) ** 2))
    for i, (x, y) in enumerate(pts):
        x0, y0 = int(x) - half, int(y) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > w or y0 + 2 * half + 1 > h:
            continue
        patch = p[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1].ravel()
        out[i] = patch - patch.mean()
    return out


def _match(ref_pts: np.ndarray, ref_desc: np.ndarray, cur_pts: np.ndarray, cur_desc: np.ndarray,
           ratio: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour matching with a Lowe ratio test."""
    if ref_pts.size == 0 or cur_pts.size == 0:
        return np.zeros((0, 2), int), np.zeros((0, 2), int)
    d = np.linalg.norm(ref_desc[:, None, :] - cur_desc[None, :, :], axis=2)
    best = np.argmin(d, axis=1)
    nn = d[np.arange(d.shape[0]), best]
    second = np.partition(d, 1, axis=1)[:, 1]
    ok = nn < ratio * np.maximum(second, 1e-9)
    # mutual check
    back = np.argmin(d, axis=0)
    mutual = back[best] == np.arange(d.shape[0])
    ok &= mutual
    a = ref_pts[ok]
    b = cur_pts[best[ok]]
    return a, b


def _rigid(p_cur: np.ndarray, p_ref: np.ndarray,
           center: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """Least-squares rigid transform mapping cur -> ref, in apply_transform terms.

    Two convention details matter:

    * the translation is expressed about ``center`` (the rotation centre used
      by :func:`apply_transform`), not about the coordinate origin - otherwise
      the rotation leaks a large spurious offset into dx/dy;
    * ``apply_transform`` uses the opposite rotation sign to the standard
      rotation matrix, so the fitted angle is negated before returning.
    """
    mu_c = p_cur.mean(axis=0)
    mu_r = p_ref.mean(axis=0)
    cc = p_cur - mu_c
    cr = p_ref - mu_r
    cov = cc.T @ cr / max(len(p_cur), 1)
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    angle = -float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    t = mu_r - rot @ mu_c
    if center is not None:
        c = np.asarray(center, dtype=float)
        t = t - c + rot @ c  # re-reference the offset to the image centre
    return float(t[0]), float(t[1]), angle


def feature_correlation(ref: np.ndarray, cur: np.ndarray, params: dict[str, Any] | None = None) -> tuple[float, float, float, dict]:
    params = dict(params or {})
    n = int(params.get("features", 300))
    ref_pts = _corners(ref, n=n)
    cur_pts = _corners(cur, n=n)
    ref_desc = _descriptors(ref, ref_pts)
    cur_desc = _descriptors(cur, cur_pts)
    a, b = _match(ref_pts, ref_desc, cur_pts, cur_desc, float(params.get("ratio", 0.85)))
    if len(a) < 6:
        return 0.0, 0.0, 0.0, {"matches": int(len(a)), "error": "too few matches"}
    # reject gross outliers by distance consistency (median shift)
    shift = np.median(b - a, axis=0)
    keep = np.linalg.norm(b - a - shift, axis=1) < max(4.0, 3.0 * np.std(b - a - a * 0))
    if keep.sum() >= 6:
        a, b = a[keep], b[keep]
    h, w = ref.shape
    dx, dy, rot = _rigid(b, a, center=((h - 1) / 2.0, (w - 1) / 2.0))  # cur points -> ref points
    return float(dx), float(dy), float(rot), {"matches": int(len(a)), "features_ref": int(len(ref_pts)),
                                              "features_cur": int(len(cur_pts))}


def mutual_information(ref: np.ndarray, cur: np.ndarray, params: dict[str, Any] | None = None) -> tuple[float, float, float, dict]:
    params = dict(params or {})
    bins = int(params.get("bins", 48))
    max_shift = float(params.get("max_shift", 12.0))
    max_rot = float(params.get("max_rot", 3.0))
    scale = int(params.get("downsample", 2))
    a = ref if scale <= 1 else ref[::scale, ::scale]
    b = cur if scale <= 1 else cur[::scale, ::scale]
    a = _prep(a, 1.0)
    b = _prep(b, 1.0)
    a_q = np.clip(((a - a.min()) / (np.ptp(a) + 1e-9) * (bins - 1)).astype(int), 0, bins - 1)
    a_q = a_q[3:-3, 3:-3]

    def cost(p: np.ndarray) -> float:
        # p is expressed in *working* (downsampled) pixels: the image fed to
        # apply_transform is already downsampled, so no extra scaling here
        dx, dy, rot = float(p[0]), float(p[1]), float(p[2])
        moved = apply_transform(b, dx, dy, rot, 1.0, order=1, cval=float(np.median(b)))
        m = _prep(moved, 1.0)
        b_q = np.clip(((m - m.min()) / (np.ptp(m) + 1e-9) * (bins - 1)).astype(int), 0, bins - 1)
        c = b_q[3:-3, 3:-3]
        if c.shape != a_q.shape:
            n = min(c.shape[0], a_q.shape[0]), min(c.shape[1], a_q.shape[1])
            c = c[:n[0], :n[1]]
            aa = a_q[:n[0], :n[1]]
        else:
            aa = a_q
        joint = np.bincount(aa.ravel() * bins + c.ravel(),
                            minlength=bins * bins).astype(float).reshape(bins, bins)
        joint /= max(joint.sum(), 1.0)
        den = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
        mask = joint > 0
        mi = float((joint[mask] * np.log(joint[mask] / np.maximum(den[mask], 1e-12))).sum())
        return -mi

    bounds = [(-max_shift / scale, max_shift / scale)] * 2 + [(-max_rot, max_rot)]
    res = optimize.minimize(cost, x0=np.array([0.0, 0.0, 0.0]), method="Powell", bounds=bounds,
                            options={"maxiter": int(params.get("maxiter", 120)),
                                     "xtol": 0.02, "ftol": 1e-3})
    dx, dy, rot = (float(res.x[0]) * scale, float(res.x[1]) * scale, float(res.x[2]))
    return dx, dy, rot, {"mi": float(-res.fun), "iterations": int(res.nit), "downsample": scale}


# ------------------------------------------------------------- transform
def _ncc_score(ref: np.ndarray, cur: np.ndarray, dx: float, dy: float, rot: float,
               scale: float, margin: int) -> float:
    """Normalised cross-correlation of the interiors after warping ``cur``."""
    moved = apply_transform(cur, dx, dy, rot, scale, cval=float(np.median(cur)))
    h, w = ref.shape
    a = ref[margin:h - margin, margin:w - margin]
    b = moved[margin:h - margin, margin:w - margin]
    a_n = (a - a.mean()) / (a.std() + 1e-9)
    b_n = (b - b.mean()) / (b.std() + 1e-9)
    return float((a_n * b_n).mean())


def _refine_ncc(ref: np.ndarray, cur: np.ndarray, dx: float, dy: float, rot: float,
                scale: float = 1.0, span_px: float = 1.5, span_rot: float = 1.0,
                passes: int = 3) -> tuple[float, float, float]:
    """Polish a coarse estimate by coordinate ascent on the interior NCC.

    The FFT estimators give a peak that can be off by several tenths of a pixel
    on smooth, band-limited slices; a small bounded search around the coarse
    answer (steps halving each pass) locks onto the true NCC maximum cheaply.
    """
    h, w = ref.shape
    margin = max(6, int(0.1 * min(h, w)))
    best = (float(dx), float(dy), float(rot))
    best_v = _ncc_score(ref, cur, *best, scale, margin)
    for k in range(passes):
        step_px = span_px / (2 ** k)
        step_rot = span_rot / (2 ** k)
        for axis, step in ((0, step_px), (1, step_px), (2, step_rot)):
            for delta in (+step, -step):
                cand = list(best)
                cand[axis] += delta
                v = _ncc_score(ref, cur, cand[0], cand[1], cand[2], scale, margin)
                if v > best_v:
                    best_v, best = v, (cand[0], cand[1], cand[2])
    return best


def apply_transform(img: np.ndarray, dx: float, dy: float, rot_deg: float = 0.0,
                    scale: float = 1.0, order: int = 1,
                    cval: float | None = None) -> np.ndarray:
    """Move the content of `img` by (dx, dy) px, rotate by rot_deg (CCW),
    optionally scale about the image centre."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.radians(rot_deg)
    s = float(scale) if scale else 1.0
    rot_inv = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    matrix = rot_inv / max(s, 1e-6)
    offset = np.array([cy, cx]) - matrix @ np.array([cy + dy, cx + dx])
    if cval is None:
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        cval = float(np.median(border))
    return ndimage.affine_transform(a, matrix=matrix, offset=offset, order=order,
                                    mode="constant", cval=cval).astype(np.float32)


def estimate(ref: np.ndarray, cur: np.ndarray, method: str = "phase",
             params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Estimate the transform that aligns `cur` onto `ref`."""
    params = dict(params or {})
    t0 = time.time()
    ref = np.asarray(ref, dtype=np.float32)
    cur = np.asarray(cur, dtype=np.float32)
    if ref.shape != cur.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {cur.shape}")

    extra: dict[str, Any] = {}
    if method == "phase":
        dx, dy, score = phase_correlation(ref, cur, int(params.get("upsample", 10)),
                                          float(params.get("smooth", 1.0)))
        rot = float(params.get("forced_rotation", 0.0))
        extra = {"peak": score}
    elif method == "cross":
        dx, dy, score = cross_correlation(ref, cur, float(params.get("smooth", 1.0)))
        rot = float(params.get("forced_rotation", 0.0))
        extra = {"peak": score}
    elif method == "feature":
        dx, dy, rot, extra = feature_correlation(ref, cur, params)
    elif method == "mutualinfo":
        dx, dy, rot, extra = mutual_information(ref, cur, params)
    else:
        raise ValueError(f"unknown alignment method '{method}'")

    # scale: estimate only when explicitly requested (disabled for quantitative work)
    scale = 1.0
    if bool(params.get("allow_scale", False)) and method in ("mutualinfo", "feature"):
        scale = float(params.get("forced_scale", 1.0))

    # polish the coarse estimate on the actual NCC surface (cheap, bounded)
    if bool(params.get("refine", True)):
        dx, dy, rot = _refine_ncc(ref, cur, dx, dy, rot, scale)

    # residual quality on the interior: border fill from the warp must not
    # dominate the score, only genuine misalignment should
    moved = apply_transform(cur, dx, dy, rot, scale, cval=float(np.median(cur)))
    h, w = ref.shape
    corner = np.radians(abs(rot)) * max(h, w) / 2.0
    margin = int(np.ceil(max(abs(dx), abs(dy)) + corner)) + 4
    if 2 * margin < min(h, w) // 2:
        sl = (slice(margin, h - margin), slice(margin, w - margin))
        a, b = ref[sl], moved[sl]
    else:
        a, b = ref.ravel(), moved.ravel()
    a_n = (a - a.mean()) / (a.std() + 1e-9)
    b_n = (b - b.mean()) / (b.std() + 1e-9)
    ncc = float((a_n * b_n).mean())
    before = float(np.corrcoef(ref.ravel() - ref.mean(), cur.ravel() - cur.mean())[0, 1])

    return {
        "dx": float(dx),
        "dy": float(dy),
        "rotation": float(rot),
        "scale": float(scale),
        "ncc_before": round(before, 5),
        "ncc_after": round(ncc, 5),
        "method": method,
        "seconds": round(time.time() - t0, 3),
        **extra,
    }


def align_volume(data: np.ndarray, ref_index: int | None = None, method: str = "phase",
                 params: dict[str, Any] | None = None, scope: str = "all",
                 manual: dict[str, float] | None = None,
                 sequential: bool = False) -> tuple[np.ndarray, dict[str, Any], dict[int, dict[str, float]]]:
    """Register slices to a reference slice.

    scope: 'all' | 'range' (params.start..params.stop) | 'single'
    manual: explicit {dx, dy, rotation, scale} applied instead of estimating
    sequential: register each slice to the previous one and accumulate
    """
    params = dict(params or {})
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    z = arr.shape[0]
    ref_index = z // 2 if ref_index is None else int(ref_index)
    ref_index = int(np.clip(ref_index, 0, z - 1))

    start, stop = 0, z
    if scope == "range":
        start = int(params.get("start", 0))
        stop = int(params.get("stop", z - 1)) + 1
    elif scope == "single":
        idx = int(params.get("index", ref_index))
        start, stop = idx, idx + 1
    start = max(0, start)
    stop = min(z, stop)

    out = arr.copy()
    transforms: dict[int, dict[str, float]] = {}
    estimates: list[dict[str, Any]] = []
    t0 = time.time()
    for i in range(start, stop):
        if i == ref_index and manual is None and scope != "single":
            continue
        if manual is not None:
            tr = {k: float(v) for k, v in manual.items()}
            est = {"dx": tr.get("dx", 0.0), "dy": tr.get("dy", 0.0),
                   "rotation": tr.get("rotation", 0.0), "scale": tr.get("scale", 1.0),
                   "method": "manual"}
        elif sequential and i > 0:
            est = estimate(out[i - 1], out[i], method, params)
            prev = transforms.get(i - 1, {"dx": 0.0, "dy": 0.0, "rotation": 0.0, "scale": 1.0})
            # accumulate (rotation/translation composition, small-angle safe)
            est = {"dx": est["dx"] + prev.get("dx", 0.0),
                   "dy": est["dy"] + prev.get("dy", 0.0),
                   "rotation": est["rotation"] + prev.get("rotation", 0.0),
                   "scale": est["scale"] * prev.get("scale", 1.0),
                   "method": method, "sequential": True}
        else:
            est = estimate(arr[ref_index], arr[i], method, params)
        out[i] = apply_transform(arr[i], est["dx"], est["dy"], est.get("rotation", 0.0),
                                 est.get("scale", 1.0), cval=float(np.median(arr[i])))
        transforms[i] = {"dx": float(est["dx"]), "dy": float(est["dy"]),
                         "rotation": float(est.get("rotation", 0.0)),
                         "scale": float(est.get("scale", 1.0))}
        estimates.append({"index": i, **{k: est[k] for k in ("dx", "dy", "rotation") if k in est}})

    info = {
        "method": "manual" if manual else method,
        "reference": ref_index,
        "scope": scope,
        "slices_transformed": len(transforms),
        "seconds": round(time.time() - t0, 3),
        "mean_dx": float(np.mean([t["dx"] for t in transforms.values()])) if transforms else 0.0,
        "mean_dy": float(np.mean([t["dy"] for t in transforms.values()])) if transforms else 0.0,
        "mean_rotation": float(np.mean([t["rotation"] for t in transforms.values()])) if transforms else 0.0,
        "max_abs_dx": float(max((abs(t["dx"]) for t in transforms.values()), default=0.0)),
        "max_abs_dy": float(max((abs(t["dy"]) for t in transforms.values()), default=0.0)),
        "estimates": estimates[:64],
    }
    return out, info, transforms
