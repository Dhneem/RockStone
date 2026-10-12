"""Crack / fracture detection in rock-core CT volumes.

Cracks are thin, air-filled voids inside the rock matrix: much darker than the
surrounding material and geometrically elongated or flat, unlike round pores.
The detector:

1. restricts the search to the specimen interior (ROI mask when set, otherwise
   an automatic air/matrix split),
2. picks a dark-voxel threshold inside that interior -- for ``"auto"`` a point
   between the air mode and the matrix median, which stays robust even when
   cracks cover well under 1 % of the volume, or an explicit user value,
3. labels connected dark components (26-connectivity) and discards those that
   are too small or too sphere-like (round pores) to be fractures,
4. measures every survivor: volume, mean aperture (2V/S with the surface
   inferred from 6-neighbour boundary faces) and dominant in-plane orientation
   (PCA of the voxel coordinates),
5. aggregates: totals, crack density, orientation histogram, per-slice area.

Pure numpy + scipy; the same code runs behind the API, in the pipeline stage
and in the unit tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from server.processing.segment import otsu_threshold

_N_ORIENT_BINS = 12  # 15 deg each, 0..180 (cracks are undirected lines)


def specimen_interior(data: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Boolean specimen interior. Falls back to an air/matrix Otsu split.

    Thin clefts that cut through the core surface (fractures reaching the rim)
    and enclosed pores are part of the specimen: a morphological closing seals
    the clefts and a hole fill sweeps in the enclosed voids, so dark crack
    voxels are not mistaken for outside air.
    """
    if mask is not None:
        return np.asarray(mask, dtype=bool), False
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=bool), True
    t = otsu_threshold(finite)
    raw = data > t
    if not raw.any():
        return raw, True
    # drop speckle components in the surrounding air (a couple of bright noise
    # voxels at the rim would otherwise be bridged by the closing below and
    # trap air pockets inside the interior); keep every significant fragment
    # so a core severed by a fracture is not reduced to one half
    lab_r, n_r = ndimage.label(raw, structure=np.ones((3, 3, 3), dtype=bool))
    if n_r > 1:
        sizes = np.bincount(lab_r.ravel())
        sizes[0] = 0
        raw = (sizes >= max(8, 0.005 * float(sizes.max()))) [lab_r]
    # Euclidean closing by a 3-voxel ball, computed with distance transforms on
    # a padded array: seals rim clefts (fractures reaching the core surface)
    # without scipy's binary_closing border artefacts (which eat outer slices).
    r = 3
    p = np.pad(raw, r, mode="constant", constant_values=False)
    dil = ndimage.distance_transform_edt(~p) <= r
    closed = ndimage.distance_transform_edt(dil) > r
    interior = closed[r:-r, r:-r, r:-r]
    return ndimage.binary_fill_holes(interior), True


def auto_threshold(interior_values: np.ndarray, air_value: float) -> float:
    """Dark-voxel threshold: air level + 25 % of the way to the matrix median.

    Anchored to the full-volume air mode rather than an Otsu of the interior:
    Otsu on a crack-free (unimodal) interior would split the matrix noise
    itself and fabricate dark candidates.
    """
    v = interior_values[np.isfinite(interior_values)]
    if v.size == 0:
        return float(air_value)
    med = float(np.median(v))
    return float(air_value + 0.25 * (med - air_value))


def detect_cracks(
    data: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: np.ndarray | None = None,
    *,
    threshold: float | str = "auto",
    min_voxels: int = 25,
    min_elongation: float = 0.55,
    max_aperture_mm: float | None = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect cracks; returns (int32 labels, JSON-safe metrics dict).

    ``threshold`` is ``"auto"`` or an explicit value in the data's own unit.
    ``min_elongation`` = 1 - sqrt(lambda_min / lambda_max) per component:
    ~0 for round pores, near 1 for planar/linear fractures.
    ``max_aperture_mm`` drops voids thicker than this: by definition a crack is
    a thin void, so this separates fractures from beam-hardening cupping
    blobs, vugs and cavities.  Pass ``None`` or 0 to keep everything.
    """
    data = np.asarray(data, dtype=np.float32)
    z, y, x = data.shape
    dz, dy, dx = (float(s) for s in spacing)
    voxel_mm3 = dz * dy * dx
    empty_labels = np.zeros(data.shape, dtype=np.int32)

    interior, auto_mask = specimen_interior(data, mask)
    interior = interior & np.isfinite(data)
    n_specimen = int(interior.sum())
    if n_specimen == 0:
        return empty_labels, _result(0, [], n_specimen, 0, 0.0, 0.0, None,
                                     np.zeros(_N_ORIENT_BINS), np.zeros(z),
                                     0.0, "user", auto_mask, spacing, dict(
                                         threshold=threshold, min_voxels=min_voxels,
                                         min_elongation=min_elongation, max_aperture_mm=max_aperture_mm))

    if isinstance(threshold, str):
        t_air = otsu_threshold(data[np.isfinite(data)])
        t = auto_threshold(data[interior], t_air)
        t_source = "auto"
    else:
        t = float(threshold)
        t_source = "user"

    cand = interior & (data < t)
    n_cand = int(cand.sum())
    if n_cand == 0:
        return empty_labels, _result(0, [], n_specimen, 0, t, 0.0, None,
                                     np.zeros(_N_ORIENT_BINS), np.zeros(z),
                                     t, t_source, auto_mask, spacing, dict(
                                         threshold=threshold, min_voxels=min_voxels,
                                         min_elongation=min_elongation, max_aperture_mm=max_aperture_mm))

    lab, n = ndimage.label(cand, structure=np.ones((3, 3, 3), dtype=bool))
    flat = lab.ravel()
    counts_all = np.bincount(flat)
    ids = np.nonzero(counts_all[1:] >= max(1, int(min_voxels)))[0] + 1
    if ids.size == 0:
        return empty_labels, _result(0, [], n_specimen, 0, t, 0.0, None,
                                     np.zeros(_N_ORIENT_BINS), np.zeros(z),
                                     t, t_source, auto_mask, spacing, dict(
                                         threshold=threshold, min_voxels=min_voxels,
                                         min_elongation=min_elongation, max_aperture_mm=max_aperture_mm))
    counts = counts_all[ids].astype(np.float64)

    # ---- per-component second moments (for elongation + orientation) ----
    def _w(axis_vals: np.ndarray) -> np.ndarray:
        return np.broadcast_to(axis_vals, data.shape).ravel()

    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    b = lambda w: np.bincount(flat, weights=w, minlength=n + 1)[ids]  # noqa: E731
    sz, sy, sx = b(_w(zz)), b(_w(yy)), b(_w(xx))
    szz, syy, sxx = b(_w(zz * zz)), b(_w(yy * yy)), b(_w(xx * xx))
    szy, szx, syx = b(_w(zz * yy)), b(_w(zz * xx)), b(_w(yy * xx))
    mz, my, mx = sz / counts, sy / counts, sx / counts

    # ---- surface area from boundary faces per axis ----
    def _internal(axis: int) -> np.ndarray:
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        la = lab[tuple(sl_a)]
        lb = lab[tuple(sl_b)]
        same = (la == lb) & (la > 0)
        return np.bincount(la[same], minlength=n + 1)[ids]

    iz, iy, ix = _internal(0), _internal(1), _internal(2)
    bnd_z = 2.0 * counts - 2.0 * iz   # z-faces (area dy*dx)
    bnd_y = 2.0 * counts - 2.0 * iy   # y-faces (area dz*dx)
    bnd_x = 2.0 * counts - 2.0 * ix   # x-faces (area dz*dy)
    surf = bnd_z * dy * dx + bnd_y * dz * dx + bnd_x * dz * dy
    vol_mm3 = counts * voxel_mm3
    aperture = np.divide(2.0 * vol_mm3, surf, out=np.zeros_like(vol_mm3),
                         where=surf > 0)

    # ---- shape + orientation via 3x3 covariance eigen decomposition ----
    comps = []
    keep = np.zeros(ids.size, dtype=bool)
    orient_hist = np.zeros(_N_ORIENT_BINS)
    for k in range(ids.size):
        c = np.array([
            [szz[k] / counts[k] - mz[k] ** 2, szy[k] / counts[k] - mz[k] * my[k],
             szx[k] / counts[k] - mz[k] * mx[k]],
            [szy[k] / counts[k] - my[k] * mz[k], syy[k] / counts[k] - my[k] ** 2,
             syx[k] / counts[k] - my[k] * mx[k]],
            [szx[k] / counts[k] - mx[k] * mz[k], syx[k] / counts[k] - mx[k] * my[k],
             sxx[k] / counts[k] - mx[k] ** 2],
        ])
        evals, evecs = np.linalg.eigh(c)
        e_min, e_max = max(float(evals[0]), 0.0), max(float(evals[2]), 1e-12)
        elong = 1.0 - float(np.sqrt(e_min / e_max))
        if elong < min_elongation:
            continue
        if max_aperture_mm and aperture[k] > max_aperture_mm:
            continue  # too thick to be a crack: vug, cavity or cupping blob
        keep[k] = True
        v = evecs[:, 2]
        ang = float(np.degrees(np.arctan2(v[1], v[2]))) % 180.0
        orient_hist[int(ang // (180.0 / _N_ORIENT_BINS)) % _N_ORIENT_BINS] += counts[k]
        comps.append({
            "id": int(ids[k]), "voxels": int(counts[k]),
            "volume_mm3": float(vol_mm3[k]), "aperture_mm": float(aperture[k]),
            "orientation_deg": round(ang, 1), "elongation": round(elong, 3),
            "center": [round(mz[k], 1), round(my[k], 1), round(mx[k], 1)],
        })

    if not comps:
        return empty_labels, _result(0, [], n_specimen, 0, t, 0.0, None,
                                     orient_hist, np.zeros(z), t, t_source,
                                     auto_mask, spacing, dict(
                                         threshold=threshold, min_voxels=min_voxels,
                                         min_elongation=min_elongation, max_aperture_mm=max_aperture_mm))

    # ---- compact labels (1..K, largest first) + aggregates ----
    comps.sort(key=lambda c: -c["voxels"])
    lut = np.zeros(n + 1, dtype=np.int32)
    order = {c["id"]: i + 1 for i, c in enumerate(comps)}
    for cid, new in order.items():
        lut[cid] = new
        c = next(c for c in comps if c["id"] == cid)
        c["id"] = new
    out = lut[lab].astype(np.int32)

    kept_voxels = int(sum(c["voxels"] for c in comps))
    per_slice = np.bincount(np.nonzero(out.ravel())[0] // (y * x),
                            minlength=z).astype(np.float64) * (dy * dx)
    total_vol = kept_voxels * voxel_mm3
    w = np.array([c["voxels"] for c in comps], dtype=np.float64)
    mean_ap = float(np.sum(w * [c["aperture_mm"] for c in comps]) / w.sum())
    if orient_hist.sum() > 0:
        dom = float((int(np.argmax(orient_hist)) + 0.5) * (180.0 / _N_ORIENT_BINS))
    else:
        dom = None

    return out, _result(len(comps), comps, n_specimen, kept_voxels, t, mean_ap,
                        dom, orient_hist, per_slice, t, t_source, auto_mask,
                        spacing, dict(threshold=threshold, min_voxels=min_voxels,
                                      min_elongation=min_elongation, max_aperture_mm=max_aperture_mm))


def _result(count: int, components: list[dict[str, Any]], specimen_voxels: int,
            kept_voxels: int, threshold: float, mean_aperture: float,
            dominant: float | None, orient_hist: np.ndarray,
            per_slice: np.ndarray, t: float, t_source: str, auto_mask: bool,
            spacing: tuple[float, float, float], params: dict[str, Any]) -> dict[str, Any]:
    voxel_mm3 = float(np.prod(spacing))
    total = sum(c["volume_mm3"] for c in components)
    return {
        "count": count,
        "specimen_voxels": int(specimen_voxels),
        "crack_voxels": int(kept_voxels),
        "volume_mm3": round(float(total), 3),
        "fraction_pct": round(100.0 * kept_voxels / max(specimen_voxels, 1), 4),
        "mean_aperture_mm": round(float(mean_aperture), 4),
        "max_aperture_mm": round(float(max((c["aperture_mm"] for c in components),
                                           default=0.0)), 4),
        "dominant_orientation_deg": dominant,
        "orientation_histogram": {
            "bin_deg": 180.0 / _N_ORIENT_BINS,
            "weights": [float(w) for w in orient_hist],
        },
        "per_slice_mm2": [float(a) for a in per_slice],
        "threshold": round(float(t), 2),
        "threshold_source": t_source,
        "auto_mask": auto_mask,
        "spacing": list(spacing),
        "components": components[:12],
        "params": params,
    }
