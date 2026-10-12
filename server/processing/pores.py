"""Pore detection and pore-size distribution in rock-core CT volumes.

Pores are the *compact* counterpart of cracks: dark, air-filled voids inside
the specimen.  The detector reuses the specimen interior and the air-anchored
dark threshold of :mod:`server.processing.cracks` to build a pore mask, then
sizes the pores the way digital-rock physics usually does - by **local
(inscribed-sphere) thickness** rather than by connected component, so an
interconnected pore network is still resolved into individual pores:

1. the Euclidean distance transform of the pore mask (in mm) gives every pore
   voxel the radius of the largest sphere that fits inside the pore space
   around it,
2. regional maxima of that field seed the pores; every pore voxel is assigned
   to its nearest seed (Voronoi in physical space) - this splits networks at
   their throats,
3. each zone is measured: voxel volume, equivalent-sphere diameter, inscribed
   radius, sphericity, PCA elongation and centroid; zones that are too small
   or too elongated (fracture apertures, not pores) are dropped,
4. the survivors are aggregated into the **pore-size distribution**: a
   histogram of equivalent-sphere diameters (log-spaced bins when sizes span
   more than a factor of three, linear otherwise) with number counts *and*
   volume per bin, number statistics (mean/median/max diameter), the mean
   local thickness, volume-weighted D10/D50/D90 quantiles and the porosity
   contributed by the resolved pores.

Pure numpy + scipy; the same code runs behind the REST API, inside the
analysis report and in the unit tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from server.processing.cracks import auto_threshold, specimen_interior
from server.processing.segment import otsu_threshold


def _seed_footprint(spacing: tuple[float, float, float]) -> tuple[int, int, int]:
    """Odd window ~3 small voxels wide on each axis (physical, anisotropy-aware)."""
    span = 3.0 * min(float(s) for s in spacing)
    sizes = []
    for s in spacing:
        n = max(3, int(np.ceil(span / max(float(s), 1e-9))))
        sizes.append(n + (1 - n % 2))
    return tuple(sizes)  # type: ignore[return-value]


def detect_pores(
    data: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: np.ndarray | None = None,
    *,
    threshold: float | str = "auto",
    min_voxels: int = 8,
    max_elongation: float | None = 0.7,
    bins: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect pores and their size distribution; returns (int32 labels, metrics).

    ``threshold`` is ``"auto"`` (air-anchored, same anchor as crack detection)
    or an explicit value in the data's own unit.  ``max_elongation`` = 1 -
    sqrt(lambda_min / lambda_max) per pore zone: fractures and aperture films
    are elongated, pores are compact, so zones above this are dropped; pass
    ``None`` or a value <= 0 to keep every void.  ``min_voxels`` discards
    zones below the reliable size.
    """
    data = np.asarray(data, dtype=np.float32)
    z, y, x = data.shape
    dz, dy, dx = (float(s) for s in spacing)
    voxel_mm3 = dz * dy * dx
    params = {"threshold": threshold, "min_voxels": int(min_voxels),
              "max_elongation": max_elongation, "bins": int(bins)}
    empty_labels = np.zeros(data.shape, dtype=np.int32)

    interior, auto_mask = specimen_interior(data, mask)
    interior = interior & np.isfinite(data)
    n_specimen = int(interior.sum())
    if n_specimen == 0:
        return empty_labels, _result([], 0, 0, 0.0, "user", auto_mask,
                                     spacing, params, bins)

    if isinstance(threshold, str):
        t = auto_threshold(data[interior], otsu_threshold(data[np.isfinite(data)]))
        t_source = "auto"
    else:
        t = float(threshold)
        t_source = "user"

    pore_mask = interior & (data < t)
    if not pore_mask.any():
        return empty_labels, _result([], n_specimen, 0, t, t_source, auto_mask,
                                     spacing, params, bins)

    # ---- 1. local thickness: inscribed-sphere radius per pore voxel ----
    edt = ndimage.distance_transform_edt(pore_mask, sampling=spacing)

    # ---- 2. seeds = regional maxima of the thickness field ----
    fp = _seed_footprint(spacing)
    peaks = (edt == ndimage.maximum_filter(edt, size=fp)) & pore_mask
    seed_lab, n_seeds = ndimage.label(peaks, structure=np.ones((3, 3, 3), dtype=bool))
    if n_seeds == 0:  # pragma: no cover - a non-empty mask always has a maximum
        return empty_labels, _result([], n_specimen, 0, t, t_source, auto_mask,
                                     spacing, params, bins)
    reps = np.array(ndimage.maximum_position(edt * (seed_lab > 0), seed_lab,
                                             range(1, n_seeds + 1)))
    seed_mask = np.zeros(data.shape, dtype=bool)
    seed_mask[tuple(reps.T)] = True

    # ---- 3. Voronoi assignment in physical space -> one zone per pore ----
    _, (iz, iy, ix) = ndimage.distance_transform_edt(
        ~seed_mask, return_indices=True, sampling=spacing)
    zone = np.where(pore_mask, seed_lab[iz, iy, ix], 0).astype(np.int32)

    # drop zones below the reliable size before the expensive per-zone passes
    ids_all = np.arange(1, n_seeds + 1)
    flat_all = zone.ravel()
    counts_all = np.bincount(flat_all, minlength=n_seeds + 1)
    alive = ids_all[counts_all[ids_all] >= max(1, int(min_voxels))]
    if alive.size == 0:
        return empty_labels, _result([], n_specimen, 0, t, t_source, auto_mask,
                                     spacing, params, bins)
    if alive.size < n_seeds:
        remap = np.zeros(n_seeds + 1, dtype=np.int32)
        remap[alive] = np.arange(1, alive.size + 1)
        zone = remap[zone]
        reps = reps[alive - 1]  # keep the seed coordinates of the survivors
        n_seeds = int(alive.size)

    # ---- 4. per-zone moments + surface area (vectorised over all zones) ----
    ids = np.arange(1, n_seeds + 1)
    flat = zone.ravel()
    counts = np.bincount(flat, minlength=n_seeds + 1)[ids].astype(np.float64)

    def _w(axis_vals: np.ndarray) -> np.ndarray:
        return np.broadcast_to(axis_vals, data.shape).ravel()

    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    b = lambda w: np.bincount(flat, weights=_w(w), minlength=n_seeds + 1)[ids]  # noqa: E731
    sz, sy, sx = b(zz), b(yy), b(xx)
    szz, syy, sxx = b(zz * zz), b(yy * yy), b(xx * xx)
    szy, szx, syx = b(zz * yy), b(zz * xx), b(yy * xx)
    mz, my, mx = sz / counts, sy / counts, sx / counts

    def _internal(axis: int) -> np.ndarray:
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        la = zone[tuple(sl_a)]
        lb = zone[tuple(sl_b)]
        same = (la == lb) & (la > 0)
        return np.bincount(la[same], minlength=n_seeds + 1)[ids]

    iz_n, iy_n, ix_n = _internal(0), _internal(1), _internal(2)
    bnd_z = 2.0 * counts - 2.0 * iz_n
    bnd_y = 2.0 * counts - 2.0 * iy_n
    bnd_x = 2.0 * counts - 2.0 * ix_n
    surf = bnd_z * dy * dx + bnd_y * dz * dx + bnd_x * dz * dy
    vol_mm3 = counts * voxel_mm3
    radius_mm = edt[tuple(reps.T)]

    cov = np.empty((n_seeds, 3, 3))
    cov[:, 0, 0] = szz / counts - mz * mz
    cov[:, 1, 1] = syy / counts - my * my
    cov[:, 2, 2] = sxx / counts - mx * mx
    cov[:, 0, 1] = cov[:, 1, 0] = szy / counts - mz * my
    cov[:, 0, 2] = cov[:, 2, 0] = szx / counts - mz * mx
    cov[:, 1, 2] = cov[:, 2, 1] = syx / counts - my * mx
    evals = np.linalg.eigh(cov)[0]
    elong = 1.0 - np.sqrt(np.maximum(evals[:, 0], 0.0) /
                          np.maximum(evals[:, 2], 1e-12))

    keep_limit = float(max_elongation) if (max_elongation is not None
                                           and float(max_elongation) > 0) else None
    keep = counts >= max(1, int(min_voxels))
    if keep_limit is not None:
        keep &= elong <= keep_limit

    comps = []
    for k in np.nonzero(keep)[0]:
        v = float(vol_mm3[k])
        a = float(surf[k])
        # Wadell sphericity: 1 for a sphere, small for ragged clusters
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * v) ** (2.0 / 3.0) / a
                      ) if a > 0 else 0.0
        comps.append({
            "id": int(ids[k]), "voxels": int(counts[k]),
            "volume_mm3": v,
            "diameter_mm": round(float((6.0 * v / np.pi) ** (1.0 / 3.0)), 4),
            "radius_mm": round(float(radius_mm[k]), 3),
            "elongation": round(float(elong[k]), 3),
            "sphericity": round(min(float(sphericity), 1.0), 3),
            "center": [round(float(mz[k]), 1), round(float(my[k]), 1),
                       round(float(mx[k]), 1)],
        })

    if not comps:
        return empty_labels, _result([], n_specimen, 0, t, t_source, auto_mask,
                                     spacing, params, bins)

    # ---- compact labels (1..K, largest first) for the slice overlay ----
    comps.sort(key=lambda c: -c["voxels"])
    lut = np.zeros(n_seeds + 1, dtype=np.int32)
    for i, c in enumerate(comps):
        lut[c["id"]] = i + 1
        c["id"] = i + 1
    out = lut[zone].astype(np.int32)

    kept_voxels = int(sum(c["voxels"] for c in comps))
    return out, _result(comps, n_specimen, kept_voxels, t, t_source, auto_mask,
                        spacing, params, bins)


def _result(components: list[dict[str, Any]], specimen_voxels: int, kept_voxels: int,
            threshold: float, t_source: str, auto_mask: bool,
            spacing: tuple[float, float, float], params: dict[str, Any],
            bins: int) -> dict[str, Any]:
    voxel_mm3 = float(np.prod(spacing))
    total_vol = float(sum(c["volume_mm3"] for c in components))
    diam = np.array([c["diameter_mm"] for c in components], dtype=np.float64)
    vols = np.array([c["volume_mm3"] for c in components], dtype=np.float64)

    if diam.size:
        lo, hi = float(diam.min()), float(diam.max())
        if hi - lo < 1e-9:
            lo, hi = max(lo * 0.9, 1e-6), hi * 1.1 + 1e-6
        log_spaced = bool(hi > 3.0 * max(lo, 1e-9))
        if log_spaced:
            edges = np.geomspace(lo, hi, max(4, int(bins)) + 1)
        else:
            edges = np.linspace(lo, hi, max(4, int(bins)) + 1)
        counts_hist, _ = np.histogram(diam, bins=edges)
        vol_hist, _ = np.histogram(diam, bins=edges, weights=vols)
        # volume-weighted quantiles of the equivalent diameter (D10/D50/D90)
        order = np.argsort(diam)
        ds, vs = diam[order], vols[order]
        cum = np.cumsum(vs) / max(float(vs.sum()), 1e-12)
        q = [float(np.interp(p, cum, ds)) for p in (0.10, 0.50, 0.90)]
        stats = {
            "mean_diameter_mm": round(float(diam.mean()), 4),
            "median_diameter_mm": round(float(np.median(diam)), 4),
            "max_diameter_mm": round(float(diam.max()), 4),
            "d10_mm": round(q[0], 4), "d50_mm": round(q[1], 4), "d90_mm": round(q[2], 4),
            "mean_local_thickness_mm": round(
                2.0 * float(np.mean([c["radius_mm"] for c in components])), 4),
        }
        distribution = {
            "bin_edges_mm": [float(e) for e in edges],
            "counts": [int(c) for c in counts_hist],
            "volume_mm3": [round(float(v), 4) for v in vol_hist],
            "log_spaced": log_spaced,
        }
    else:
        stats = {"mean_diameter_mm": 0.0, "median_diameter_mm": 0.0,
                 "max_diameter_mm": 0.0, "d10_mm": 0.0, "d50_mm": 0.0, "d90_mm": 0.0,
                 "mean_local_thickness_mm": 0.0}
        distribution = {"bin_edges_mm": [], "counts": [], "volume_mm3": [],
                        "log_spaced": False}

    mean_sph = float(np.mean([c["sphericity"] for c in components])) if components else 0.0
    return {
        "count": len(components),
        "specimen_voxels": int(specimen_voxels),
        "pore_voxels": int(kept_voxels),
        "volume_mm3": round(total_vol, 3),
        "volume_cm3": round(total_vol / 1000.0, 5),
        "porosity_pct": round(100.0 * kept_voxels / max(specimen_voxels, 1), 4),
        "mean_sphericity": round(mean_sph, 3),
        **stats,
        "threshold": round(float(threshold), 2),
        "threshold_source": t_source,
        "auto_mask": auto_mask,
        "spacing": list(spacing),
        "distribution": distribution,
        "components": components[:12],
        "params": params,
    }
