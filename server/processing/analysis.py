"""Quantitative analysis: global stats, masked stats, histograms, porosity.

Everything is reported together with the voxel size so volumes come out in
mm^3 / cm^3 and masses in g when the unit is density.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from server.processing.segment import histogram, otsu_threshold


def _finite(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    a = arr if mask is None else arr[mask.astype(bool)]
    return a[np.isfinite(a)]


def describe(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
        "skewness": float(_skew(values)),
        "kurtosis": float(_kurtosis(values)),
    }


def _skew(v: np.ndarray) -> float:
    m, s = float(v.mean()), float(v.std())
    if s < 1e-12:
        return 0.0
    return float(((v - m) ** 3).mean() / s ** 3)


def _kurtosis(v: np.ndarray) -> float:
    m, s = float(v.mean()), float(v.std())
    if s < 1e-12:
        return 0.0
    return float(((v - m) ** 4).mean() / s ** 4 - 3.0)


def analyze(
    data: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: np.ndarray | None = None,
    unit: str = "intensity",
    segmentation: np.ndarray | None = None,
    bins: int = 256,
    porosity_threshold: float | None = None,
    pore_size: bool = True,
) -> dict[str, Any]:
    """Full quantitative report for the volume (or a masked region of it)."""
    t0 = time.time()
    arr = np.asarray(data, dtype=np.float32)
    m = mask.astype(bool) if mask is not None else None
    vox_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    inside = _finite(arr, m)
    report: dict[str, Any] = {
        "unit": unit,
        "spacing_mm": {"z": spacing[0], "y": spacing[1], "x": spacing[2]},
        "voxel_volume_mm3": vox_mm3,
        "global": describe(arr[np.isfinite(arr)]),
        "roi": describe(inside),
        "histogram": histogram(arr, bins=bins, mask=m),
    }

    total_voxels = int(arr.size)
    roi_voxels = int(inside.size)
    report["coverage"] = {
        "total_voxels": total_voxels,
        "roi_voxels": roi_voxels,
        "roi_fraction": roi_voxels / max(total_voxels, 1),
        "roi_volume_cm3": roi_voxels * vox_mm3 / 1000.0,
    }

    # porosity / phase fraction from a threshold (auto-Otsu when not given)
    thr_source = "manual"
    if porosity_threshold is None:
        thr_source = "otsu"
        if inside.size > 1000:
            porosity_threshold = float(otsu_threshold(inside))
        else:
            porosity_threshold = None
    if porosity_threshold is not None and inside.size:
        solid = inside > porosity_threshold
        report["phases"] = {
            "threshold": float(porosity_threshold),
            "threshold_source": thr_source,
            "solid_voxels": int(solid.sum()),
            "pore_voxels": int((~solid).sum()),
            "solid_fraction": float(solid.mean()),
            "porosity": float(1.0 - solid.mean()),
            "solid_volume_cm3": float(solid.sum() * vox_mm3 / 1000.0),
            "pore_volume_cm3": float((~solid).sum() * vox_mm3 / 1000.0),
        }
        if unit == "density":
            report["phases"]["solid_mean_density"] = float(inside[solid].mean()) if solid.any() else None
            report["phases"]["bulk_density"] = float(inside.mean())
            report["phases"]["mass_g"] = float(inside.sum() * vox_mm3)  # g/cm^3 * mm^3 / 1000

    if segmentation is not None:
        from server.processing.segment import class_stats

        labels = segmentation
        stats = class_stats(labels, vox_mm3)
        sel = m if m is not None else np.ones(labels.shape, dtype=bool)
        for s in stats:
            lab = s["label"]
            sel_lab = (labels == lab) & sel
            if sel_lab.any():
                vals = arr[sel_lab]
                s["mean_value"] = float(np.nanmean(vals))
                s["std_value"] = float(np.nanstd(vals))
        report["classes"] = stats

    # pore-size distribution: resolved (segmentable) voids inside the specimen
    if pore_size:
        from server.processing.pores import detect_pores

        _, pores = detect_pores(arr, spacing, m)
        report["pore_size"] = pores

    # z-profile: useful to spot slices that are misaligned or drifted
    if arr.ndim == 3 and arr.shape[0] > 2:
        step = max(1, arr.shape[0] // 64)
        zs = list(range(0, arr.shape[0], step))
        profile = []
        for z in zs:
            v = _finite(arr[z], m[z] if m is not None and m.ndim == 3 else (m if m is not None else None))
            profile.append({"z": z, "mean": float(v.mean()) if v.size else 0.0,
                            "std": float(v.std()) if v.size else 0.0})
        report["z_profile"] = profile

    report["seconds"] = round(time.time() - t0, 3)
    return report


def radial_density_profile(data: np.ndarray, mask: np.ndarray, n_bins: int = 24) -> dict[str, Any]:
    """Density/attenuation vs normalised radius - the quantitative cupping view."""
    from server.processing.beamhardening import radial_profile, cupping_severity

    arr = data if data.ndim == 3 else data[None, ...]
    m = mask if mask.ndim == 3 else mask[None, ...]
    z = arr.shape[0] // 2
    prof = radial_profile(arr[z], m[z] if m.ndim == 3 else m[0], n_bins)
    return {"profile": prof, "cupping_severity": cupping_severity(prof), "slice": z}
