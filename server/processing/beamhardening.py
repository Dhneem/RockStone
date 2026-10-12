"""Beam-hardening tools.

Two separate capabilities, as requested:

* :func:`detect`   - quantify how badly a volume is affected (radial cupping
  profile, dark-streak fraction) and suggest a correction.
* :func:`correct`  - four interchangeable correction models, all with an
  adjustable `strength` blend so before/after comparison is meaningful:

  - ``polynomial`` : fit a polynomial through reference standards of known
    value (air / water / any material ROI you point at).
  - ``water``      : water + material linearisation (physical two/three point
    calibration, the standard "water calibration" workflow).
  - ``thickness``  : model-free cupping removal - fit the trend of value vs
    chord length through the specimen and subtract it.
  - ``spectral``   : explicit Cu/Al filter correction: build the tube spectrum
    from the calibration data (kVp, inherent Al, added filter) and invert the
    polychromatic forward model per voxel.

Units: ``HU`` volumes are converted to linear mu (1/mm) using NIST water at the
reference energy for the physical models; ``polynomial``/``water`` work in the
volume's own units.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import ndimage

from server.processing import spectra
from server.processing.roi import auto_boundary, auto_boundary_3d, sample_stats, shape_mask
from server.processing.segment import otsu_threshold

REFERENCE_ENERGY = spectra.REFERENCE_ENERGY  # keV
MU_WATER = float(spectra.linear_attenuation("water", np.array([REFERENCE_ENERGY]))[0])  # 1/mm at 60 keV


def mu_water_reference(unit: str, mu_ref: float | None = None) -> float | None:
    """Water reference used to turn HU into linear mu.

    Datasets that know their acquisition (metadata ``mu_water_reference``)
    supply the apparent thin-water value of that scan; otherwise the mono
    60 keV value is the conventional fallback.
    """
    if (unit or "").lower() != "hu":
        return None
    return float(mu_ref) if mu_ref else MU_WATER


# ------------------------------------------------------------------ units
def to_mu(data: np.ndarray, unit: str, mu_ref: float | None = None) -> np.ndarray:
    """Convert a volume to linear attenuation [1/mm]."""
    u = (unit or "intensity").lower()
    if u == "hu":
        ref = mu_ref if mu_ref else MU_WATER
        return ref * (1.0 + data / 1000.0)
    if u in ("mu", "1/mm", "mm^-1"):
        return data
    raise ValueError(f"unit '{unit}' has no absolute scale - use 'polynomial'/'thickness' "
                     f"correction or run calibration first")


def from_mu(mu: np.ndarray, unit: str, mu_ref: float | None = None) -> np.ndarray:
    u = (unit or "intensity").lower()
    if u == "hu":
        ref = mu_ref if mu_ref else MU_WATER
        return (mu / ref - 1.0) * 1000.0
    if u in ("mu", "1/mm", "mm^-1"):
        return mu
    raise ValueError(f"unit '{unit}' has no absolute scale")


# -------------------------------------------------------------- geometry
def fit_circle_radius(mask: np.ndarray) -> tuple[float, bool, float]:
    """(equivalent radius in px, is_circular, circularity 4*pi*A/P^2).

    The classical 4*pi*A/P^2 test is biased low here: the perimeter comes from
    a 3x3 erosion, which measures a ~sqrt(2)-px thick ring and reports 0.62 even
    for a perfect disc.  Roundness is therefore decided by the spread of the
    boundary-pixel radii about the centroid instead (a disc has ~1-2% spread,
    an irregular specimen much more); ``circ`` is still returned as an
    informative number.
    """
    area = float(mask.sum())
    if area <= 0:
        return 0.0, False, 0.0
    r = float(np.sqrt(area / np.pi))
    # perimeter via erosion (robust for pixel masks)
    er = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))
    border = mask & ~er
    per = float(border.sum())
    circ = 4.0 * np.pi * area / max(per * per, 1e-9)
    ys, xs = np.nonzero(border)
    if ys.size < 16:
        return r, False, circ
    cy, cx = float(ys.mean()), float(xs.mean())
    rb = np.hypot(ys - cy, xs - cx)
    spread = float(rb.std() / max(rb.mean(), 1e-9))
    return r, spread < 0.06, circ


def chord_field(mask: np.ndarray, spacing_xy: tuple[float, float] = (1.0, 1.0),
                mode: str = "auto", center: tuple[float, float] | None = None) -> np.ndarray:
    """Estimate ray chord length L(x, y) [mm] through the specimen.

    ``circle`` uses L = 2*sqrt(R^2 - r^2) for a cylindrical core (exact),
    ``dt`` uses twice the Euclidean distance transform (general shape),
    ``auto`` picks the circular model when the boundary is round.
    """
    mask = mask.astype(bool)
    sy, sx = spacing_xy
    pixel = float((sy + sx) / 2.0)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if mode == "auto":
        r, is_circ, _ = fit_circle_radius(mask)
        mode = "circle" if is_circ else "dt"
    if mode == "circle":
        r_px, _, _ = fit_circle_radius(mask)
        if center is None:
            ys, xs = np.nonzero(mask)
            cy, cx = float(ys.mean()), float(xs.mean())
        else:
            cx, cy = center
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        d2 = ((yy - cy) * sy) ** 2 + ((xx - cx) * sx) ** 2
        L = 2.0 * np.sqrt(np.maximum((r_px * pixel) ** 2 - d2, 0.0))
    else:
        dt = ndimage.distance_transform_edt(mask, sampling=(sy, sx))
        L = 2.0 * dt
    return np.where(mask, L, 0.0).astype(np.float32)


def _get_mask(data: np.ndarray, mask: np.ndarray | None, unit: str) -> np.ndarray:
    """Specimen mask for a 2-D or 3-D array (auto-detected when not given)."""
    if mask is not None:
        return mask.astype(bool)
    mid = data[data.shape[0] // 2] if data.ndim == 3 else data
    m, _ = auto_boundary(mid)
    return m


# ------------------------------------------------------------- detection
def radial_profile(slice2d: np.ndarray, mask: np.ndarray, n_bins: int = 32) -> dict[str, Any]:
    """Mean value vs normalised radius inside the specimen mask."""
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"radius": [], "mean": [], "count": []}
    cy, cx = float(ys.mean()), float(xs.mean())
    yy, xx = np.mgrid[0:slice2d.shape[0], 0:slice2d.shape[1]]
    d = np.hypot(yy - cy, xx - cx)
    r_max = float(d[mask].max()) or 1.0
    rn = d[mask] / r_max
    vals = slice2d[mask]
    idx = np.clip((rn * n_bins).astype(int), 0, n_bins - 1)
    centers, means, counts = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 3:
            continue
        centers.append(round((b + 0.5) / n_bins, 4))
        means.append(float(vals[sel].mean()))
        counts.append(int(sel.sum()))
    return {"radius": centers, "mean": means, "count": counts}


def cupping_severity(profile: dict[str, Any]) -> float:
    """(shell - core) / scale : positive values mean the classic cup artifact."""
    r = np.asarray(profile["radius"], float)
    m = np.asarray(profile["mean"], float)
    if r.size < 6:
        return 0.0
    core = m[r < 0.3].mean() if (r < 0.3).any() else m[:4].mean()
    shell = m[r > 0.75].mean() if (r > 0.75).any() else m[-4:].mean()
    scale = max(abs(core), abs(shell), float(np.ptp(m)) * 2.0, 1e-6)
    return float((shell - core) / scale)


def streak_fraction(slice2d: np.ndarray, mask: np.ndarray, k: float = 3.0) -> float:
    """Fraction of interior pixels sitting well below the local smooth level."""
    data = np.asarray(slice2d, dtype=np.float32)
    smooth = ndimage.gaussian_filter(data, 4.0)
    resid = data - smooth
    std = float(np.nanstd(resid)) + 1e-9
    interior = mask.astype(bool) & (ndimage.binary_erosion(mask, np.ones((9, 9), bool)))
    if interior.sum() < 16:
        interior = mask.astype(bool)
    dark = interior & (resid < -k * std)
    return float(dark.sum() / max(interior.sum(), 1))


def detect(data: np.ndarray, mask: np.ndarray | None = None, unit: str = "intensity",
           spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict[str, Any]:
    """Beam-hardening assessment for a volume or a single slice."""
    arr = data if data.ndim == 3 else data[None, ...]
    m = _get_mask(arr, mask, unit)
    m2 = m[m.shape[0] // 2] if m.ndim == 3 else m
    s2 = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr[0]
    if m2.ndim == 3:
        m2 = m2[m2.shape[0] // 2]

    prof = radial_profile(s2, m2)
    severity = cupping_severity(prof)
    streak = float(np.mean([streak_fraction(arr[i], m2 if m.ndim == 3 or m2.shape == arr.shape[-2:] else m)
                            for i in range(0, arr.shape[0], max(1, arr.shape[0] // 8))]))
    core = float(np.mean([p for r, p in zip(prof["radius"], prof["mean"]) if r < 0.3] or [0.0]))
    shell = float(np.mean([p for r, p in zip(prof["radius"], prof["mean"]) if r > 0.75] or [0.0]))

    abs_sev = abs(severity)
    if abs_sev < 0.02 and streak < 0.02:
        verdict, suggestion = "mild", {"method": "none", "strength": 0.0,
                                       "reason": "cupping and streaks within tolerance"}
    elif abs_sev < 0.06:
        verdict = "moderate"
        suggestion = {"method": "thickness", "strength": 0.6,
                      "reason": "moderate cupping - thickness trend removal is enough"}
    else:
        verdict = "severe"
        suggestion = {"method": "spectral", "strength": 1.0,
                      "reason": "strong cupping - use the spectrum/filter model if calibration "
                                "data (kVp, filter) are known, else 'thickness'"}
    if streak >= 0.03 and verdict == "mild":
        verdict = "moderate"
        suggestion = {"method": "thickness", "strength": 0.5, "reason": "dark streaks detected"}

    return {
        "cupping_profile": prof,
        "cupping_severity": severity,
        "core_mean": core,
        "shell_mean": shell,
        "dark_streak_fraction": streak,
        "verdict": verdict,
        "suggestion": suggestion,
        "specimen_area_fraction": float(m2.mean()),
        "note": "positive severity = value rises towards the rim (cup artifact)",
    }


# ------------------------------------------------------------- standards
def measure_standard(data: np.ndarray, spec: dict[str, Any], mask: np.ndarray | None) -> dict[str, Any]:
    """Mean value of a reference standard.

    ``spec`` either carries an explicit ROI (``roi`` key, see roi.shape_mask)
    or an ``auto`` key: 'air' (everything outside the specimen) or 'water'
    (in-specimen pixels closest to the nominal water value).
    """
    arr = data if data.ndim == 3 else data[None, ...]
    s2 = arr[arr.shape[0] // 2]
    if "roi" in spec:
        vals = sample_stats(s2, spec["roi"])
        vals["id"] = spec.get("id", "roi")
        return vals
    auto = spec.get("auto")
    m = _get_mask(arr, mask, spec.get("unit", "intensity"))
    m2 = m[m.shape[0] // 2] if m.ndim == 3 else m
    if auto == "air":
        vals = s2[~m2]
        ids = "air"
    elif auto == "water":
        nominal = float(spec.get("nominal", 0.0))
        window = float(spec.get("window", 250.0))
        # water inserts usually sit outside the specimen mask, so search the whole
        # slice; air sits ~1000 HU away and never falls inside the window
        pool = s2[np.abs(s2 - nominal) < window]
        if pool.size < 50:
            flat = np.sort(s2[np.isfinite(s2)])
            order = np.argsort(np.abs(flat - nominal))[: max(50, flat.size // 20)]
            pool = flat[order]
        vals = pool
        ids = "water"
    else:
        raise ValueError(f"unknown auto standard '{auto}'")
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"could not measure standard '{auto}'")
    return {"n": int(vals.size), "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()), "id": ids}


# -------------------------------------------------------------- correction
def correct(
    data: np.ndarray,
    method: str = "polynomial",
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    unit: str = "intensity",
    mask: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mu_water: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a beam-hardening correction.  Works on 2-D or 3-D input."""
    params = dict(params or {})
    strength = float(np.clip(strength, 0.0, 1.0))
    t0 = time.time()
    arr = np.asarray(data, dtype=np.float32)
    m = _get_mask(arr, mask, unit)

    before = detect(arr, m, unit, spacing)
    if method in ("none", "identity", None):
        out = arr.copy()
        info = {"method": "none", "strength": 0.0}
    elif method in ("polynomial", "water"):
        out, info = _correct_standards(arr, method, params, m)
    elif method == "thickness":
        out, info = _correct_thickness(arr, params, m, spacing)
    elif method == "spectral":
        out, info = _correct_spectral(arr, params, m, spacing, unit, mu_water)
    else:
        raise ValueError(f"unknown beam-hardening method '{method}'")

    blended = arr + strength * (out - arr)
    blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    after = detect(blended, m, unit, spacing)
    info.update({
        "strength": strength,
        "unit": unit,
        "seconds": round(time.time() - t0, 3),
        "before": {"cupping_severity": before["cupping_severity"],
                   "core_mean": before["core_mean"], "shell_mean": before["shell_mean"],
                   "dark_streak_fraction": before["dark_streak_fraction"]},
        "after": {"cupping_severity": after["cupping_severity"],
                  "core_mean": after["core_mean"], "shell_mean": after["shell_mean"],
                  "dark_streak_fraction": after["dark_streak_fraction"]},
    })
    return blended, info


def _correct_standards(arr: np.ndarray, method: str, params: dict[str, Any],
                       mask: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Polynomial / water calibration through reference standards."""
    standards = params.get("standards") or (
        [{"auto": "air", "true": -1000.0, "id": "air"},
         {"auto": "water", "true": 0.0, "id": "water"}]
    )
    measured = []
    targets = []
    for s in standards:
        st = dict(s)
        st.setdefault("unit", params.get("unit", "intensity"))
        m = measure_standard(arr, st, mask)
        measured.append(float(m["mean"]))
        targets.append(float(s.get("true", s.get("value", 0.0))))
    measured_a = np.asarray(measured)
    targets_a = np.asarray(targets)

    if method == "water":
        degree = int(params.get("degree", 1 if len(measured) < 3 else 2))
    else:
        degree = int(params.get("degree", min(len(measured) - 1, 3)))
    degree = max(0, min(degree, len(measured) - 1))
    coeffs = np.polyfit(measured_a, targets_a, degree) if len(measured) > 0 else np.array([0.0])
    out = np.polyval(coeffs, arr).astype(np.float32)
    resid = float(np.max(np.abs(np.polyval(coeffs, measured_a) - targets_a))) if measured else 0.0
    return out, {
        "method": method,
        "degree": degree,
        "coefficients": [float(c) for c in np.atleast_1d(coeffs)],
        "standards": [{"id": s.get("id", s.get("auto", "roi")), "measured": mv, "true": tv}
                      for s, mv, tv in zip(standards, measured, targets)],
        "max_standard_residual": resid,
    }


def _correct_thickness(arr: np.ndarray, params: dict[str, Any], mask: np.ndarray | None,
                       spacing: tuple[float, float, float]) -> tuple[np.ndarray, dict[str, Any]]:
    """Cupping removal: fit value-vs-chord-length trend, subtract it."""
    if mask is None or not np.asarray(mask).any():
        raise ValueError("thickness correction needs a specimen mask (auto-detection failed)")
    m = np.asarray(mask).astype(bool)
    degree = int(params.get("degree", 2))
    n_bins = int(params.get("bins", 24))
    mode = params.get("chord_mode", "auto")
    sy, sx = spacing[1], spacing[2]

    def one(slice2d: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        L = chord_field(m2, (sy, sx), mode=mode)
        inside = m2 & (L > 0)
        if inside.sum() < 50:
            return slice2d, {"skipped": True}
        li = L[inside]
        vi = slice2d[inside]
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(li, qs))
        if edges.size < 3:
            return slice2d, {"skipped": True}
        idx = np.clip(np.searchsorted(edges, li, side="right") - 1, 0, edges.size - 2)
        centers, means = [], []
        for b in range(edges.size - 1):
            sel = idx == b
            if sel.sum() < 3:
                continue
            centers.append(float(li[sel].mean()))
            means.append(float(vi[sel].mean()))
        if len(centers) <= degree:
            return slice2d, {"skipped": True}
        scale = max(float(L.max()), 1e-6)
        c = np.asarray(centers) / scale
        coeffs = np.polyfit(c, np.asarray(means), degree)
        trend = np.polyval(coeffs, L / scale)
        corr = -(trend - float(np.mean(trend[inside])))  # keep the mean level
        return (slice2d + corr).astype(np.float32), {
            "trend_coefficients": [float(x) for x in coeffs],
            "chord_max_mm": float(L.max()),
            "chord_mode": mode,
        }

    if arr.ndim == 2:
        out, info = one(arr, m if m.ndim == 2 else m[m.shape[0] // 2])
        return out, {"method": "thickness", **info}
    out = np.empty_like(arr)
    infos = []
    for i in range(arr.shape[0]):
        mi = m[i] if m.ndim == 3 else m
        res, info = one(arr[i], mi)
        out[i] = res
        if i == arr.shape[0] // 2:
            infos.append(info)
    return out, {"method": "thickness", "applied_to": "all slices", **(infos[0] if infos else {})}


def _correct_spectral(arr: np.ndarray, params: dict[str, Any], mask: np.ndarray | None,
                      spacing: tuple[float, float, float], unit: str,
                      mu_water: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Cu/Al filter correction: invert the polychromatic forward model."""
    ref = params.get("mu_water_reference", mu_water)
    mu = to_mu(arr, unit, ref)  # raises for scale-less units
    kvp = float(params.get("kvp", 80.0))
    inherent = float(params.get("inherent_al_mm", 2.5))
    added_filter = params.get("added_filter") or None
    added_mm = float(params.get("added_mm", 0.0) or 0.0)
    material = params.get("material", "water")
    if added_filter in ("none", "", "None"):
        added_filter = None
    energies, weights = spectra.tube_spectrum(kvp, int(params.get("bins", 64)), inherent,
                                              added_filter, added_mm)
    if mask is None or not np.asarray(mask).any():
        raise ValueError("spectral correction needs a specimen mask (auto-detection failed)")
    m = np.asarray(mask).astype(bool)
    sy, sx = spacing[1], spacing[2]
    mode = params.get("chord_mode", "auto")

    def one(mu_slice: np.ndarray, m2: np.ndarray) -> np.ndarray:
        L = chord_field(m2, (sy, sx), mode=mode)
        L = np.where(L > 0, L, 0.0)
        return spectra.invert_polychromatic_mu(
            material, energies, weights, mu_slice, L,
            reference_energy=float(params.get("reference_energy", REFERENCE_ENERGY)),
        ).astype(np.float32)

    if mu.ndim == 2:
        mu0 = one(mu, m if m.ndim == 2 else m[m.shape[0] // 2])
    else:
        mu0 = np.empty_like(mu)
        for i in range(mu.shape[0]):
            mi = m[i] if m.ndim == 3 else m
            mu0[i] = one(mu[i], mi)
    # input HU were scaled by the scan's apparent water reference, but the
    # corrected output should land on the conventional mono-energetic water
    # scale (water = 0 HU at the reference energy)
    return from_mu(mu0, unit, MU_WATER), {
        "method": "spectral",
        "kvp": kvp,
        "inherent_al_mm": inherent,
        "added_filter": added_filter,
        "added_mm": added_mm,
        "material": material,
        "reference_energy_keV": float(params.get("reference_energy", REFERENCE_ENERGY)),
        "mean_energy_keV": float((energies * weights).sum()),
        "chord_mode": mode,
    }
