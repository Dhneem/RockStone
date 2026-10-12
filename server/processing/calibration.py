"""Intensity calibration: map raw/HU values onto a physical scale.

Two-point (or N-point) calibration with reference standards of known value:
air, water, or any ROI you point at.  The result can be expressed in HU, in
linear attenuation (1/mm) or in density (g/cm^3) for quantitative analysis.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from server.processing.beamhardening import MU_WATER, measure_standard, to_mu
from server.processing.roi import sample_stats, shape_mask


def calibrate(
    data: np.ndarray,
    standards: list[dict[str, Any]] | None = None,
    target: str = "density",
    degree: int = 1,
    unit: str = "intensity",
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit `target = p(value)` through the standards and apply it to the data.

    standards: [{auto:'air'|'water'} or {roi:{...}}, true: <value in target units>]
    target: 'density' | 'HU' | 'mu'
    """
    t0 = time.time()
    arr = np.asarray(data, dtype=np.float32)
    if standards is None:
        if target == "density":
            standards = [{"auto": "air", "true": 0.0, "id": "air"},
                         {"auto": "water", "true": 1.0, "id": "water"}]
        elif target == "HU":
            standards = [{"auto": "air", "true": -1000.0, "id": "air"},
                         {"auto": "water", "true": 0.0, "id": "water"}]
        else:  # mu
            standards = [{"auto": "air", "true": 0.0, "id": "air"},
                         {"auto": "water", "true": MU_WATER, "id": "water"}]

    measured, targets = [], []
    for s in standards:
        st = dict(s)
        st.setdefault("unit", unit)
        if "roi" in st:
            s2 = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr
            stats = sample_stats(s2, st["roi"])
            mean = float(stats["mean"])
        else:
            mean = float(measure_standard(arr, st, mask)["mean"])
        measured.append(mean)
        targets.append(float(s["true"]))

    if len(measured) < 2:
        raise ValueError("calibration needs at least two standards")
    deg = max(1, min(int(degree), len(measured) - 1))
    coeffs = np.polyfit(np.asarray(measured), np.asarray(targets), deg)
    out = np.polyval(coeffs, arr).astype(np.float32)

    new_unit = {"density": "density", "HU": "HU", "mu": "1/mm"}.get(target, target)
    info = {
        "method": "polynomial_standard_fit",
        "target": target,
        "degree": deg,
        "coefficients": [float(c) for c in coeffs],
        "standards": [{"id": s.get("id", s.get("auto", "roi")), "measured": m, "true": t}
                      for s, m, t in zip(standards, measured, targets)],
        "unit_from": unit,
        "unit_to": new_unit,
        "seconds": round(time.time() - t0, 3),
    }
    return out, info


def hu_to_density(hu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    """Standard two-point HU -> density (g/cm^3) mapping assuming mu ~ density."""
    mu = MU_WATER * (1.0 + hu / 1000.0)
    return (mu / mu_water).astype(np.float32)
