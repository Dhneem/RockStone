"""Approximate X-ray physics: mass attenuation tables and tube spectra.

The attenuation coefficients are rounded tabulations of NIST X-Ray Mass
Attenuation Coefficient data (water/air from table 4, Al/Cu/Si from the
element tables) - accurate to a few percent, which is plenty for artifact
modelling and correction, though not metrology grade.  Energies are in keV,
mu/rho in cm^2/g, densities in g/cm^3, linear mu in 1/mm.
"""

from __future__ import annotations

import numpy as np

ENERGIES = np.array([10, 15, 20, 30, 40, 50, 60, 80, 100, 150], dtype=np.float64)

ATTEN = {
    "water": np.array([5.329, 1.673, 0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707, 0.1505]),
    "air": np.array([5.120, 1.614, 0.7779, 0.3538, 0.2485, 0.2080, 0.1875, 0.1662, 0.1541, 0.1356]),
    "aluminum": np.array([26.23, 7.955, 3.441, 1.128, 0.5685, 0.3681, 0.2778, 0.2018, 0.1704, 0.1378]),
    "copper": np.array([215.9, 74.05, 33.79, 10.92, 4.862, 2.613, 1.593, 0.7630, 0.4584, 0.2217]),
    "silicon": np.array([33.89, 10.34, 4.464, 1.436, 0.7012, 0.4385, 0.3207, 0.2228, 0.1835, 0.1448]),
}

DENSITY = {"water": 1.0, "air": 0.001205, "aluminum": 2.699, "copper": 8.96, "silicon": 2.33}

# Compounds built from element tables by mass fraction.  Oxygen is approximated
# by dry air (N and O are adjacent and differ by <3% in the diagnostic range).
_COMPOUND_MIXES = {"quartz": [(0.4674, "silicon"), (0.5326, "air")],
                   "rock": [(0.4674, "silicon"), (0.5326, "air")]}
_COMPOUND_DENSITY = {"quartz": 2.65, "rock": 2.65}

REFERENCE_ENERGY = 60.0  # keV - the "mono" energy quoted for corrections


def mass_attenuation(material: str, energies: np.ndarray) -> np.ndarray:
    """Log-interpolated mu/rho [cm^2/g] for `material` at `energies` [keV]."""
    key = material.lower()
    if key in ATTEN:
        table = ATTEN[key]
    elif key in _COMPOUND_MIXES:
        table = sum(w * ATTEN[m] for w, m in _COMPOUND_MIXES[key])
    else:
        raise KeyError(f"unknown material '{material}' (have {sorted(ATTEN)})")
    energies = np.asarray(energies, dtype=np.float64)
    return np.exp(
        np.interp(energies, ENERGIES, np.log(table), left=np.log(table[0]), right=np.log(table[-1]))
    )


def linear_attenuation(material: str, energies: np.ndarray, density: float | None = None) -> np.ndarray:
    """mu [1/mm]: mu/rho [cm^2/g] * rho [g/cm^3] / 10."""
    key = material.lower()
    if density is None:
        density = DENSITY.get(key, _COMPOUND_DENSITY.get(key, 1.0))
    return mass_attenuation(key, energies) * density / 10.0


def material_mu_shape(
    material: str,
    energies: np.ndarray,
    reference_energy: float = REFERENCE_ENERGY,
    density: float | None = None,
) -> tuple[np.ndarray, float]:
    """Energy dependence mu(E)/mu(Eref) plus mu(Eref) [1/mm] for a material."""
    mu = linear_attenuation(material, energies, density)
    mu_ref = float(np.exp(np.interp(np.log(reference_energy), np.log(energies), np.log(mu))))
    return mu / mu_ref, mu_ref


def polychromatic_mu(
    material: str,
    energies: np.ndarray,
    weights: np.ndarray,
    thickness_mm: np.ndarray,
    density: float | None = None,
) -> np.ndarray:
    """Beam-hardened apparent attenuation -ln(sum w e^{-mu(E) L}) / L [1/mm].

    Tends to the monochromatic value as L -> 0 and falls off with thickness
    (the cupping term).  `material` supplies the energy dependence.
    """
    mu = linear_attenuation(material, energies, density)
    L = np.asarray(thickness_mm, dtype=np.float64)
    return _mu_eff(mu, weights, L)


def _mu_eff(mu_energy: np.ndarray, weights: np.ndarray, L: np.ndarray) -> np.ndarray:
    Lr = np.atleast_1d(L).ravel()
    ex = np.exp(-np.multiply.outer(Lr, mu_energy))
    t = ex @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(np.maximum(t, 1e-300)) / np.maximum(Lr, 1e-9)
    out = np.where(Lr < 1e-6, float(mu_energy @ weights), out)
    return out.reshape(np.shape(L))


def polychromatic_mu0(
    mu_shape: np.ndarray,
    weights: np.ndarray,
    mu0: np.ndarray,
    thickness_mm: np.ndarray,
) -> np.ndarray:
    """Apparent mu for a material whose monochromatic value is mu0, using the
    energy shape mu(E)/mu(Eref).  Broadcasts mu0 against thickness_mm."""
    mu0_b, L_b = np.broadcast_arrays(np.asarray(mu0, float), np.asarray(thickness_mm, float))
    shape = mu0_b.shape
    mu_energy = mu_shape * mu0_b.ravel()[:, None]
    L = L_b.ravel()
    ex = np.exp(-mu_energy * L[:, None])
    t = ex @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(np.maximum(t, 1e-300)) / np.maximum(L, 1e-9)
    out = np.where(L < 1e-6, mu_energy @ weights, out)
    return out.reshape(shape)


def invert_polychromatic_mu(
    material: str,
    energies: np.ndarray,
    weights: np.ndarray,
    mu_measured: np.ndarray,
    thickness_mm: np.ndarray,
    reference_energy: float = REFERENCE_ENERGY,
    density: float | None = None,
) -> np.ndarray:
    """Solve for the monochromatic mu0 (at `reference_energy`) that would yield
    the measured beam-hardened value at the given thickness.

    A monotone lookup table per thickness knot, inverted by interpolation -
    faster and more stable than a per-voxel root find.
    """
    mu_shape, _ = material_mu_shape(material, energies, reference_energy, density)
    shape = np.broadcast_shapes(np.shape(mu_measured), np.shape(thickness_mm))
    L = np.broadcast_to(np.asarray(thickness_mm, float), shape).ravel()
    M = np.broadcast_to(np.asarray(mu_measured, float), shape).ravel()

    mu0_grid = np.geomspace(1e-4, max(0.5, float(np.nanmax(M)) * 8.0), 96)
    # knot spacing over the *material* chords only - outside-mask L == 0 pixels
    # would otherwise waste most of the quantile knots at L = 0
    Lp = L[L > 1e-6]
    knots = np.unique(Lp if Lp.size else L)
    if knots.size > 12:
        knots = np.quantile(Lp, np.linspace(0, 1, 12))
    if knots.size == 1:
        knots = np.array([knots[0], knots[0]])
    curve = np.stack(
        [polychromatic_mu0(mu_shape, weights, mu0_grid, np.full(mu0_grid.size, max(k, 1e-3)))
         for k in knots]
    )

    inv = np.empty((curve.shape[0], M.size))
    for i in range(curve.shape[0]):
        row = np.maximum.accumulate(curve[i])
        inv[i] = np.interp(M, row, mu0_grid, left=mu0_grid[0], right=mu0_grid[-1])

    hi_idx = np.clip(np.searchsorted(knots, L), 1, knots.size - 1)
    lo_idx = hi_idx - 1
    lo, hi = knots[lo_idx], knots[hi_idx]
    t = np.where(hi > lo, (L - lo) / np.maximum(hi - lo, 1e-12), 0.0)
    cols = np.arange(M.size)
    mu0 = inv[lo_idx, cols] * (1.0 - t) + inv[hi_idx, cols] * t
    return mu0.reshape(shape)


def tube_spectrum(
    kvp: float = 80.0,
    bins: int = 64,
    inherent_al_mm: float = 2.5,
    added_filter: str | None = None,
    added_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate tungsten-bremsstrahlung spectrum.

    Returns (energies [keV], weights) normalised to sum 1.  Tungsten characteristic
    K lines are added above 70 kVp.
    """
    e_max = float(kvp)
    e = np.linspace(max(10.0, e_max / bins), e_max, bins)
    w = ((e_max - e) / np.maximum(e, 1.0)) ** 1.2
    w *= np.exp(-linear_attenuation("aluminum", e) * inherent_al_mm)
    if added_filter and added_mm > 0:
        w *= np.exp(-linear_attenuation(added_filter, e) * added_mm)
    if kvp > 69.5:
        kalpha = np.exp(-0.5 * ((e - 59.3) / 1.4) ** 2)
        kbeta = np.exp(-0.5 * ((e - 67.2) / 1.6) ** 2)
        w = w + 0.12 * w.max() * kalpha + 0.04 * w.max() * kbeta
    w = np.clip(w, 0, None)
    w /= w.sum()
    return e, w
