"""Synthetic rock-core CT sample with realistic artifacts.

The phantom is a cylindrical sandstone core (quartz matrix, grains, pores,
high-attenuation pyrite specks) plus a water calibration tube, rendered through
a polychromatic beam model so the reconstruction shows genuine beam-hardening
cupping.  On top of that we add detector rings, Poisson-like noise and
slice-to-slice registration jitter - i.e. every artifact the workbench has a
tool for.

Everything is derived from the same NIST-based spectra module the correction
tools use, so plugging the acquisition parameters (kVp, filter) back into the
beam-hardening correction recovers a clean volume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from server.processing import spectra
from server.store import Volume


def _value_noise(shape: tuple[int, int], scale: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth random field in [0, 1] via upsampled coarse noise."""
    small = rng.random((max(2, shape[0] // scale), max(2, shape[1] // scale)))
    big = ndimage.zoom(small, (shape[0] / small.shape[0], shape[1] / small.shape[1]), order=3)
    big = big[: shape[0], : shape[1]]
    if big.shape != shape:
        pad = ((0, shape[0] - big.shape[0]), (0, shape[1] - big.shape[1]))
        big = np.pad(big, pad, mode="edge")
    lo, hi = float(big.min()), float(big.max())
    return (big - lo) / max(hi - lo, 1e-9)


def phantom_slice(
    shape: tuple[int, int],
    rng: np.random.Generator,
    rock_radius_px: float,
    water_center: tuple[float, float],
    water_radius_px: float,
    porosity: float = 0.10,
    grain_fraction: float = 0.035,
) -> dict[str, np.ndarray]:
    """Materials + geometry for a single slice (as float coefficients 0/1)."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # rock core with a slightly rough boundary
    rough = _value_noise(shape, 24, rng) - 0.5
    rho = np.hypot(yy - cy, xx - cx) + rough * 6.0
    rock = rho <= rock_radius_px

    # water calibration tube
    rho_w = np.hypot(yy - water_center[1], xx - water_center[0])
    water = rho_w <= water_radius_px

    # internal structure
    grain_field = _value_noise(shape, 6, rng)
    pore_field = _value_noise(shape, 5, rng)
    fine = _value_noise(shape, 2, rng)
    grains = rock & (grain_field > 1.0 - grain_fraction)
    pores = rock & (pore_field < porosity) & ~grains
    matrix = rock & ~grains & ~pores

    return {"rock_mask": rock, "water": water, "matrix": matrix, "grains": grains,
            "pores": pores, "rho": rho, "rho_w": rho_w, "fine": fine}


def generate(
    shape: tuple[int, int, int] = (128, 288, 288),
    spacing: tuple[float, float, float] = (0.3, 0.2, 0.2),  # z, y, x in mm
    kvp: float = 80.0,
    inherent_al_mm: float = 2.5,
    added_filter: str = "aluminum",
    added_mm: float = 3.0,
    noise_hu: float = 12.0,
    rings: int = 10,
    ring_hu: float = 26.0,
    jitter_px: float = 1.6,
    jitter_deg: float = 0.30,
    seed: int = 7,
    porosity: float = 0.10,
) -> tuple[Volume, dict[str, Any]]:
    """Build the synthetic CT volume.  Returns (volume, ground_truth)."""
    rng = np.random.default_rng(seed)
    z, h, w = shape
    _, sy, sx = spacing
    pixel = (sy + sx) / 2.0

    # geometry: core + water tube, both centred in the field of view
    rock_radius_px = 0.34 * min(h, w)
    water_radius_px = 0.075 * min(h, w)
    water_cx = w / 2.0 + rock_radius_px + water_radius_px + 0.055 * w
    water_center = (water_cx, h / 2.0)  # (x, y)

    energies, weights = spectra.tube_spectrum(kvp, 64, inherent_al_mm, added_filter, added_mm)
    q_shape, _ = spectra.material_mu_shape("quartz", energies)
    w_shape, _ = spectra.material_mu_shape("water", energies)

    mu_matrix0 = float(spectra.linear_attenuation("quartz", np.array([60.0]))[0])  # 1/mm
    mu_grain0 = mu_matrix0 * 3.2
    mu_water0 = float(spectra.linear_attenuation("water", np.array([60.0]))[0])
    # scanners define HU from air + water *as measured in this scan*, so the
    # reference is the apparent (beam-hardened) attenuation of a thin water
    # reference rather than the mono-energetic value
    mu_water_ref = float((weights * spectra.linear_attenuation("water", energies)).sum())
    MU_WATER = mu_water_ref

    # per-slice jitter (reconstruction instability) - random walk
    dxs = np.cumsum(rng.normal(0, jitter_px / 2.0, z))
    dys = np.cumsum(rng.normal(0, jitter_px / 2.0, z))
    rots = np.cumsum(rng.normal(0, jitter_deg / 2.0, z))
    dxs -= dxs.mean()
    dys -= dys.mean()
    rots -= rots.mean()

    # detector ring profile: a few faulty bins, concentric with the centre
    r_max = float(min(h, w) / 2.0)
    n_bins = int(r_max)
    ring_profile = np.zeros(n_bins)
    if rings > 0:
        bad = rng.choice(n_bins, size=int(rings), replace=False)
        for b in bad:
            amp = rng.uniform(-1.0, 1.0) * ring_hu
            width = rng.uniform(0.8, 2.0)
            rr = np.arange(n_bins)
            ring_profile += amp * np.exp(-0.5 * ((rr - b) / width) ** 2)

    ground = {
        "phantom": {
            "rock_radius_mm": rock_radius_px * pixel,
            "water_tube": {"center_px": [float(water_center[0]), float(water_center[1])],
                           "radius_px": water_radius_px, "radius_mm": water_radius_px * pixel},
            "porosity": porosity,
        },
        "acquisition": {"kvp": kvp, "inherent_al_mm": inherent_al_mm,
                        "added_filter": added_filter, "added_mm": added_mm,
                        "material": "quartz", "reference_energy_keV": 60.0},
        "artifacts": {"noise_hu": noise_hu, "rings": int(rings), "ring_hu": ring_hu,
                      "jitter_px": jitter_px, "jitter_deg": jitter_deg},
        "standards": [
            {"id": "air", "auto": "air", "true": -1000.0},
            {"id": "water", "auto": "water", "true": 0.0, "nominal": 0.0, "window": 250.0},
        ],
        "truth": {"mu_matrix_1_per_mm": mu_matrix0, "mu_water_1_per_mm": mu_water0,
                  "mu_water_reference_1_per_mm": mu_water_ref,
                  "matrix_HU_mono60": 1000.0 * (mu_matrix0 / mu_water0 - 1.0)},
    }

    vol = np.empty(shape, dtype=np.float32)
    mm_per_px = pixel
    for iz in range(z):
        ph = phantom_slice((h, w), rng, rock_radius_px, water_center, water_radius_px, porosity)
        rock_mask, water_mask = ph["rock_mask"], ph["water"]
        matrix, grains, pores = ph["matrix"], ph["grains"], ph["pores"]

        # monochromatic mu0 field at the reference energy
        mu0 = np.zeros((h, w), dtype=np.float64)
        variation = (_value_noise((h, w), 10, rng) - 0.5) * 0.10  # +-5% bedding density variation
        mu0[matrix] = mu_matrix0 * (1.0 + variation[matrix])
        mu0[grains] = mu_grain0
        mu0[pores] = 0.004  # air-filled pore
        mu0[water_mask] = mu_water0

        # --- polychromatic forward model (chord length through each cylinder)
        R = rock_radius_px * mm_per_px
        rho_mm = ph["rho"] * mm_per_px
        L_rock = 2.0 * np.sqrt(np.maximum(R ** 2 - rho_mm ** 2, 0.0))
        rw = water_radius_px * mm_per_px
        rho_w_mm = ph["rho_w"] * mm_per_px
        L_water = 2.0 * np.sqrt(np.maximum(rw ** 2 - rho_w_mm ** 2, 0.0))

        L = np.where(rock_mask, L_rock, 0.0) + np.where(water_mask, L_water, 0.0)
        mu_meas = np.zeros((h, w), dtype=np.float64)
        rock_sel = rock_mask & (L > 0)
        water_sel = water_mask & (L > 0)
        if rock_sel.any():
            mu_meas[rock_sel] = spectra.polychromatic_mu0(q_shape, weights, mu0[rock_sel], L[rock_sel])
        if water_sel.any():
            mu_meas[water_sel] = spectra.polychromatic_mu0(w_shape, weights, mu0[water_sel], L[water_sel])

        # --- to HU
        hu = (mu_meas / MU_WATER - 1.0) * 1000.0
        hu[~rock_mask & ~water_mask] = -1000.0  # air

        # --- detector rings (concentric, image centre)
        rr = np.hypot(np.mgrid[0:h, 0:w][0] - h / 2.0, np.mgrid[0:h, 0:w][1] - w / 2.0)
        idx = np.clip(rr.astype(int), 0, n_bins - 1)
        hu += ring_profile[idx]

        # --- Poisson-like noise (scaled by attenuation)
        signal = np.clip((hu + 1000.0) / 1000.0, 0.0, None)
        sigma = noise_hu * (0.35 + 0.65 * np.sqrt(signal))
        hu = hu + rng.normal(0.0, 1.0, (h, w)) * sigma

        # --- slight in-plane blur from the reconstruction kernel
        hu = ndimage.gaussian_filter(hu, 0.55)

        vol[iz] = hu.astype(np.float32)

        if abs(dxs[iz]) > 1e-6 or abs(dys[iz]) > 1e-6 or abs(rots[iz]) > 1e-6:
            vol[iz] = _warp(vol[iz], dxs[iz], dys[iz], rots[iz])

    ground["artifacts"]["actual_jitter"] = {
        "max_dx_px": float(np.max(np.abs(dxs))), "max_dy_px": float(np.max(np.abs(dys))),
        "max_rot_deg": float(np.max(np.abs(rots))),
        "dx": [round(float(v), 3) for v in dxs],
        "dy": [round(float(v), 3) for v in dys],
        "rot": [round(float(v), 4) for v in rots],
    }

    meta = {
        "format": "synthetic-phantom",
        "description": "Synthetic sandstone core + water calibration tube (polychromatic forward model)",
        "manufacturer": "Simulated CT",
        "manufacturer_model": "Rockstone SynthScan",
        "modality": "CT",
        "kvp": kvp,
        "convolution_kernel": "FBP sim",
        "unit": "HU",
        "detections": {
            "slice_spacing": {"value": spacing[0], "source": "declared (synthetic)"},
            "pixel_size": {"value": (spacing[1], spacing[2]), "source": "declared (synthetic)"},
            "image_dimensions": {"value": [w, h, z], "source": "measured"},
            "scanner_metadata": {"value": f"{kvp} kVp, {inherent_al_mm} mm Al inherent + "
                                          f"{added_mm} mm {added_filter} added filtration",
                                 "source": "synthetic ground truth"},
        },
        "spacing_source": {"z": "declared", "y": "declared", "x": "declared"},
        "acquisition": ground["acquisition"],
        "standards": ground["standards"],
        "mu_water_reference": mu_water_ref,
        "ground_truth": ground,
    }

    vol_obj = Volume(data=vol, spacing=spacing, metadata=meta, name="synthetic-rock-core", unit="HU")
    return vol_obj, ground


def _warp(img: np.ndarray, dx: float, dy: float, rot_deg: float, order: int = 1) -> np.ndarray:
    from server.processing.align import apply_transform

    return apply_transform(img, dx, dy, rot_deg, 1.0, order=order,
                           cval=float(np.median(img))).astype(np.float32)


# ------------------------------------------------------------ export sample
def write_sample_dataset(vol: Volume, out_dir: Path, formats: tuple[str, ...] = ("tiff", "dicom", "raw")) -> dict[str, Any]:
    """Write the sample volume out as TIFF stack / DICOM series / RAW for import tests."""
    out_dir = Path(out_dir)
    written: dict[str, Any] = {}
    data = vol.data

    if "tiff" in formats:
        import tifffile

        d = out_dir / "tiff"
        d.mkdir(parents=True, exist_ok=True)
        lo, hi = float(data.min()), float(data.max())
        scaled = np.clip((data - lo) / max(float(np.ptp(data)), 1e-9) * 65535, 0, 65535).astype(np.uint16)
        # resolution is pixels per cm (TIFF has no mm unit); description JSON
        # carries the value scale + slice spacing back through the importer
        xres = 10.0 / vol.spacing[2]
        yres = 10.0 / vol.spacing[1]
        for i in range(data.shape[0]):
            tifffile.imwrite(d / f"slice_{i:04d}.tif", scaled[i],
                             resolution=(xres, yres), resolutionunit="CENTIMETER",
                             metadata={"spacing": vol.spacing[0],
                                       "value_lo": lo, "value_hi": hi,
                                       "value_unit": vol.unit})
        written["tiff"] = {"dir": str(d), "files": int(data.shape[0])}

    if "dicom" in formats:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        d = out_dir / "dicom"
        d.mkdir(parents=True, exist_ok=True)
        study_uid = generate_uid()
        series_uid = generate_uid()
        slope = 1.0
        intercept = 0.0
        for i in range(data.shape[0]):
            sl = data[i]
            q = np.clip((sl - intercept) / slope, -32768, 32767).astype(np.int16)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = Dataset()
            ds.file_meta = meta
            ds.SOPClassUID = pydicom.uid.CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.PatientName = "Synthetic^Core"
            ds.PatientID = "SYN-001"
            ds.Modality = "CT"
            ds.Manufacturer = "Simulated CT"
            ds.ManufacturerModelName = "Rockstone SynthScan"
            ds.SeriesDescription = "Rock core phantom"
            ds.StudyDescription = "Synthetic beam-hardening sample"
            ds.StudyDate = "20260101"
            ds.KVP = str(int(vol.metadata.get("kvp", 80)))
            ds.ConvolutionKernel = "FBP sim"
            ds.InstanceNumber = i + 1
            ds.SeriesNumber = 1
            ds.ImagePositionPatient = [-100.0, -100.0, float(i * vol.spacing[0])]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.SliceThickness = str(vol.spacing[0])
            ds.SpacingBetweenSlices = str(vol.spacing[0])
            ds.PixelSpacing = [vol.spacing[1], vol.spacing[2]]
            ds.Rows, ds.Columns = q.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleSlope = str(slope)
            ds.RescaleIntercept = str(intercept)
            ds.RescaleType = "HU"
            ds.PixelData = q.tobytes()
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.save_as(d / f"slice_{i:04d}.dcm", enforce_file_format=True)
        written["dicom"] = {"dir": str(d), "files": int(data.shape[0])}

    if "raw" in formats:
        d = out_dir / "raw"
        d.mkdir(parents=True, exist_ok=True)
        raw_path = d / "volume.raw"
        lo, hi = float(data.min()), float(data.max())
        q = np.clip((data - lo) / max(float(np.ptp(data)), 1e-9) * 65535, 0, 65535).astype(np.uint16)
        q.tofile(raw_path)
        layout = {"dtype": "uint16", "width": int(data.shape[2]), "height": int(data.shape[1]),
                  "depth": int(data.shape[0]), "header_bytes": 0, "byte_order": "<",
                  "spacing": list(vol.spacing), "value_lo": lo, "value_hi": hi,
                  "value_unit": vol.unit}
        (d / "volume.json").write_text(json.dumps(layout, indent=2))
        written["raw"] = {"file": str(raw_path), "layout": layout}

    return written


def carve_fractures(vol: Volume, seed: int = 7) -> Volume:
    """Return a copy of ``vol`` with a few thin air-filled fractures carved in.

    Two steep planar cracks cross the core with slightly rough walls (value
    noise), so the detector has ground truth to agree with.  Only touches the
    in-memory copy; the on-disk cache stays fracture-free.
    """
    rng = np.random.default_rng(seed + 991)
    data = vol.data.copy()
    z, y, x = data.shape
    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    matrix = data > -500.0  # rock matrix, excludes air + open pores
    truth: list[dict[str, Any]] = []
    for k in range(2):
        cy, cx = rng.uniform(0.35, 0.65) * y, rng.uniform(0.35, 0.65) * x
        # steep normal -> the crack cuts across most slices
        nx = float(rng.uniform(-0.35, 0.35))
        ny = float(rng.uniform(-0.35, 0.35))
        nz = float(rng.choice([-1.0, 1.0]))
        rough = (_value_noise((z, y), scale=12, rng=rng)[:, :, None] - 0.5) * 6.0
        d = (xx - cx) * nx + (yy - cy) * ny + (zz - z / 2) * nz + rough
        half = 0.7  # ~1.4 px wide: thin fracture
        crack = (np.abs(d) <= half) & matrix
        data[crack] = -1000.0 + rng.normal(0, 15, int(crack.sum())).astype(np.float32)
        truth.append({"center_px": [round(float(cy), 1), round(float(cx), 1)],
                      "normal": [round(nx, 3), round(ny, 3), nz],
                      "half_width_px": half, "voxels": int(crack.sum())})
    out = Volume(data=data, spacing=vol.spacing, name=vol.name + " (fractured)",
                 unit=vol.unit, origin=vol.origin)
    out.metadata = dict(vol.metadata)
    out.metadata["fractures"] = {"count": len(truth), "seed": seed + 991,
                                 "planes": truth}
    return out


def load_or_generate(shape: tuple[int, int, int] = (128, 288, 288), seed: int = 7,
                     fractures: bool = False) -> Volume:
    """Cached sample: generate once, reload from disk afterwards."""
    cache = Path("data/sample/cache")
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "meta.json"
    vol_path = cache / "volume.npy"
    if meta_path.exists() and vol_path.exists():
        meta = json.loads(meta_path.read_text())
        arr = np.load(vol_path)
        vol = Volume(data=arr, spacing=tuple(meta["spacing"]), metadata=meta["metadata"],
                     name=meta.get("name", "synthetic-rock-core"), unit=meta.get("unit", "HU"))
    else:
        vol, _ = generate(shape=shape, seed=seed)
        np.save(vol_path, vol.data)
        meta_path.write_text(json.dumps({"spacing": list(vol.spacing), "metadata": vol.metadata,
                                         "name": vol.name, "unit": vol.unit}, indent=2, default=str))
    return carve_fractures(vol, seed=seed) if fractures else vol
