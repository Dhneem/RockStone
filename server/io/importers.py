"""Dataset importers: DICOM series, TIFF/PNG/JPEG stacks and RAW volumes.

Every importer returns a :class:`~server.store.Volume` whose metadata contains
whatever could be detected automatically (slice spacing, pixel size, image
dimensions and scanner/reconstruction metadata) plus a `detections` dict that
describes the provenance of each value so the UI can flag assumed values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from server.store import Volume

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(name: str) -> list[Any]:
    """Sort helper so slice2 sorts before slice10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _detect(files: list[Path], kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Uniform 'detection' report: value + how it was obtained."""
    det: dict[str, Any] = {}
    for key, value in (extra or {}).items():
        det[key] = value
    det.setdefault("source", kind)
    return det


# --------------------------------------------------------------------- DICOM
def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(132)
        return len(head) >= 132 and head[128:132] == b"DICM"
    except OSError:
        return False


def load_dicom(paths: list[Path]) -> Volume:
    """Load a DICOM series (folder or explicit file list) and stack it."""
    import pydicom

    files = [p for p in paths if p.is_file()]
    if not files:
        raise ValueError("no files given")

    datasets = []
    for p in sorted(files):
        try:
            if not is_dicom(p):
                continue
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=False)
            if hasattr(ds, "PixelData"):
                datasets.append((p, ds))
        except Exception:
            continue
    if not datasets:
        raise ValueError("no readable DICOM files with pixel data found")

    # group by series so a folder with several studies still works
    series: dict[str, list[tuple[Path, Any]]] = {}
    for p, ds in datasets:
        key = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        series.setdefault(key, []).append((p, ds))
    key = max(series, key=lambda k: len(series[k]))
    entries = series[key]

    def sort_key(item: tuple[Path, Any]):
        _, ds = item
        ipp = getattr(ds, "ImagePositionPatient", None)
        orient = getattr(ds, "ImageOrientationPatient", None)
        if ipp is not None and orient is not None:
            ipp = [float(v) for v in ipp]
            row = [float(v) for v in orient[:3]]
            col = [float(v) for v in orient[3:6]]
            normal = [
                row[1] * col[2] - row[2] * col[1],
                row[2] * col[0] - row[0] * col[2],
                row[0] * col[1] - row[1] * col[0],
            ]
            pos = sum(a * b for a, b in zip(ipp, normal))
            return (0, pos)
        inst = getattr(ds, "InstanceNumber", None)
        try:
            return (1, float(inst))
        except (TypeError, ValueError):
            return (2, natural_key(item[0].name))  # type: ignore[arg-type]

    entries.sort(key=sort_key)

    slices = []
    for _, ds in entries:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            arr = arr * slope + intercept
        slices.append(arr)

    vol = np.stack(slices, axis=0).astype(np.float32)
    ds0 = entries[0][1]

    # --- pixel size
    pixel_spacing = getattr(ds0, "PixelSpacing", None)
    if pixel_spacing is not None:
        dy, dx = float(pixel_spacing[0]), float(pixel_spacing[1])
        spacing_source = "dicom:PixelSpacing"
    else:
        dy = dx = 1.0
        spacing_source = "assumed (no PixelSpacing)"

    # --- slice spacing: prefer the real gap between ImagePositionPatient
    dz = None
    dz_source = "assumed"
    if len(entries) > 1:
        positions = []
        for _, ds in entries:
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None:
                positions.append(float(ipp[2]))
        if len(positions) == len(entries) and len(set(positions)) > 1:
            diffs = np.diff(sorted(positions))
            dz = float(np.median(diffs))
            dz_source = "dicom:ImagePositionPatient"
    if dz is None:
        st = getattr(ds0, "SliceThickness", None)
        if st:
            dz = float(st)
            dz_source = "dicom:SliceThickness"
    if dz is None or dz <= 0:
        ss = getattr(ds0, "SpacingBetweenSlices", None)
        if ss:
            dz = float(ss)
            dz_source = "dicom:SpacingBetweenSlices"
    if dz is None or dz <= 0:
        dz = dy
        dz_source = "assumed (= pixel size)"

    meta = {
        "format": "DICOM",
        "modality": str(getattr(ds0, "Modality", "") or ""),
        "manufacturer": str(getattr(ds0, "Manufacturer", "") or ""),
        "manufacturer_model": str(getattr(ds0, "ManufacturerModelName", "") or ""),
        "station": str(getattr(ds0, "StationName", "") or ""),
        "kvp": _num(getattr(ds0, "KVP", None)),
        "exposure": _num(getattr(ds0, "Exposure", None)),
        "convolution_kernel": str(getattr(ds0, "ConvolutionKernel", "") or ""),
        "reconstruction_diameter": _num(getattr(ds0, "ReconstructionDiameter", None)),
        "slice_thickness": _num(getattr(ds0, "SliceThickness", None)),
        "rescale_slope": _num(getattr(ds0, "RescaleSlope", None)) or 1.0,
        "rescale_intercept": _num(getattr(ds0, "RescaleIntercept", None)) or 0.0,
        "series_description": str(getattr(ds0, "SeriesDescription", "") or ""),
        "study_description": str(getattr(ds0, "StudyDescription", "") or ""),
        "study_date": str(getattr(ds0, "StudyDate", "") or ""),
        "patient_id": str(getattr(ds0, "PatientID", "") or ""),
        "image_type": str(getattr(ds0, "ImageType", "") or ""),
        "series_uid": str(getattr(ds0, "SeriesInstanceUID", "") or ""),
        "n_series_found": len(series),
        "bits_stored": _num(getattr(ds0, "BitsStored", None)),
        "samples_per_pixel": _num(getattr(ds0, "SamplesPerPixel", None)) or 1,
        "photometric": str(getattr(ds0, "PhotometricInterpretation", "") or ""),
        "number_of_frames": 1,
    }
    detections = _detect([], "dicom", {
        "slice_spacing": {"value": dz, "source": dz_source},
        "pixel_size": {"value": (dy, dx), "source": spacing_source},
        "image_dimensions": {"value": [vol.shape[2], vol.shape[1], vol.shape[0]], "source": "measured"},
        "scanner_metadata": {"value": "manufacturer/model/kernel/kVp from DICOM tags", "source": "dicom:tags"},
    })
    meta["detections"] = detections
    meta["spacing_source"] = {"z": dz_source, "y": spacing_source, "x": spacing_source}

    return Volume(data=vol, spacing=(dz, dy, dx), metadata=meta,
                  name=meta["series_description"] or "dicom-series", unit="HU")


def _num(v: Any) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------------ TIFF / PNG / JPEG
def load_stack(paths: list[Path], spacing: tuple[float, float, float] | None = None) -> Volume:
    """Load a stack of 2-D images, or a single multi-page TIFF."""
    files = [p for p in paths if p.is_file()]
    if not files:
        raise ValueError("no image files given")
    files.sort(key=lambda p: natural_key(p.name))

    planes: list[np.ndarray] = []
    tiff_meta: dict[str, Any] = {}
    tifffile = None
    try:
        import tifffile as _tifffile

        tifffile = _tifffile
    except Exception:
        pass

    # multi-page tiff?
    if len(files) == 1 and tifffile is not None and files[0].suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(str(files[0])) as tf:
            pages = tf.pages
            for page in pages:
                planes.append(np.asarray(page.asarray()).astype(np.float32))
            series = tf.series[0] if tf.series else None
            if series is not None:
                shape = tuple(series.shape)
                tiff_meta["tiff_shape"] = list(shape)
                axes = getattr(series, "axes", "")
                tiff_meta["tiff_axes"] = str(axes)
            first = pages[0]
            for tag_name in ("XResolution", "YResolution", "ResolutionUnit"):
                try:
                    tiff_meta[tag_name] = str(first.tags[tag_name].value)
                except Exception:
                    pass
            # value scale embedded by our exporter (ImageDescription JSON)
            desc = getattr(first, "description", "") or ""
            try:
                md = json.loads(desc)
                if isinstance(md, dict) and "value_lo" in md and "value_hi" in md:
                    tiff_meta["value_range"] = {"lo": float(md["value_lo"]),
                                                "hi": float(md["value_hi"]),
                                                "unit": str(md.get("value_unit", "intensity"))}
                    try:
                        dz = float(md.get("spacing", 0.0))
                        if dz > 0:
                            tiff_meta["slice_spacing"] = dz
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass
    else:
        from PIL import Image

        for p in files:
            with Image.open(p) as im:
                if im.mode not in ("I;16", "I;16B", "I", "F", "L", "I;16L"):
                    im = im.convert("F")
                planes.append(np.asarray(im).astype(np.float32))
            if tifffile is not None and p.suffix.lower() in {".tif", ".tiff"}:
                try:
                    with tifffile.TiffFile(str(p)) as tf:
                        page = tf.pages[0]
                        for tag_name in ("XResolution", "YResolution", "ResolutionUnit"):
                            try:
                                tiff_meta.setdefault(tag_name, str(page.tags[tag_name].value))
                            except Exception:
                                pass
                        # value scale / slice spacing embedded as JSON (ImageDescription)
                        desc = getattr(page, "description", "") or ""
                        md = json.loads(desc)
                        if isinstance(md, dict):
                            if "value_lo" in md and "value_hi" in md and "value_range" not in tiff_meta:
                                tiff_meta["value_range"] = {
                                    "lo": float(md["value_lo"]),
                                    "hi": float(md["value_hi"]),
                                    "unit": str(md.get("value_unit", "intensity")),
                                }
                            dz_json = float(md.get("spacing", 0.0) or 0.0)
                            if dz_json > 0 and "slice_spacing" not in tiff_meta:
                                tiff_meta["slice_spacing"] = dz_json
                except Exception:
                    pass

    shapes = {p.shape for p in planes}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent slice sizes in stack: {sorted(shapes)}")
    vol = np.stack(planes, axis=0).astype(np.float32)

    dz, dy, dx = 1.0, 1.0, 1.0
    source = {"z": "assumed (1.0)", "y": "assumed (1.0)", "x": "assumed (1.0)"}
    if tiff_meta.get("slice_spacing", 0) > 0:
        dz = float(tiff_meta["slice_spacing"])
        source["z"] = "tiff:ImageDescription JSON"

    # TIFF resolution tags -> pixel size (ResolutionUnit 2 = inch, 3 = cm)
    if "XResolution" in tiff_meta and "YResolution" in tiff_meta:
        try:
            xr = _ratio(tiff_meta["XResolution"])
            yr = _ratio(tiff_meta["YResolution"])
            unit = tiff_meta.get("ResolutionUnit", "2")
            unit = unit.strip("() ").split(",")[0]
            per = 25.4 if unit in ("2", "inch") else 10.0 if unit in ("3", "cm") else 25.4
            if xr > 0 and yr > 0:
                dx = per / xr
                dy = per / yr
                source["x"] = source["y"] = "tiff:Resolution tags"
        except Exception:
            pass

    if spacing is not None:
        dz, dy, dx = spacing
        source = {"z": "user", "y": "user", "x": "user"}

    unit = "intensity"
    vr = tiff_meta.get("value_range")
    if vr:
        # exporter stored uint16 quantisation of the real values
        lo, hi = float(vr["lo"]), float(vr["hi"])
        if hi - lo > 1e-12:
            vol = vol / 65535.0 * (hi - lo) + lo
        unit = str(vr.get("unit", "intensity"))
        tiff_meta["value_scale_restored"] = True

    meta = {
        "format": "TIFF/PNG/JPEG stack",
        "files": len(files),
        "first_file": files[0].name,
        "dtype_in": str(planes[0].dtype),
        "tiff": tiff_meta,
        "detections": _detect([], "stack", {
            "slice_spacing": {"value": dz, "source": source["z"]},
            "pixel_size": {"value": (dy, dx), "source": source["x"]},
            "image_dimensions": {"value": [vol.shape[2], vol.shape[1], vol.shape[0]], "source": "measured"},
            "scanner_metadata": {"value": None, "source": "not available for image stacks"},
        }),
        "spacing_source": source,
    }
    return Volume(data=vol.astype(np.float32), spacing=(dz, dy, dx), metadata=meta,
                  name=files[0].stem.rsplit("_", 1)[0] or "stack", unit=unit)


def _ratio(v: str) -> float:
    parts = v.strip("() ").split(",")
    a, b = (float(parts[0]), float(parts[1])) if len(parts) >= 2 else (float(parts[0]), 1.0)
    return a / b if b else 0.0


# --------------------------------------------------------------------- RAW
RAW_DTYPES = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}


def guess_raw_layout(file_size: int, dtype: str, header_bytes: int = 0) -> list[dict[str, Any]]:
    """Guess plausible (width, height, depth) layouts for a headerless RAW file."""
    dt = np.dtype(RAW_DTYPES[dtype])
    n = (file_size - header_bytes) // dt.itemsize
    if n <= 0:
        return []
    candidates = []
    for side in (256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536, 2048):
        if n % (side * side) == 0:
            depth = n // (side * side)
            if 1 <= depth <= 8192:
                candidates.append({"width": side, "height": side, "depth": depth, "confidence": 0.9 if side in (512, 1024, 256) else 0.6})
    if not candidates:
        # fall back: try plausible rectangular (width, height) pairs
        sides = (32, 48, 64, 96, 128, 144, 160, 180, 200, 256, 320, 384, 512, 768, 1024)
        for w in sides:
            for h in sides:
                if w * h <= n and n % (w * h) == 0:
                    depth = n // (w * h)
                    if 1 <= depth <= 8192:
                        candidates.append({"width": w, "height": h, "depth": depth,
                                           "confidence": 0.4 if w == h else 0.3})
    candidates.sort(key=lambda c: (-c["confidence"], -c["depth"]))
    return candidates[:6]


def load_raw(
    path: Path,
    dtype: str = "uint16",
    width: int = 512,
    height: int = 512,
    depth: int | None = None,
    header_bytes: int = 0,
    byte_order: str = "<",
    spacing: tuple[float, float, float] | None = None,
    endian_big: bool = False,
) -> Volume:
    if dtype not in RAW_DTYPES:
        raise ValueError(f"unsupported RAW dtype '{dtype}'")
    dt = np.dtype(RAW_DTYPES[dtype]).newbyteorder(">" if endian_big else byte_order)
    file_size = path.stat().st_size
    usable = (file_size - header_bytes) // dt.itemsize
    if depth is None:
        if width * height <= 0:
            raise ValueError("bad width/height")
        depth = usable // (width * height)
    need = width * height * depth
    if need > usable:
        raise ValueError(f"file too small: need {need} values, has {usable}")
    arr = np.fromfile(path, dtype=dt, count=need, offset=header_bytes)
    vol = arr.reshape(depth, height, width).astype(np.float32)

    dz, dy, dx = spacing if spacing is not None else (1.0, 1.0, 1.0)
    guesses = guess_raw_layout(file_size, dtype, header_bytes)
    meta = {
        "format": "RAW",
        "dtype": dtype,
        "byte_order": ">" if (endian_big or byte_order == ">") else "<",
        "header_bytes": header_bytes,
        "file_size": file_size,
        "file_name": path.name,
        "raw_layout_guesses": guesses,
        "detections": _detect([], "raw", {
            "slice_spacing": {"value": dz, "source": "user" if spacing else "assumed (1.0)"},
            "pixel_size": {"value": (dy, dx), "source": "user" if spacing else "assumed (1.0)"},
            "image_dimensions": {"value": [width, height, depth], "source": "declared by user"},
            "scanner_metadata": {"value": None, "source": "not available for RAW files"},
        }),
        "spacing_source": {"z": "user" if spacing else "assumed", "y": "user" if spacing else "assumed",
                           "x": "user" if spacing else "assumed"},
    }
    return Volume(data=vol, spacing=(dz, dy, dx), metadata=meta,
                  name=path.stem, unit="intensity")


# --------------------------------------------------------------- entrypoint
def import_path(
    path: str,
    kind: str = "auto",
    raw_params: dict[str, Any] | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> Volume:
    """Import a folder or file, auto-detecting the format unless `kind` says otherwise."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"path not found: {path}")
    raw_params = raw_params or {}

    if p.is_file():
        files = [p]
    else:
        files = [f for f in sorted(p.rglob("*")) if f.is_file()]

    if kind == "auto":
        dicom_files = [f for f in files if f.suffix.lower() == ".dcm" or is_dicom(f)]
        if dicom_files:
            kind = "dicom"
        elif any(f.suffix.lower() in IMAGE_EXTS for f in files):
            kind = "stack"
        elif (p.suffix.lower() in {".raw", ".vol", ".bin", ".dat"} or p.is_file()
              or any(f.suffix.lower() in {".raw", ".vol", ".bin", ".dat"} for f in files)):
            kind = "raw"
        else:
            raise ValueError(f"could not detect format of {path}")

    if kind == "dicom":
        dcm_files = [f for f in files if is_dicom(f)] or files
        return load_dicom(dcm_files)
    if kind == "stack":
        imgs = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
        return load_stack(imgs, spacing=spacing)
    if kind == "raw":
        if p.is_dir():
            raws = [f for f in files if f.suffix.lower() in {".raw", ".vol", ".bin", ".dat"}]
            if not raws:
                raise ValueError("no .raw/.vol/.bin file inside the folder")
            p = raws[0]
        # sidecar layout JSON (written next to the file) -> exact dimensions,
        # spacing and value scale; explicit raw_params always win
        sidecar: dict[str, Any] = {}
        for cand in (p.with_suffix(".json"), p.parent / "layout.json", p.parent / "volume.json"):
            if cand.is_file():
                try:
                    sidecar = json.loads(cand.read_text())
                    if isinstance(sidecar, dict):
                        break
                except (OSError, ValueError):
                    sidecar = {}
        for key in ("dtype", "width", "height", "depth", "header_bytes", "byte_order"):
            if key in sidecar and key not in raw_params:
                raw_params[key] = sidecar[key]
        if spacing is None and sidecar.get("spacing"):
            spacing = tuple(float(s) for s in sidecar["spacing"])
        vol = load_raw(
            p,
            dtype=raw_params.get("dtype", "uint16"),
            width=int(raw_params.get("width", 512)),
            height=int(raw_params.get("height", 512)),
            depth=raw_params.get("depth"),
            header_bytes=int(raw_params.get("header_bytes", 0)),
            byte_order=raw_params.get("byte_order", "<"),
            spacing=spacing,
            endian_big=bool(raw_params.get("endian_big", False)),
        )
        if sidecar:
            src = "raw:sidecar JSON"
            det = vol.metadata["detections"]
            if spacing is not None:
                det["slice_spacing"]["source"] = src
                det["pixel_size"]["source"] = src
                vol.metadata["spacing_source"] = {"z": src, "y": src, "x": src}
            if "width" in sidecar or "height" in sidecar:
                det["image_dimensions"]["source"] = src
            if "value_lo" in sidecar and "value_hi" in sidecar:
                lo, hi = float(sidecar["value_lo"]), float(sidecar["value_hi"])
                if hi - lo > 1e-12:
                    vol.data = (vol.data / 65535.0 * (hi - lo) + lo).astype(np.float32)
                    vol.unit = str(sidecar.get("value_unit", vol.unit))
                    vol.metadata["value_scale_restored"] = True
            if "value_unit" in sidecar:
                vol.unit = str(sidecar["value_unit"])
        return vol
    raise ValueError(f"unknown format kind '{kind}'")
