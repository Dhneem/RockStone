"""Volume exporters: TIFF/PNG stacks, RAW, NRRD, CSV statistics, JSON metadata."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from server.store import Volume


def _q8(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = float(np.nanmin(a)) if lo is None else lo
    hi = float(np.nanmax(a)) if hi is None else hi
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def export_tiff_stack(vol: Volume, level: tuple[float, float] | None = None) -> bytes:
    import tifffile

    buf = io.BytesIO()
    lo, hi = level if level else (float(vol.data.min()), float(vol.data.max()))
    scaled = np.clip((vol.data.astype(np.float64) - lo) / max(hi - lo, 1e-12), 0, 1)
    scaled = (scaled * 65535.0).astype(np.uint16)
    # TIFF has no mm resolution unit: encode pixels-per-cm so the importer
    # can recover the true pixel size (ResolutionUnit 3 = cm)
    xres = 10.0 / max(vol.spacing[2], 1e-9)
    yres = 10.0 / max(vol.spacing[1], 1e-9)
    with tifffile.TiffWriter(buf, bigtiff=False) as tw:
        tw.write(
            scaled,
            photometric="minisblack",
            resolution=(xres, yres),
            resolutionunit="CENTIMETER",
            metadata={
                "axes": "ZYX",
                "spacing": vol.spacing[0],
                "unit": "mm",
                "resolution": (1.0 / vol.spacing[2], 1.0 / vol.spacing[1], "mm"),
                # so the importer can restore the original value scale
                "value_lo": lo,
                "value_hi": hi,
                "value_unit": vol.unit,
            },
        )
    return buf.getvalue()


def export_png_zip(vol: Volume, level: tuple[float, float] | None = None) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    lo, hi = level if level else (float(vol.data.min()), float(vol.data.max()))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(vol.shape[0]):
            img = Image.fromarray(_q8(vol.data[i], lo, hi))
            b = io.BytesIO()
            img.save(b, format="PNG")
            zf.writestr(f"slice_{i:05d}.png", b.getvalue())
    return buf.getvalue()


def export_raw(vol: Volume, dtype: str = "float32", endian_big: bool = False) -> bytes:
    dt = np.dtype(dtype).newbyteorder(">" if endian_big else "<")
    return np.ascontiguousarray(vol.data.astype(dt)).tobytes()


def export_nrrd(vol: Volume, dtype: str = "float32", endian_big: bool = False) -> bytes:
    """Minimal NRRD-0.5 writer (attached data, little or big endian)."""
    arr = vol.data.astype(np.dtype(dtype))
    sz, sy, sx = vol.spacing
    header = (
        "NRRD0005\n"
        f"type: {'float' if dtype == 'float32' else dtype.replace('32', '32').replace('64', '64')}\n"
        "dimension: 3\n"
        "space: left-posterior-superior\n"
        f"sizes: {arr.shape[2]} {arr.shape[1]} {arr.shape[0]}\n"
        "space directions: (1,0,0) (0,1,0) (0,0,1)\n"
        "kinds: domain domain domain\n"
        f"endian: {'big' if endian_big else 'little'}\n"
        "encoding: raw\n"
        "space origin: (0,0,0)\n"
        f"# voxel size mm: x={sx} y={sy} z={sz}\n"
        "\n"
    )
    data = np.ascontiguousarray(arr.astype(np.dtype(dtype).newbyteorder(">" if endian_big else "<")))
    return header.encode("ascii") + data.tobytes()


def export_stats_csv(stats: dict[str, Any]) -> str:
    lines = ["section,key,value"]
    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (list, tuple)):
            lines.append(f'"{prefix}","{json.dumps(list(obj))}"')
        else:
            lines.append(f'"{prefix}","{obj}"')
    walk("", stats)
    return "\n".join(lines)


def export_json(meta: dict[str, Any]) -> str:
    def conv(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(meta, indent=2, default=conv)
