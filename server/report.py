"""Core-CT report plate: a single-page figure rendered to PNG.

Recreates the classic core-scanner layout in one image:

* stacked per-depth CT-distribution traces (with driller's depth),
* a histogram map (depth vs value, colour = frequency),
* horizontal and vertical slabs through the core centre,
* a grid of colour-coded cross-sections,
* a CT colour bar with an approximate-density scale for HU data.

Only numpy + Pillow are used (both project dependencies); every helper is
kept pure so the plate is unit-testable without a running server.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------- colormaps
# classic "jet" control points (blue -> cyan -> yellow -> red)
_JET_STOPS = [
    (0.000, (0, 0, 131)),
    (0.125, (0, 0, 255)),
    (0.375, (0, 255, 255)),
    (0.625, (255, 255, 0)),
    (0.875, (255, 0, 0)),
    (1.000, (128, 0, 0)),
]

_W, _H = 1800, 1250
_MARGIN = 24
_BAND_H = 40
_BAND_Y = 96
_PLOT_TOP = _BAND_Y + _BAND_H + 12
_PLOT_BOT = 1060

# panel x-ranges: traces | map | slabs | slice grid
_PANELS = {
    "traces": (_MARGIN, _MARGIN + 470),
    "map": (_MARGIN + 490, _MARGIN + 870),
    "slabs": (_MARGIN + 890, _MARGIN + 1330),
    "grid": (_MARGIN + 1350, _MARGIN + 1752),
}

_BG = (255, 255, 255)
_BAND_BG = (184, 212, 232)
_BAND_FG = (24, 48, 66)
_AXIS = (60, 70, 80)
_GRIDLINE = (225, 232, 238)
_TEXT = (30, 36, 42)
_MUTE = (110, 120, 130)
_DARK_BG = (6, 10, 16)
_TRACE_FILL = (111, 183, 201)
_TRACE_LINE = (36, 106, 124)


def jet_lut(n: int = 256) -> np.ndarray:
    """(n, 3) uint8 jet lookup table."""
    xs = np.linspace(0.0, 1.0, n)
    stops_x = [s[0] for s in _JET_STOPS]
    cols = np.stack([
        np.interp(xs, stops_x, [s[1][c] for s in _JET_STOPS]) for c in range(3)
    ], axis=1)
    return np.clip(cols, 0, 255).astype(np.uint8)


def colorize(v01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map an array of values in [0, 1] to RGB uint8 through the LUT."""
    idx = np.clip(np.nan_to_num(v01, nan=0.0), 0.0, 1.0) * (len(lut) - 1)
    return lut[idx.astype(np.uint8)]


def density_scale(unit: str | None, lo: float, hi: float,
                  n: int = 5) -> list[tuple[float, float]] | None:
    """Approximate-density tick pairs (value, g/cm^3) for HU volumes.

    Uses the standard two-point relation rho ~= 1 + HU/1000 (mu ~ density);
    returns None for non-HU volumes.
    """
    if (unit or "intensity").strip().lower() != "hu":
        return None
    return [(float(v), 1.0 + float(v) / 1000.0) for v in np.linspace(lo, hi, n)]


# ------------------------------------------------------------------ helpers
_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_BOLD_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_CACHE: dict[tuple[int, bool], Any] = {}


def _font(size: int, bold: bool = False) -> Any:
    key = (size, bold)
    if key not in _FONT_CACHE:
        f = None
        for p in (_FONT_BOLD_PATHS if bold else _FONT_PATHS):
            try:
                f = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
        if f is None:
            try:
                f = ImageFont.load_default(size=size)  # Pillow >= 10.1
            except TypeError:
                f = ImageFont.load_default()
        _FONT_CACHE[key] = f
    return _FONT_CACHE[key]


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    return [float(v) for v in np.linspace(lo, hi, n)]


def _band(img: Image.Image, draw: ImageDraw.ImageDraw, x0: int, x1: int,
          title: str) -> None:
    draw.rectangle([x0, _BAND_Y, x1, _BAND_Y + _BAND_H], fill=_BAND_BG,
                   outline=(120, 150, 172))
    draw.text((x0 + 12, _BAND_Y + (_BAND_H - 22) // 2), title,
              font=_font(17, bold=True), fill=_BAND_FG)


def _border(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(list(box), outline=(150, 160, 170))


def _paste(img: Image.Image, im: Image.Image, xy: tuple[int, int]) -> None:
    img.paste(im.convert("RGB"), xy)


def _window(arr: np.ndarray, lo: float | None, hi: float | None) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        base_lo, base_hi = 0.0, 1.0
    else:
        base_lo, base_hi = (float(v) for v in np.percentile(finite, [1.0, 99.0]))
    out_lo = float(lo) if lo is not None else base_lo
    out_hi = float(hi) if hi is not None else base_hi
    if out_hi - out_lo < 1e-6:
        out_hi = out_lo + 1.0
    return out_lo, out_hi


def _level_slices(z: int, levels: int) -> list[np.ndarray]:
    """Index arrays grouping the stack into ~levels depth bins."""
    levels = max(2, min(levels, z))
    edges = np.linspace(0, z, levels + 1).round().astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = max(b, a + 1)
        out.append(np.arange(a, min(b, z)))
    return out


# --------------------------------------------------------------- sub-plates
def _depth_traces(img: Image.Image, draw: ImageDraw.ImageDraw, data: np.ndarray,
                  lo: float, hi: float, levels: int, bins: int,
                  spacing_z: float, unit: str) -> None:
    """Stacked per-depth CT-distribution traces (driller's depth axis)."""
    x0, x1 = _PANELS["traces"]
    px0, px1 = x0 + 64, x1 - 12
    py0, py1 = _PLOT_TOP + 8, _PLOT_BOT - 46

    edges = np.linspace(lo, hi, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    groups = _level_slices(data.shape[0], levels)

    hists: list[np.ndarray] = []
    depths: list[float] = []
    max_all = 1.0
    for idx in groups:
        block = data[idx.min():idx.max() + 1]
        flat = block[:: max(1, block.shape[0] // 2), ::3, ::3].ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            flat = np.array([lo])
        h, _ = np.histogram(flat, bins=edges)
        hists.append(h.astype(np.float64))
        depths.append(float(idx.mean() + 0.5) * spacing_z)
        max_all = max(max_all, float(h.max()))

    n = len(hists)
    row_h = (py1 - py0) / max(n, 1)
    amp = row_h * 1.9          # traces overlap slightly, like the scanner plate
    xs = px0 + (centres - lo) / max(hi - lo, 1e-12) * (px1 - px0)

    for t in _ticks(lo, hi):
        gx = px0 + (t - lo) / max(hi - lo, 1e-12) * (px1 - px0)
        draw.line([gx, py0, gx, py1], fill=_GRIDLINE, width=1)

    # draw deepest first so shallower traces overlap on top
    for i in range(n - 1, -1, -1):
        base_y = py0 + (i + 1) * row_h
        prof = hists[i] / max_all * amp
        pts = [(float(xs[j]), float(base_y - prof[j])) for j in range(bins)]
        poly = [(float(px0), base_y)] + pts + [(float(px1), base_y)]
        draw.polygon(poly, fill=_TRACE_FILL)
        draw.line(pts, fill=_TRACE_LINE, width=1)

    draw.line([px0, py1, px1, py1], fill=_AXIS, width=2)
    draw.line([px0, py0, px0, py1], fill=_AXIS, width=2)
    f_tick = _font(12)
    for t in _ticks(lo, hi):
        gx = px0 + (t - lo) / max(hi - lo, 1e-12) * (px1 - px0)
        draw.line([gx, py1, gx, py1 + 5], fill=_AXIS, width=1)
        lbl = _fmt(t)
        draw.text((gx - draw.textlength(lbl, font=f_tick) / 2, py1 + 8), lbl,
                  font=f_tick, fill=_TEXT)
    depth_max = max(depths[-1] if depths else 1.0, 1e-12)
    for t in _ticks(0.0, depth_max, 6):
        gy = py1 - (t / depth_max) * (py1 - py0)
        draw.line([px0 - 5, gy, px0, gy], fill=_AXIS, width=1)
        lbl = _fmt(t)
        draw.text((px0 - 10 - draw.textlength(lbl, font=f_tick), gy - 7), lbl,
                  font=f_tick, fill=_TEXT)
    axis_lbl = f"CT distribution ({unit})"
    draw.text(((px0 + px1) / 2 - draw.textlength(axis_lbl, font=_font(13)) / 2, py1 + 26),
              axis_lbl, font=_font(13), fill=_TEXT)
    for k, ch in enumerate("depth (mm)"):
        draw.text((x0 + 18, py0 + 10 + k * 13), ch, font=_font(11), fill=_MUTE)


def _hist_map(data: np.ndarray, lo: float, hi: float, levels: int,
              bins: int) -> np.ndarray:
    """(levels, bins, 3) uint8 — depth vs value, jet colour = log frequency."""
    edges = np.linspace(lo, hi, bins + 1)
    rows = []
    peak = 1.0
    for idx in _level_slices(data.shape[0], levels):
        block = data[idx.min():idx.max() + 1]
        flat = block[:: max(1, block.shape[0] // 2), ::2, ::2].ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            flat = np.array([lo])
        h, _ = np.histogram(flat, bins=edges)
        rows.append(h.astype(np.float64))
        peak = max(peak, float(h.max()))
    counts = np.stack(rows)
    v = np.log1p(counts) / np.log1p(peak)
    rgb = colorize(v, jet_lut())
    rgb[counts <= 0] = _DARK_BG
    return rgb.astype(np.uint8)


def _map_panel(img: Image.Image, draw: ImageDraw.ImageDraw, data: np.ndarray,
               lo: float, hi: float, levels: int, bins: int,
               spacing_z: float) -> None:
    x0, x1 = _PANELS["map"]
    px0, px1 = x0 + 34, x1 - 12
    py0, py1 = _PLOT_TOP + 8, _PLOT_BOT - 46

    arr = _hist_map(data, lo, hi, levels, bins)
    im = Image.fromarray(arr).resize((px1 - px0, py1 - py0), Image.BILINEAR)
    _paste(img, im, (px0, py0))
    _border(draw, (px0, py0, px1, py1))

    f_tick = _font(12)
    for t in _ticks(lo, hi):
        gx = px0 + (t - lo) / max(hi - lo, 1e-12) * (px1 - px0)
        lbl = _fmt(t)
        draw.text((gx - draw.textlength(lbl, font=f_tick) / 2, py1 + 8), lbl,
                  font=f_tick, fill=_TEXT)
    depth_max = data.shape[0] * spacing_z
    for t in _ticks(0.0, depth_max, 6):
        gy = py1 - (t / max(depth_max, 1e-12)) * (py1 - py0)
        lbl = _fmt(t)
        draw.text((px0 - 10 - draw.textlength(lbl, font=f_tick), gy - 7), lbl,
                  font=f_tick, fill=_TEXT)

    # frequency colour bar below the map ("Higher Frequency >>>")
    bx0, bx1, by = px0, min(px0 + 300, px1), _PLOT_BOT + 26
    grad = colorize(np.linspace(0, 1, bx1 - bx0)[None, :], jet_lut())[0]
    _paste(img, Image.fromarray(np.tile(grad, (16, 1, 1))), (bx0, by))
    _border(draw, (bx0, by, bx1, by + 16))
    draw.text((bx0, by + 22), "Higher Frequency >>>", font=_font(12), fill=_TEXT)


def _slabs(img: Image.Image, draw: ImageDraw.ImageDraw, data: np.ndarray,
           lo: float, hi: float) -> None:
    """Vertical (XZ) + horizontal (XY) slabs through the core centre."""
    x0, x1 = _PANELS["slabs"]
    top, bot = _PLOT_TOP + 8, _PLOT_BOT - 20
    gap = 16
    cell_w = (x1 - x0 - gap) // 2
    z, y, _x = data.shape
    lut = jet_lut()

    def _rgb(plane: np.ndarray, box_x: int) -> None:
        q = np.clip((np.nan_to_num(plane, nan=lo) - lo) / max(hi - lo, 1e-12), 0, 1)
        im = Image.fromarray(colorize(q, lut))
        bw, bh = cell_w, bot - top
        scale = min(bw / im.width, bh / im.height)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.BILINEAR)
        ox = box_x + (cell_w - im.width) // 2
        oy = top + (bot - top - im.height) // 2
        _paste(img, im, (ox, oy))
        _border(draw, (ox, oy, ox + im.width, oy + im.height))

    _rgb(data[:, y // 2, :], x0)                    # vertical slab: XZ at mid-Y
    _rgb(data[z // 2, :, :], x0 + cell_w + gap)     # horizontal slab: XY at mid-depth

    draw.text((x0, bot + 6), "vertical (XZ @ mid-Y)", font=_font(12), fill=_MUTE)
    draw.text((x0 + cell_w + gap, bot + 6), "horizontal (XY @ mid-Z)",
              font=_font(12), fill=_MUTE)


def _slice_grid(img: Image.Image, draw: ImageDraw.ImageDraw, data: np.ndarray,
                lo: float, hi: float, count: int, cols: int,
                spacing_z: float) -> None:
    """Grid of colour-coded cross-sections on a dark ground, like the plate."""
    x0, x1 = _PANELS["grid"]
    top, bot = _PLOT_TOP, _PLOT_BOT - 20
    draw.rectangle([x0, top, x1, bot], fill=_DARK_BG)

    z = data.shape[0]
    count = max(1, min(count, z))
    idxs = np.unique(np.linspace(0, z - 1, count).round().astype(int))
    rows = int(np.ceil(len(idxs) / cols))
    pad = 4
    cell = min((x1 - x0 - pad * (cols + 1)) // cols,
               (bot - top - pad * (rows + 1)) // rows)
    grid_w = cell * cols + pad * (cols + 1)
    grid_h = cell * rows + pad * (rows + 1)
    ox = x0 + ((x1 - x0) - grid_w) // 2
    oy = top + ((bot - top) - grid_h) // 2

    lut = jet_lut()
    f_lab = _font(10)
    for k, zi in enumerate(idxs):
        r, c = divmod(k, cols)
        cx = ox + pad + c * (cell + pad)
        cy = oy + pad + r * (cell + pad)
        q = np.clip((np.nan_to_num(data[zi], nan=lo) - lo) / max(hi - lo, 1e-12), 0, 1)
        im = Image.fromarray(colorize(q, lut)).resize((cell, cell), Image.BILINEAR)
        _paste(img, im, (cx, cy))
        draw.text((cx + 3, cy + 3), f"{zi * spacing_z:.0f}mm", font=f_lab,
                  fill=(210, 225, 240))

    for r in range(rows + 1):
        gy = oy + pad + r * (cell + pad) - pad // 2
        if oy <= gy <= bot:
            draw.line([x0 + 4, gy, x1 - 4, gy], fill=(40, 52, 64), width=1)
    for c in range(cols + 1):
        gx = ox + pad + c * (cell + pad) - pad // 2
        if x0 <= gx <= x1:
            draw.line([gx, oy, gx, oy + grid_h], fill=(40, 52, 64), width=1)
    _border(draw, (x0, top, x1, bot))


def _value_bar(img: Image.Image, draw: ImageDraw.ImageDraw, lo: float, hi: float,
               unit: str) -> None:
    """CT colour bar + approximate density scale (bottom right)."""
    bx0, bx1 = _PANELS["grid"]
    by = _PLOT_BOT + 14
    bw = bx1 - bx0
    grad = colorize(np.linspace(0, 1, bw)[None, :], jet_lut())[0]
    _paste(img, Image.fromarray(np.tile(grad, (26, 1, 1))), (bx0, by))
    _border(draw, (bx0, by, bx1, by + 26))

    f_tick = _font(12)
    for i, t in enumerate(_ticks(lo, hi)):
        gx = bx0 + i / 4.0 * bw
        lbl = _fmt(t)
        draw.line([gx, by - 5, gx, by], fill=_AXIS, width=1)
        draw.text((min(gx - draw.textlength(lbl, font=f_tick) / 2, bx1 - 30), by - 18),
                  lbl, font=f_tick, fill=_TEXT)
    unit_lbl = "CT" if unit.lower() == "hu" else f"value ({unit})"
    draw.text((bx0, by - 36), unit_lbl, font=_font(13, bold=True), fill=_TEXT)
    dens = density_scale(unit, lo, hi)
    if dens:
        for i, (_v, d) in enumerate(dens):
            gx = bx0 + i / 4.0 * bw
            lbl = f"{d:.1f}"
            draw.line([gx, by + 26, gx, by + 31], fill=_AXIS, width=1)
            draw.text((min(gx - draw.textlength(lbl, font=f_tick) / 2, bx1 - 24), by + 34),
                      lbl, font=f_tick, fill=_TEXT)
        draw.text((bx0, by + 54), "Approx. Density (g/cm³)", font=_font(13, bold=True),
                  fill=_TEXT)


# ------------------------------------------------------------------- plate
def render_plate(data: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 unit: str = "intensity", name: str = "volume",
                 lo: float | None = None, hi: float | None = None,
                 depth_levels: int = 20, bins: int = 120, grid: int = 18,
                 cols: int = 3) -> bytes:
    """Render the report plate for a (z, y, x) volume; returns PNG bytes."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    z, y, x = arr.shape
    spacing_z, spacing_y, spacing_x = (float(s) for s in spacing)
    lo, hi = _window(arr, lo, hi)

    levels = max(4, min(int(depth_levels), z, 40))
    bins = max(16, min(int(bins), 256))

    img = Image.new("RGB", (_W, _H), _BG)
    draw = ImageDraw.Draw(img)

    draw.text((_MARGIN, 22), "Rockstone CT Workbench — CT Report Plate",
              font=_font(24, bold=True), fill=_TEXT)
    meta = (f"{name}  ·  {z}×{y}×{x}  ·  spacing {spacing_z:g}/{spacing_y:g}/{spacing_x:g} mm"
            f"  ·  unit: {unit}")
    draw.text((_MARGIN, 58), meta, font=_font(14), fill=_MUTE)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    draw.text((_W - _MARGIN - draw.textlength(stamp, font=_font(14)) - 2, 58), stamp,
              font=_font(14), fill=_MUTE)

    _band(img, draw, *_PANELS["traces"], "Histogram Depth (with driller's depth)")
    _band(img, draw, *_PANELS["map"], "Histogram Map")
    _band(img, draw, *_PANELS["slabs"], "Horizontal and vertical slabs")
    _band(img, draw, *_PANELS["grid"], "Slice images with color codes")

    _depth_traces(img, draw, arr, lo, hi, levels, bins, spacing_z, unit)
    _map_panel(img, draw, arr, lo, hi, levels, bins, spacing_z)
    _slabs(img, draw, arr, lo, hi)
    _slice_grid(img, draw, arr, lo, hi, grid, max(2, min(int(cols), 6)), spacing_z)
    _value_bar(img, draw, lo, hi, unit)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
