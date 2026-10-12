"""API smoke test: exercises every /api/* endpoint against a running server.

Run:  .venv/Scripts/python scripts/smoke_api.py [base_url]
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
PASS: list[str] = []
FAIL: list[str] = []


def call(method: str, path: str, body: dict | None = None, raw: bool = False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read()
            code = resp.status
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        code = exc.code
        ctype = exc.headers.get("Content-Type", "")
    if raw:
        return code, payload, ctype
    try:
        return code, json.loads(payload), ctype
    except ValueError:
        return code, payload[:200], ctype


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------- read-only
code, js, _ = call("GET", "/api/health")
check("health", code == 200 and js.get("ok"), str(js))

code, js, _ = call("GET", "/api/summary")
check("summary", code == 200 and js.get("loaded"), f"shape={js.get('shape')}")

code, js, _ = call("GET", "/api/detections")
check("detections", code == 200 and js.get("detections"),
      str(list(js.get("detections", {}).keys())))

code, js, _ = call("GET", "/api/browse?path=data")
check("browse", code == 200 and "sample" in js.get("files", []) + js.get("dirs", []),
      f"{len(js.get('dirs', []))} dirs {len(js.get('files', []))} files")

code, png, ctype = call("GET", "/api/slice.png?index=48", raw=True)
check("slice.png", code == 200 and png[:8] == b"\x89PNG\r\n\x1a\n", f"{len(png)}B {ctype}")

code, png, _ = call("GET", "/api/slice.png?index=48&overlay=mask", raw=True)
check("slice.png overlay=mask", code == 200 and png[:4] == b"\x89PNG", f"{len(png)}B")

code, png, _ = call("GET", "/api/slice.png?index=48&overlay=mask&color=%2300ff00", raw=True)
check("slice.png overlay=mask color", code == 200 and png[:4] == b"\x89PNG", f"{len(png)}B")

code, js, _ = call("GET", "/api/slice/info?index=48")
check("slice/info", code == 200 and "mean" in js, f"mean={js.get('mean'):.1f}")

code, brick, _ = call("GET", "/api/volume/brick?max_dim=96", raw=True)
check("volume/brick", code == 200 and len(brick) > 1000, f"{len(brick)} bytes")

# higher-resolution bricks (3D viewer Resolution options) + server clamp
for hi in (384, 512, 9999):
    code, b, _ = call("GET", f"/api/volume/brick?max_dim={hi}", raw=True)
    check(f"volume/brick max_dim={hi}", code == 200 and len(b) >= len(brick),
          f"{len(b)} bytes (vs {len(brick)} @96)")

code, js, _ = call("POST", "/api/bh/detect", {"index": 48})
check("bh/detect", code == 200 and "detect" in js,
      f"severity={js.get('detect', {}).get('cupping_severity')}")

code, js, _ = call("POST", "/api/bh/preview",
                   {"method": "thickness", "strength": 1.0, "index": 48})
check("bh/preview", code == 200 and js.get("before", "").startswith("data:image/png"),
      f"severity {js.get('detect_before', {}).get('cupping_severity')} -> "
      f"{js.get('detect_after', {}).get('cupping_severity')}")

code, js, _ = call("POST", "/api/roi/preview",
                   {"spec": {"type": "ellipse", "cx": 128, "cy": 128, "rx": 80, "ry": 90},
                    "index": 48})
check("roi/preview ellipse", code == 200 and js.get("area_fraction", 0) > 0,
      f"area={js.get('area_fraction', 0):.3f}")

code, js, _ = call("POST", "/api/roi/preview", {"spec": {"type": "auto"}, "index": 48})
check("roi/preview auto", code == 200 and js.get("area_fraction", 0) > 0.1,
      f"area={js.get('area_fraction', 0):.3f}")

code, js, _ = call("POST", "/api/roi/preview",
                   {"spec": {"type": "remove_air"}, "index": 48})
check("roi/preview remove_air", code == 200, f"area={js.get('area_fraction', 0):.3f}")

code, js, _ = call("POST", "/api/roi/preview",
                   {"spec": {"type": "artifact", "mode": "residual"}, "index": 48})
check("roi/preview artifact", code == 200, f"area={js.get('area_fraction', 0):.3f}")

code, js, _ = call("POST", "/api/roi/preview",
                   {"spec": {"type": "polygon",
                             "points": [[40, 40], [200, 50], [180, 200], [30, 180]]},
                    "index": 48})
check("roi/preview freehand", code == 200, f"area={js.get('area_fraction', 0):.3f}")

# ellipse 80×90 px on a 256² slice, pixel size 0.2 mm -> pi·80·90·0.04 ≈ 904.8 mm²
code, js, _ = call("POST", "/api/roi/area",
                   {"spec": {"type": "ellipse", "cx": 128, "cy": 128, "rx": 80, "ry": 90},
                    "index": 48})
expected = 3.14159265 * 80 * 90 * 0.2 * 0.2
check("roi/area", code == 200 and abs(js.get("area_mm2", 0) - expected) < expected * 0.02
      and js.get("spacing_mm") == [0.2, 0.2],
      f"mm2={js.get('area_mm2', 0):.1f} (expected ~{expected:.1f}) px={js.get('area_px')}")

code, js, _ = call("POST", "/api/roi/area", {"spec": {"type": "auto"}, "index": 48})
check("roi/area auto", code == 200 and js.get("area_px", 0) > 0 and js.get("area_mm2", 0) > 0,
      f"px={js.get('area_px')} mm2={js.get('area_mm2', 0):.1f}")

for method in ("phase", "cross", "feature", "mutualinfo"):
    code, js, _ = call("GET", f"/api/align/panel?index=49&ref=48&method={method}")
    est = js.get("estimate", {}) if code == 200 else {}
    check(f"align/panel {method}", code == 200 and est.get("ncc_after", 0) > 0.9,
          f"dx={est.get('dx')} dy={est.get('dy')} rot={est.get('rotation')} "
          f"ncc={est.get('ncc_after')}")

code, js, _ = call("POST", "/api/align/estimate",
                   {"index": 49, "ref_index": 48, "method": "phase"})
check("align/estimate", code == 200 and "estimate" in js,
      str(js.get("estimate", {}).get("dx")))

code, js, _ = call("POST", "/api/align/preview",
                   {"index": 49, "ref_index": 48, "dx": 1.5, "dy": -0.5, "rotation": 0.3})
check("align/preview", code == 200 and js.get("aligned_png", "").startswith("data:image"))

code, js, _ = call("GET", "/api/pipeline/stages")
ids = [s["id"] for s in js.get("stages", [])]
check("pipeline/stages", code == 200 and ids[0] == "inspect" and ids[-1] == "export",
      ",".join(ids))

code, js, _ = call("POST", "/api/stage/preview",
                   {"stage_id": "denoise", "params": {"method": "gaussian", "sigma": 1.2},
                    "index": 48})
check("stage/preview denoise", code == 200 and js.get("before", "").startswith("data:image"),
      str(js.get("info"))[:80])

code, js, _ = call("POST", "/api/stage/preview",
                   {"stage_id": "ring_removal", "params": {"strength": 1.0}, "index": 48})
check("stage/preview rings", code == 200 and js.get("before", "").startswith("data:image"))

code, js, _ = call("POST", "/api/stage/preview",
                   {"stage_id": "crop",
                    "params": {"mode": "shape", "shape": {"type": "circle", "cx": 128,
                                                          "cy": 128, "r": 100}},
                    "index": 48})
check("stage/preview crop", code == 200 and "area_fraction" in str(js.get("info", {})))

code, js, _ = call("POST", "/api/stage/preview",
                   {"stage_id": "beam_hardening",
                    "params": {"method": "spectral", "strength": 1.0}, "index": 48})
check("stage/preview beam_hardening spectral", code == 200,
      str(js.get("info", js))[:100])

# ------------------------------------------------------------- mutating calls
code, js, _ = call("POST", "/api/segment", {"method": "otsu", "index": 48})
check("segment", code == 200 and js.get("classes"),
      f"classes={len(js.get('classes', []))}")

code, js, _ = call("GET", "/api/analysis")
check("analysis", code == 200 and js.get("report"),
      f"keys={list((js.get('report') or {}).keys())[:6]}")
check("analysis pore_size section",
      bool((js.get("report") or {}).get("pore_size")) and
      "d50_mm" in (js.get("report") or {}).get("pore_size", {}),
      f"pore keys={list(((js.get('report') or {}).get('pore_size') or {}).keys())[:8]}")

code, js, _ = call("POST", "/api/bh/apply", {"method": "spectral", "strength": 1.0,
                                             "params": {}})
if code != 200:
    code, js, _ = call("POST", "/api/bh/apply", {"method": "thickness", "strength": 1.0})
check("bh/apply", code == 200,
      f"severity {js.get('detect_before', {}).get('cupping_severity')} -> "
      f"{js.get('detect_after', {}).get('cupping_severity')}")

# before/after comparison must be present and visually different while air is still noisy
code, js, _ = call("POST", "/api/roi/remove-air", {"index": 48})
check("roi/remove-air", code == 200 and 0 < js.get("area_fraction", 0) < 1
      and js.get("before", "").startswith("data:image/png")
      and js.get("after", "").startswith("data:image/png")
      and js["before"] != js["after"],
      f"thr={js.get('threshold'):.1f} kept={js.get('area_fraction', 0):.3f} "
      f"fill={js.get('fill'):.1f} before/after imgs differ")

code, js, _ = call("POST", "/api/roi/apply",
                   {"spec": {"type": "auto"}, "scope": "slice", "index": 48})
check("roi/apply", code == 200 and js.get("summary", {}).get("has_mask"),
      str(js.get("info"))[:80])

# regression: segmentation overlay must render while a (3-D) mask exists
code, js, _ = call("POST", "/api/segment", {"method": "otsu", "index": 48})
check("segment with mask", code == 200 and js.get("classes"),
      f"classes={len(js.get('classes', []))}")

code, js, _ = call("POST", "/api/roi/clean", {"fill": "auto"})
check("roi/clean", code == 200, f"fill={js.get('fill')}")

code, js, _ = call("DELETE", "/api/roi/mask")
check("roi/mask delete", code == 200)

code, js, _ = call("POST", "/api/align/apply",
                   {"mode": "single", "index": 50, "ref_index": 48, "method": "phase"})
check("align/apply single", code == 200, str(js.get("info"))[:100])

code, js, _ = call("POST", "/api/align/apply",
                   {"mode": "all", "ref_index": 48, "method": "phase"})
check("align/apply all", code == 200 and js.get("info", {}).get("n_transforms", 0) > 0,
      f"transforms={js.get('info', {}).get('n_transforms')}")

code, js, _ = call("POST", "/api/calibrate", {"target": "density", "degree": 1})
check("calibrate", code == 200, f"unit={js.get('summary', {}).get('unit')}")

code, js, _ = call("POST", "/api/undo")
check("undo", code == 200 and js.get("ok"), str(js.get("ok")))

# ----------------------------------------------------------- crack detection
code, js, _ = call("POST", "/api/sample", {"fractures": True})
check("sample fractures", code == 200 and js.get("loaded"),
      f"shape={js.get('shape')}")

code, js, _ = call("POST", "/api/cracks/detect", {"index": 48, "color": "#3399ff"})
res = js.get("result", {})
check("cracks/detect", code == 200 and res.get("count", 0) >= 1 and
      js.get("color") == "#3399ff",
      f"count={res.get('count')} vol={res.get('volume_mm3')} "
      f"aperture={res.get('mean_aperture_mm')} color={res.get('color')}")
check("cracks/detect overlay", js.get("overlay", "").startswith("data:image/png"),
      str(js.get("overlay", ""))[:30])

code, png, _ = call("GET", "/api/slice.png?overlay=cracks&index=48", raw=True)
check("slice.png overlay=cracks", code == 200 and png[:4] == b"\x89PNG",
      f"{len(png)}B")

code, png, _ = call("GET", "/api/slice.png?overlay=cracks&index=48&color=%23ff0000", raw=True)
check("slice.png cracks tint color", code == 200 and png[:4] == b"\x89PNG",
      f"{len(png)}B")

code, js, _ = call("GET", "/api/cracks?index=48")
check("cracks/get (restore)", code == 200 and
      (js.get("result") or {}).get("count", 0) >= 1 and
      js.get("overlay", "").startswith("data:image/png") and
      js.get("color") == "#3399ff",
      f"count={(js.get('result') or {}).get('count')} color={js.get('color')}")

code, js, _ = call("POST", "/api/cracks/detect",
                   {"min_voxels": 999999, "index": 48})
check("cracks/detect none", code == 200 and js.get("result", {}).get("count") == 0,
      f"count={js.get('result', {}).get('count')}")

code, js, _ = call("DELETE", "/api/cracks")
check("cracks delete", code == 200 and js.get("ok"), str(js.get("ok")))

# ------------------------------------------------------------- pore-size dist
code, js, _ = call("POST", "/api/pores", {"index": 48, "color": "#40d2be"})
res = js.get("result", {})
dist = res.get("distribution", {})
check("pores/analyse", code == 200 and res.get("count", 0) >= 1 and
      js.get("color") == "#40d2be",
      f"count={res.get('count')} d50={res.get('d50_mm')} "
      f"pore%={res.get('porosity_pct')} color={js.get('color')}")
check("pores distribution", sum(dist.get("counts", [])) == res.get("count") and
      len(dist.get("bin_edges_mm", [])) >= 2,
      f"bins={len(dist.get('counts', []))} log={dist.get('log_spaced')}")
check("pores quantiles", 0 < res.get("d10_mm", 0) <= res.get("d50_mm", 0) <= res.get("d90_mm", 0),
      f"D10/D50/D90={res.get('d10_mm')}/{res.get('d50_mm')}/{res.get('d90_mm')}")
check("pores overlay", js.get("overlay", "").startswith("data:image/png"),
      str(js.get("overlay", ""))[:30])

code, png, _ = call("GET", "/api/slice.png?overlay=pores&index=48", raw=True)
check("slice.png overlay=pores", code == 200 and png[:4] == b"\x89PNG", f"{len(png)}B")

code, js, _ = call("GET", "/api/pores?index=48")
check("pores/get (restore)", code == 200 and
      (js.get("result") or {}).get("count", 0) >= 1 and
      js.get("color") == "#40d2be",
      f"count={(js.get('result') or {}).get('count')} color={js.get('color')}")

code, js, _ = call("DELETE", "/api/pores")
check("pores delete", code == 200 and js.get("ok"), str(js.get("ok")))

code, js, _ = call("POST", "/api/sample", {})
check("sample reload (plain)", code == 200 and js.get("loaded"),
      str(js.get("shape")))

# --------------------------------------------------------------- pipeline run
code, stages, _ = call("GET", "/api/pipeline/stages")
plan = [{"id": s["id"], "enabled": s["id"] not in ("calibration",), "params": dict(s["params"])}
        for s in stages["stages"]]
code, js, _ = call("POST", "/api/pipeline/run", {"plan": plan})
check("pipeline/run", code == 200 and js.get("started"), str(js)[:80])

deadline = time.time() + 600
state = None
while time.time() < deadline:
    code, js, _ = call("GET", "/api/pipeline/progress")
    state = js.get("progress", {})
    if not js.get("running") and state.get("state") in ("done", "error"):
        break
    time.sleep(1.0)
results = js.get("results", [])
bad = [r for r in results if not r.get("ok")]
check("pipeline finished", state is not None and state.get("state") == "done",
      f"state={state} stages={len(results)}")
check("pipeline all stages ok", not bad,
      f"failed={[ (b['id'], b.get('error')) for b in bad ]}")
check("pipeline reconstruction", bool(js.get("reconstruction")),
      str(js.get("reconstruction"))[:100])

# ------------------------------------------------------------------- exports
for fmt, ext in (("tiff", "tif"), ("png", "zip"), ("raw", "raw"),
                 ("nrrd", "nrrd"), ("csv", "csv"), ("json", "json")):
    code, payload, ctype = call("GET", f"/api/export/{fmt}", raw=True)
    check(f"export {fmt}", code == 200 and len(payload) > 20,
          f"{len(payload)}B {ctype}")

code, js, _ = call("POST", "/api/sample", {})
check("sample reload", code == 200 and js.get("loaded"),
      str(js.get("shape")))

code, js, _ = call("POST", "/api/import", {"path": "data/sample/dicom", "kind": "auto"})
check("import dicom dir", code == 200 and js.get("summary", {}).get("loaded"),
      f"shape={js.get('summary', {}).get('shape')}")

code, js, _ = call("POST", "/api/import", {"path": "data/sample/tiff", "kind": "auto"})
check("import tiff dir", code == 200 and js.get("summary", {}).get("unit") == "HU",
      f"unit={js.get('summary', {}).get('unit')}")

code, js, _ = call("POST", "/api/import", {"path": "data/sample/raw", "kind": "auto"})
check("import raw dir", code == 200 and js.get("summary", {}).get("loaded"),
      f"detections={list(js.get('detections', {}).keys())}")

code, js, _ = call("POST", "/api/import", {"path": "no/such/path"})
check("import bad path -> 400", code == 400, f"code={code}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("failed:", FAIL)
    sys.exit(1)
