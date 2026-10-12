"""Compare beam-hardening correction methods on the synthetic phantom.

Run:  .venv/Scripts/python -m scripts.compare_corrections   (or python scripts/compare_corrections.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.synth import load_or_generate
from server.processing import beamhardening as bh
from server.processing.roi import auto_boundary

SHAPE = (96, 256, 256)


def main() -> None:
    vol = load_or_generate(SHAPE)
    data = vol.data
    spacing = vol.spacing
    truth = vol.metadata.get("ground_truth", {}).get("truth", {})
    print(f"volume {data.shape} {vol.unit}  spacing={spacing}")
    print(f"truth matrix_HU_mono60 = {truth.get('matrix_HU_mono60'):.1f}"
          f"  mu_water_reference = {vol.metadata.get('mu_water_reference'):.5f}")

    mask = auto_boundary(data[data.shape[0] // 2])
    if isinstance(mask, tuple):
        mask = mask[0]
    print(f"mask area = {int(mask.sum())} px")

    base = bh.detect(data, mask, vol.unit, spacing)
    print(f"BEFORE severity={base['cupping_severity']:+.4f} core={base['core_mean']:.0f} "
          f"shell={base['shell_mean']:.0f} streak={base['dark_streak_fraction']:.3f} "
          f"-> {base['verdict']}")

    params_common = {"mu_water_reference": vol.metadata.get("mu_water_reference")}
    acq = vol.metadata.get("acquisition", {}) or {}
    for method, params in [
        ("polynomial", dict(params_common)),
        ("water", dict(params_common)),
        ("thickness", dict(params_common)),
        ("spectral", dict(params_common, kvp=float(acq.get("kvp", 80)),
                          inherent_al_mm=float(acq.get("inherent_al_mm", 0.0)),
                          added_filter=str(acq.get("added_filter", "aluminum")),
                          added_mm=float(acq.get("added_mm", 3.0)),
                          material=str(acq.get("material", "quartz")))),
    ]:
        t0 = time.time()
        try:
            out, info = bh.correct(data, method, params, strength=1.0, unit=vol.unit,
                                   mask=mask, spacing=spacing,
                                   mu_water=vol.metadata.get("mu_water_reference"))
        except Exception as exc:  # noqa: BLE001
            print(f"{method:12s} FAILED: {exc}")
            continue
        a = info["after"]
        print(f"{method:12s} severity={a['cupping_severity']:+.4f} core={a['core_mean']:.0f} "
              f"shell={a['shell_mean']:.0f} streak={a['dark_streak_fraction']:.3f} "
              f"({time.time() - t0:.2f}s)")
        del out


if __name__ == "__main__":
    main()
