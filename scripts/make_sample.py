"""Generate the bundled demo dataset under ``data/sample``.

Creates the cached synthetic rock-core volume (if missing) and writes it out as
a TIFF slice stack, a DICOM series and a headerless RAW volume, plus a
``manifest.json`` the UI can list when the user imports "the sample".

Run:  .venv/Scripts/python scripts/make_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.synth import load_or_generate, write_sample_dataset

SHAPE = (96, 256, 256)
OUT = Path("data/sample")


def main() -> None:
    vol = load_or_generate(SHAPE)
    print(f"volume {vol.shape} {vol.unit} spacing={vol.spacing} -> {OUT}")

    written = write_sample_dataset(vol, OUT, formats=("tiff", "dicom", "raw"))

    manifest = {
        "name": vol.name,
        "shape": {"z": vol.shape[0], "y": vol.shape[1], "x": vol.shape[2]},
        "spacing_mm": {"z": vol.spacing[0], "y": vol.spacing[1], "x": vol.spacing[2]},
        "unit": vol.unit,
        "mu_water_reference": vol.metadata.get("mu_water_reference"),
        "acquisition": vol.metadata.get("acquisition"),
        "datasets": written,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    for kind, info in written.items():
        print(f"  {kind:6s} {info}")
    print(f"manifest -> {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
