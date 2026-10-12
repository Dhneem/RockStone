"""Sanity-check the generated manual + cheat-sheet PDFs: pages, figures, text.

Usage: .venv/Scripts/python scripts/verify_manual_pdf.py
Requires: pypdf (pip install pypdf) -- used for manual verification only.
"""
import re
import sys

from pypdf import PdfReader

PDF = "docs/Rockstone-CT-Workbench-Manual.pdf"
data = open(PDF, "rb").read()

reader = PdfReader(PDF)
pages = len(reader.pages)
images = len(re.findall(rb"/Width\s+(\d+)", data))
eof_ok = data.rstrip().endswith(b"%%EOF")
text = "\n".join((p.extract_text() or "") for p in reader.pages)

probes = [
    "Rockstone CT Workbench", "Contents", "Installation", "uvicorn",
    "synthetic sample", "Beam-Hardening", "Spectral (Cu/Al", "Region Crop",
    "Slice Alignment", "Rotation", "phase correlation", "Processing Pipeline",
    "Ring Artifact Removal", "Quantitative Analysis", "NRRD", "REST API Reference",
    "/api/pipeline/run", "Testing", "pytest", "Limitations", "in-memory",
    "Versioning", "Changelog", "Release history", "2.0", "2.1",
    "Crack / Fracture Detection", "aperture", "/api/cracks", "12 stages",
    "Pore-size distribution", "inscribed-sphere", "D10 / D50 / D90", "/api/pores",
    "local thickness", "2.2", "51 tests", "70",
]
missing = [p for p in probes if p.lower() not in text.lower()]

print(f"manual: pages={pages} images={images} eof_ok={eof_ok} text_chars={len(text)}")
print("missing probes:", missing or "none")
ok = eof_ok and pages >= 18 and images == 9 and not missing

# ---- one-page quick-start cheat sheet ----
CS = "docs/Rockstone-CT-Quick-Start.pdf"
cs_data = open(CS, "rb").read()
cs = PdfReader(CS)
cs_text = " ".join((p.extract_text() or "") for p in cs.pages)
cs_probes = [
    "Rockstone", "Cheat Sheet", "uvicorn", "Import", "Beam hardening",
    "Spectral", "Cu+Al", "Align slices", "Run the pipeline", "Analyse",
    "NRRD", "Undo", "pytest", "2.2", "Detect cracks", "pore",
]
cs_missing = [p for p in cs_probes if p.lower() not in cs_text.lower()]
cs_ok = (len(cs.pages) == 1 and cs_data.rstrip().endswith(b"%%EOF") and not cs_missing)
print(f"cheatsheet: pages={len(cs.pages)} eof_ok={cs_data.rstrip().endswith(b'%%EOF')} text_chars={len(cs_text)}")
print("missing probes:", cs_missing or "none")

ok = ok and cs_ok
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
