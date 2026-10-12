"""In-memory workspace: the active volume, undo history, masks and transforms.

History entries are stored quantised to uint16 (scale + offset) so a 300 MB
float32 volume does not cost 300 MB per undo step.  The quantisation error for a
typical CT range (-1024 .. 3000 HU) is well below 0.1 HU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Volume:
    """A scalar 3-D volume (z, y, x) plus physical spacing and metadata."""

    data: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # z, y, x in mm
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = "volume"
    unit: str = "intensity"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.data.ndim == 2:
            self.data = self.data[None, ...]
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        self.data = np.ascontiguousarray(self.data)
        self.spacing = tuple(float(s) for s in self.spacing)  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        z, y, x = self.spacing
        return z * y * x

    def stats(self) -> dict[str, Any]:
        d = self.data
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        }


@dataclass
class HistoryEntry:
    """A restorable snapshot of the volume data."""

    data: np.ndarray  # uint16
    scale: float
    offset: float
    label: str
    timestamp: float

    @classmethod
    def from_array(cls, arr: np.ndarray, label: str) -> "HistoryEntry":
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        scale = (hi - lo) / 65534.0
        q = np.clip((arr - lo) / scale, 0, 65534.0)
        return cls(q.astype(np.uint16), float(scale), float(lo), label, time.time())

    def restore(self) -> np.ndarray:
        return (self.data.astype(np.float32) * self.scale + self.offset).astype(np.float32)


class Workspace:
    """Everything the running application knows about the current dataset."""

    MAX_HISTORY = 8

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.volume: Volume | None = None
        # pristine copy of the imported data, used for before/after comparisons
        self.baseline: np.ndarray | None = None
        self.history: list[HistoryEntry] = []
        # 2-D ROI mask (y, x) applied to every slice, and optional per-slice mask
        self.mask2d: np.ndarray | None = None
        self.mask3d: np.ndarray | None = None
        # per-slice rigid transforms applied by the alignment stage: {z: {dx,dy,rot,scale}}
        self.transforms: dict[int, dict[str, float]] = {}
        self.segmentation: np.ndarray | None = None
        # crack/fracture detection result: int32 component labels + metrics
        self.crack_labels: np.ndarray | None = None
        self.crack_result: dict[str, Any] | None = None
        self.crack_color: tuple[int, int, int] | None = None  # overlay tint; None = default orange
        # pore-size distribution result: int32 component labels + metrics
        self.pore_labels: np.ndarray | None = None
        self.pore_result: dict[str, Any] | None = None
        self.pore_color: tuple[int, int, int] | None = None  # overlay tint; None = default teal
        self.pipeline_log: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ volume
    def set_volume(self, vol: Volume, keep_baseline: bool = True) -> None:
        with self.lock:
            self.volume = vol
            if keep_baseline:
                self.baseline = vol.data.copy()
            self.history = []
            self.mask2d = self.mask3d = None
            self.segmentation = None
            self.crack_labels = None
            self.crack_result = None
            self.pore_labels = None
            self.pore_result = None
            self.pore_color = None
            self.transforms = {}
            self.pipeline_log = []
            self.notes = []

    def push_history(self, label: str) -> None:
        with self.lock:
            if self.volume is None:
                return
            self.history.append(HistoryEntry.from_array(self.volume.data, label))
            if len(self.history) > self.MAX_HISTORY:
                self.history.pop(0)

    def undo(self) -> bool:
        with self.lock:
            if not self.history or self.volume is None:
                return False
            entry = self.history.pop()
            self.volume.data = entry.restore()
            self.segmentation = None
            return True

    def replace_data(self, data: np.ndarray) -> None:
        with self.lock:
            if self.volume is None:
                raise RuntimeError("no volume loaded")
            self.volume.data = data.astype(np.float32, copy=False)

    def effective_mask(self) -> np.ndarray | None:
        """3-D boolean mask covering the voxels kept in the analysis, or None."""
        if self.volume is None:
            return None
        if self.mask3d is not None:
            return self.mask3d.astype(bool)
        if self.mask2d is not None:
            return np.broadcast_to(self.mask2d.astype(bool), self.volume.shape)
        return None

    def summary(self) -> dict[str, Any]:
        with self.lock:
            if self.volume is None:
                return {"loaded": False}
            v = self.volume
            return {
                "loaded": True,
                "name": v.name,
                "shape": {"z": v.shape[0], "y": v.shape[1], "x": v.shape[2]},
                "spacing": {"z": v.spacing[0], "y": v.spacing[1], "x": v.spacing[2]},
                "voxel_mm": {
                    "x": v.spacing[2],
                    "y": v.spacing[1],
                    "z": v.spacing[0],
                    "pixel_size": (v.spacing[1] + v.spacing[2]) / 2.0,
                },
                "unit": v.unit,
                "stats": v.stats(),
                "metadata": v.metadata,
                "history": [h.label for h in self.history],
                "has_mask": self.mask2d is not None or self.mask3d is not None,
                "n_transforms": len(self.transforms),
                "has_segmentation": self.segmentation is not None,
                "baseline_ready": self.baseline is not None,
            }


WORKSPACE = Workspace()
