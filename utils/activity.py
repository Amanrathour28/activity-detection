"""
utils/activity.py
==================
Thin wrapper around models/activity_pipeline.pkl.
Exposes a simple predict(bgr_frame) → dict API.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Ensure project root is importable so core.activity_pipeline resolves
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.activity_pipeline import ActivityPipeline  # noqa: F401  (required for pkl)

_PKL_PATH = _root / "models" / "activity_pipeline.pkl"


class ActivityDetector:
    """
    Wraps ActivityPipeline to provide a clean one-call API.

    Usage:
        det = ActivityDetector()
        result = det.predict(bgr_frame)
        print(result["posture"], result["intake"])
    """

    def __init__(self):
        if not _PKL_PATH.exists():
            raise FileNotFoundError(
                f"[ActivityDetector] Pipeline not found: {_PKL_PATH}\n"
                "Run: python core/build_pipeline.py"
            )
        with open(str(_PKL_PATH), "rb") as f:
            self._pipeline: ActivityPipeline = pickle.load(f)
        print(f"[ActivityDetector] Loaded pipeline v{self._pipeline.VERSION}")

    def predict(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Run posture + intake detection on a full BGR frame."""
        return self._pipeline.predict(bgr_frame)

    def reset(self):
        """Reset session state (call when target changes or is lost)."""
        self._pipeline.reset()

    @property
    def version(self) -> str:
        return self._pipeline.VERSION
