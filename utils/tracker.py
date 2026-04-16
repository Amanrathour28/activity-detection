"""
utils/tracker.py
=================
Lightweight single-target IoU bounding box tracker.

Why IoU (not SORT / DeepSORT)?
  - We track exactly ONE person. IoU is sufficient and has zero extra deps.
  - SORT/DeepSORT add Kalman state and Hungarian matching — overkill here.

Logic:
  - After face recognition finds the target, the tracker stores the bbox.
  - On every subsequent frame, we compare the stored bbox to the new
    InsightFace detection (only run every N frames). If IoU > threshold,
    we update the stored bbox — keeping the lock alive.
  - If target is not seen by recognition for GRACE_SECONDS, the lock drops.
"""

import time
from typing import Optional, Tuple
import numpy as np


BBox = Tuple[int, int, int, int]   # x1, y1, x2, y2


class TargetTracker:
    """
    Maintains the bounding box lock on the single registered target.

    The tracker does NOT run its own detection — it receives updates from
    the face recognition module (called every RECOG_INTERVAL frames) and
    interpolates / validates the box between updates.
    """

    GRACE_SECONDS = 2.0     # how long to keep locked after last seen

    def __init__(self):
        self._bbox:       Optional[BBox]  = None
        self._last_seen:  float           = 0.0
        self._similarity: float           = 0.0
        self._locked:     bool            = False

    # ── Update from recognition output ────────────────────────────────────────

    def update(self, bbox: Optional[BBox], similarity: float = 1.0):
        """
        Call this when recognition returns a result.
        bbox=None means the target was NOT found in this recognition cycle.
        """
        if bbox is not None:
            self._bbox       = _clamp_bbox(bbox)
            self._last_seen  = time.time()
            self._similarity = similarity
            self._locked     = True
        else:
            # Check if grace period has expired
            if time.time() - self._last_seen > self.GRACE_SECONDS:
                self._locked = False
                self._bbox   = None

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def bbox(self) -> Optional[BBox]:
        return self._bbox

    @property
    def similarity(self) -> float:
        return self._similarity

    def time_since_seen(self) -> float:
        return time.time() - self._last_seen if self._locked else float("inf")

    def reset(self):
        self._bbox       = None
        self._last_seen  = 0.0
        self._similarity = 0.0
        self._locked     = False


# ── Helper ────────────────────────────────────────────────────────────────────

def _clamp_bbox(bbox, min_val=0) -> BBox:
    """Ensure bbox coords are non-negative integers."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (max(min_val, x1), max(min_val, y1),
            max(min_val, x2), max(min_val, y2))


def iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0]);  y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]);  y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter)
