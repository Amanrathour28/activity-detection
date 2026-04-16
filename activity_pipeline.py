"""
activity_pipeline.py
=====================
Standalone module for the Activity + Intake Detection Pipeline.

This is the ONLY file needed (alongside activity_pipeline.pkl) to load
and use the pipeline in any external project.

Usage:
    import pickle
    from activity_pipeline import ActivityPipeline   # makes pkl loadable
    pipeline = pickle.load(open("activity_pipeline.pkl", "rb"))
    result = pipeline.predict(bgr_frame)

Dependencies:
    pip install mediapipe>=0.10.0 numpy>=1.21 opencv-python>=4.5
"""

import pickle
import time
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any, Deque


# ════════════════════════════════════════════════════════════════════
#  LABEL METADATA
# ════════════════════════════════════════════════════════════════════

POSTURE_ACTIONS: Dict[str, tuple] = {
    "sleeping": ("SLEEPING",    "Person sleeping on the floor",  (30,  80,  220), "[ZZZ]"),
    "lying":    ("LYING DOWN",  "Person lying on the floor",     (200, 60,  255), "[LIE]"),
    "falling":  ("FALLING!",    "Person is falling",             (0,   60,  255), "[!!!]"),
    "sitting":  ("SITTING",     "Person is sitting",             (255, 140, 0),   "[ S ]"),
    "standing": ("STANDING",    "Person is standing still",      (50,  200, 50),  "[ | ]"),
    "walking":  ("WALKING",     "Person is walking or moving",   (255, 180, 20),  "[-->]"),
    "unknown":  ("DETECTING..", "Waiting for pose detection",    (120, 120, 120), "[ ? ]"),
}

INTAKE_COLORS: Dict[str, tuple] = {
    "EATING":     (0,   200, 80),
    "DRINKING":   (20,  180, 255),
    "NOT EATING": (80,  80,  80),
}

# ════════════════════════════════════════════════════════════════════
#  CONFIG CONSTANTS
# ════════════════════════════════════════════════════════════════════

_MIN_INTAKE_FRAMES  = 4
_BPM_WINDOW_SECONDS = 60
_WRIST_HISTORY_LEN  = 10
_SMOOTHING_FRAMES   = 8
_SHOULDER_RATIO     = 0.60

# MediaPipe Pose landmark indices
_LM_MOUTH_L    = 9
_LM_MOUTH_R    = 10
_LM_L_WRIST    = 15
_LM_R_WRIST    = 16
_LM_L_SHOULDER = 11
_LM_R_SHOULDER = 12
_LM_L_HIP      = 23
_LM_R_HIP      = 24
_LM_L_KNEE     = 25
_LM_R_KNEE     = 26


# ════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS (pure, stateless)
# ════════════════════════════════════════════════════════════════════

def _px(lm, idx: int, fw: int, fh: int) -> Tuple[int, int]:
    return (int(lm[idx].x * fw), int(lm[idx].y * fh))

def _dist(a: Tuple, b: Tuple) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

def _midpoint(a: Tuple, b: Tuple) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

def _angle_with_vertical(p1: Tuple, p2: Tuple) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(float(np.degrees(np.arctan2(abs(dx), abs(dy)))))

def _get_mouth_info(pose_lm, fw: int, fh: int):
    if pose_lm is None:
        return None, None
    lm = pose_lm.landmark
    if lm[_LM_MOUTH_L].visibility < 0.4 and lm[_LM_MOUTH_R].visibility < 0.4:
        return None, None
    ml = _px(lm, _LM_MOUTH_L, fw, fh)
    mr = _px(lm, _LM_MOUTH_R, fw, fh)
    cx = (ml[0] + mr[0]) // 2
    cy = (ml[1] + mr[1]) // 2
    box_w = max(40, abs(ml[0] - mr[0]) + 30)
    box_h = max(30, box_w - 10)
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    return (cx, cy), (x1, y1, box_w, box_h)

def _get_dynamic_threshold(pose_lm, fw: int, fh: int, shoulder_ratio: float) -> float:
    if pose_lm is None:
        return 100.0
    lm = pose_lm.landmark
    ls = _px(lm, _LM_L_SHOULDER, fw, fh)
    rs = _px(lm, _LM_R_SHOULDER, fw, fh)
    return max(50.0, _dist(ls, rs) * shoulder_ratio)

def _closest_wrist(pose_lm, mouth_center, fw: int, fh: int):
    if pose_lm is None or mouth_center is None:
        return None, float("inf"), None
    lm = pose_lm.landmark
    lw = _px(lm, _LM_L_WRIST, fw, fh)
    rw = _px(lm, _LM_R_WRIST, fw, fh)
    ld = _dist(lw, mouth_center) if lm[_LM_L_WRIST].visibility > 0.4 else float("inf")
    rd = _dist(rw, mouth_center) if lm[_LM_R_WRIST].visibility > 0.4 else float("inf")
    if ld == float("inf") and rd == float("inf"):
        return None, float("inf"), None
    if ld <= rd:
        return lw, ld, "L"
    return rw, rd, "R"


# ════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE CLASS
# ════════════════════════════════════════════════════════════════════

class ActivityPipeline:
    """
    Self-contained Activity + Intake Detection Pipeline.

    Encapsulates:
      - PostureDetector (sleeping / lying / falling / sitting / standing / walking)
      - IntakeDetector  (eating / drinking — wrist-to-mouth proximity)

    Both detectors use MediaPipe Pose landmarks. No face recognition.

    Quick start
    -----------
    >>> import pickle
    >>> from activity_pipeline import ActivityPipeline
    >>> pipeline = pickle.load(open("activity_pipeline.pkl", "rb"))
    >>> result = pipeline.predict(bgr_frame)   # OpenCV BGR numpy array
    >>> print(result["posture"], result["intake"])
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        smoothing_frames: int           = _SMOOTHING_FRAMES,
        min_intake_frames: int          = _MIN_INTAKE_FRAMES,
        sitting_hip_y_threshold: float  = 0.52,
        sitting_knee_diff_max: float    = 0.18,
        walk_motion_threshold: float    = 1500.0,
        shoulder_ratio: float           = _SHOULDER_RATIO,
    ):
        # Config (serialized)
        self.smoothing_frames        = smoothing_frames
        self.min_intake_frames       = min_intake_frames
        self.sitting_hip_y_threshold = sitting_hip_y_threshold
        self.sitting_knee_diff_max   = sitting_knee_diff_max
        self.walk_motion_threshold   = walk_motion_threshold
        self.shoulder_ratio          = shoulder_ratio

        # Posture state
        self._p_history:        deque = deque(maxlen=smoothing_frames)
        self._p_angle_history:  deque = deque(maxlen=20)
        self._p_horiz_start:    Optional[float] = None
        self._p_horiz_dur:      float = 0.0
        self._p_fall_start:     Optional[float] = None
        self._p_fall_confirmed: bool  = False
        self._p_last_angle:     float = 0.0

        # Intake state
        self._i_bite_times:   list  = []
        self._i_bite_count:   int   = 0
        self._i_consecutive:  int   = 0
        self._i_was_intake:   bool  = False
        self._i_wrist_y_hist: deque = deque(maxlen=_WRIST_HISTORY_LEN)

        # Motion state
        self._prev_gray = None

        # MediaPipe (NOT serialized — lazy-loaded on first predict())
        self._mp_pose_obj  = None
        self._mp_face_obj  = None
        self._is_mp_ready  = False

        # Timing
        self._last_frame_time: float = 0.0
        self._fps:             float = 0.0

    # ── Pickle protocol ────────────────────────────────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_mp_pose_obj", "_mp_face_obj", "_is_mp_ready", "_prev_gray"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mp_pose_obj  = None
        self._mp_face_obj  = None
        self._is_mp_ready  = False
        self._prev_gray    = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full pipeline on a single OpenCV BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            uint8 BGR image, any resolution (e.g. 480x640x3 or 720x1280x3).

        Returns
        -------
        dict
            posture             : str   e.g. "standing"
            posture_label       : str   e.g. "STANDING"
            posture_color       : list  BGR e.g. [50, 200, 50]
            posture_icon        : str   e.g. "[ | ]"
            posture_desc        : str   e.g. "Person is standing still"
            body_angle          : float degrees from vertical
            horizontal_duration : float seconds lying/sleeping (else 0.0)
            intake              : str   "NOT EATING" | "EATING" | "DRINKING"
            is_intake           : bool
            intake_color        : list  BGR
            bite_count          : int
            bpm                 : float bites-per-minute (rolling 60s)
            confidence          : float 0.0-1.0 wrist proximity
            fps                 : float inferred from frame timing
        """
        self._ensure_mp_ready()
        import cv2

        now = time.time()
        self._fps = 1.0 / max(now - self._last_frame_time, 1e-9) \
                    if self._last_frame_time else 0.0
        self._last_frame_time = now

        # Motion score
        gray  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gblur = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            motion_score = 0.0
        else:
            diff         = cv2.absdiff(self._prev_gray, gblur)
            motion_score = float(np.sum(diff > 25))
        self._prev_gray = gblur

        # MediaPipe inference
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_result = self._mp_pose_obj.process(rgb)
        face_result = self._mp_face_obj.process(rgb)
        rgb.flags.writeable = True

        return self._build_output(pose_result, face_result,
                                  bgr_frame.shape, motion_score)

    def predict_landmarks(
        self,
        pose_landmarks,
        frame_shape: tuple,
        motion_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run the pipeline with pre-computed MediaPipe pose landmarks.

        Use this when MediaPipe Pose is already running elsewhere in your app
        and you want to avoid running it twice.

        Parameters
        ----------
        pose_landmarks : NormalizedLandmarkList | None
            mediapipe pose_result.pose_landmarks
        frame_shape : tuple
            (height, width, channels)
        motion_score : float, optional
            Pre-computed pixel-diff motion (pass 0 if unknown)

        Returns
        -------
        dict — same structure as predict()
        """
        class _FakePose:
            def __init__(self, lm):
                self.pose_landmarks = lm

        return self._build_output(
            _FakePose(pose_landmarks), None, frame_shape, motion_score
        )

    def reset(self):
        """
        Reset all session state (bite counts, posture history, motion buffer).
        Call between sessions or when switching subjects.
        """
        self._p_history.clear()
        self._p_angle_history.clear()
        self._p_horiz_start    = None
        self._p_horiz_dur      = 0.0
        self._p_fall_start     = None
        self._p_fall_confirmed = False
        self._p_last_angle     = 0.0
        self._i_bite_times     = []
        self._i_bite_count     = 0
        self._i_consecutive    = 0
        self._i_was_intake     = False
        self._i_wrist_y_hist.clear()
        self._prev_gray        = None

    def get_labels(self) -> Dict[str, Any]:
        """Returns all possible labels and their metadata."""
        return {
            "posture_labels": list(POSTURE_ACTIONS.keys()),
            "intake_labels":  ["NOT EATING", "EATING", "DRINKING"],
            "posture_detail": {k: {"label": v[0], "description": v[1],
                                   "color_bgr": list(v[2])}
                               for k, v in POSTURE_ACTIONS.items()},
        }

    # ── Internal: build result dict ────────────────────────────────────────────

    def _build_output(self, pose_result, face_result, frame_shape, motion_score):
        posture_out = self._run_posture(pose_result, motion_score)
        intake_out  = self._run_intake(pose_result, face_result, frame_shape)

        p_info  = POSTURE_ACTIONS[posture_out["action"]]
        i_color = INTAKE_COLORS.get(intake_out["label"], (80, 80, 80))

        return {
            "posture":             posture_out["action"],
            "posture_label":       p_info[0],
            "posture_color":       list(p_info[2]),
            "posture_icon":        p_info[3],
            "posture_desc":        p_info[1],
            "body_angle":          posture_out["body_angle"],
            "horizontal_duration": posture_out["horiz_dur"],
            "intake":              intake_out["label"],
            "is_intake":           intake_out["is_intake"],
            "intake_color":        list(i_color),
            "bite_count":          intake_out["bite_count"],
            "bpm":                 intake_out["bpm"],
            "confidence":          intake_out["confidence"],
            "fps":                 round(self._fps, 1),
        }

    # ── Lazy MediaPipe init ────────────────────────────────────────────────────

    def _ensure_mp_ready(self):
        if self._is_mp_ready:
            return
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is required.\n"
                "Install: pip install mediapipe>=0.10.0"
            )
        self._mp_pose_obj = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ).__enter__()
        self._mp_face_obj = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
        ).__enter__()
        self._is_mp_ready = True

    def __del__(self):
        try:
            if self._mp_pose_obj:
                self._mp_pose_obj.__exit__(None, None, None)
            if self._mp_face_obj:
                self._mp_face_obj.__exit__(None, None, None)
        except Exception:
            pass

    # ── Posture detection ──────────────────────────────────────────────────────

    def _run_posture(self, pose_result, motion_score: float) -> dict:
        raw = self._posture_classify(pose_result, motion_score)
        self._p_history.append(raw)
        counts: Dict[str, int] = {}
        for a in self._p_history:
            counts[a] = counts.get(a, 0) + 1
        best = max(counts, key=counts.get)
        return {"action": best, "body_angle": self._p_last_angle,
                "horiz_dur": self._p_horiz_dur}

    def _posture_classify(self, pose_result, motion_score: float) -> str:
        if pose_result is None or pose_result.pose_landmarks is None:
            self._update_horizontal(False)
            self._update_fall(False)
            return "unknown"
        lm = pose_result.pose_landmarks.landmark
        vis = [lm[i].visibility for i in [11, 12, 23, 24]]
        if min(vis) < 0.3:
            self._update_horizontal(False)
            self._update_fall(False)
            return "unknown"
        l_sh = (lm[11].x, lm[11].y);  r_sh = (lm[12].x, lm[12].y)
        l_hi = (lm[23].x, lm[23].y);  r_hi = (lm[24].x, lm[24].y)
        l_kn = (lm[25].x, lm[25].y);  r_kn = (lm[26].x, lm[26].y)
        sh_mid  = _midpoint(l_sh, r_sh)
        hip_mid = _midpoint(l_hi, r_hi)
        kn_mid  = _midpoint(l_kn, r_kn)
        angle = _angle_with_vertical(sh_mid, hip_mid)
        self._p_last_angle = angle
        self._p_angle_history.append(angle)
        is_horiz = angle > 60
        self._update_horizontal(is_horiz)
        if is_horiz:
            self._update_fall(False)
            return "sleeping" if self._p_horiz_dur > 3.0 else "lying"
        self._update_fall(self._check_falling(angle))
        if self._p_fall_confirmed:
            return "falling"
        if hip_mid[1] > self.sitting_hip_y_threshold and \
           abs(kn_mid[1] - hip_mid[1]) < self.sitting_knee_diff_max:
            return "sitting"
        if motion_score > self.walk_motion_threshold:
            return "walking"
        return "standing"

    def _check_falling(self, angle: float) -> bool:
        if angle < 40 or len(self._p_angle_history) < 8:
            return False
        recent = list(self._p_angle_history)[-8:]
        ang_vel = (recent[-1] - recent[0]) / len(recent)
        return ang_vel > 2.5 or angle > 55

    def _update_horizontal(self, is_horiz: bool):
        if is_horiz:
            if self._p_horiz_start is None:
                self._p_horiz_start = time.time()
            self._p_horiz_dur = time.time() - self._p_horiz_start
        else:
            self._p_horiz_start = None
            self._p_horiz_dur   = 0.0

    def _update_fall(self, is_falling: bool):
        if is_falling:
            if self._p_fall_start is None:
                self._p_fall_start = time.time()
            if time.time() - self._p_fall_start > 0.4:
                self._p_fall_confirmed = True
        else:
            self._p_fall_start     = None
            self._p_fall_confirmed = False

    # ── Intake detection ───────────────────────────────────────────────────────

    def _run_intake(self, pose_result, face_result, frame_shape: tuple) -> dict:
        fh, fw = frame_shape[:2]
        pose_lm = pose_result.pose_landmarks \
                  if (pose_result and pose_result.pose_landmarks) else None
        mouth_center, _ = _get_mouth_info(pose_lm, fw, fh)
        threshold = _get_dynamic_threshold(pose_lm, fw, fh, self.shoulder_ratio)
        wrist_pt, wrist_dist, _ = _closest_wrist(pose_lm, mouth_center, fw, fh)
        conf = max(0.0, 1.0 - (wrist_dist / threshold)) \
               if wrist_dist < threshold else 0.0
        raw_intake = wrist_dist < threshold
        if wrist_pt is not None:
            self._i_wrist_y_hist.append(wrist_pt[1])
        self._i_consecutive = self._i_consecutive + 1 if raw_intake else 0
        confirmed = self._i_consecutive >= self.min_intake_frames
        if confirmed and not self._i_was_intake:
            self._record_bite()
        self._i_was_intake = confirmed
        label = self._classify_intake_type(confirmed, wrist_pt, mouth_center)
        return {"is_intake": confirmed, "label": label,
                "bpm": self._calc_bpm(), "bite_count": self._i_bite_count,
                "confidence": conf}

    def _record_bite(self):
        now = time.time()
        self._i_bite_times.append(now)
        self._i_bite_count += 1
        cutoff = now - _BPM_WINDOW_SECONDS
        self._i_bite_times = [t for t in self._i_bite_times if t >= cutoff]

    def _calc_bpm(self) -> float:
        if len(self._i_bite_times) < 2:
            return 0.0
        elapsed = self._i_bite_times[-1] - self._i_bite_times[0]
        if elapsed <= 0:
            return 0.0
        return round((len(self._i_bite_times) - 1) / elapsed * 60, 1)

    def _classify_intake_type(self, confirmed, wrist_pt, mouth_center) -> str:
        if not confirmed or wrist_pt is None or mouth_center is None:
            return "NOT EATING"
        return "DRINKING" if wrist_pt[1] > (mouth_center[1] + 15) else "EATING"
