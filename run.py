"""
run.py  —  Unified Activity Detection System
=============================================

Runs in ONE command:
    python run.py              # default camera
    python run.py --cam 1      # alternate camera index

Pipeline:
    ┌──────────────────────────────────────────┐
    │  Phase 1: REGISTRATION                   │
    │   Webcam opens → press SPACE × 5         │
    │   InsightFace embeds face → stored in RAM│
    └──────────────┬───────────────────────────┘
                   │  (camera stays open, no restart)
    ┌──────────────▼───────────────────────────┐
    │  Phase 2: INFERENCE                      │
    │  Every 15 frames → InsightFace re-check  │
    │  Every 1 frame   → IoU tracker           │
    │  Every 1 frame   → MediaPipe Activity    │
    │  Display: ONLY target person bbox + info │
    └──────────────────────────────────────────┘

Controls (Phase 2):
    Q → Quit
    R → Re-register (restart Phase 1)
"""

import sys
import time
import argparse
import cv2
import numpy as np

from utils.face_auth  import FaceAuth
from utils.tracker    import TargetTracker
from utils.activity   import ActivityDetector


# ── Tuning constants ──────────────────────────────────────────────────────────
RECOG_INTERVAL   = 15     # run face recognition every N frames
CAPTURE_COUNT    = 5      # SPACE presses needed for registration
CAPTURE_COOLDOWN = 0.5    # seconds between SPACE presses (avoid duplicates)

# ── Colour palette ────────────────────────────────────────────────────────────
CLR_GREEN  = (50,  220,  50)
CLR_YELLOW = (30,  200, 255)
CLR_RED    = (40,   40, 200)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (15,   15,  15)
CLR_GREY   = (150, 150, 150)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

def phase_register(cap: cv2.VideoCapture, face_auth: FaceAuth) -> bool:
    """
    Interactive face registration.
    Returns True when embedding is locked; False if user pressed Q.
    """
    last_capture_t = 0.0
    face_auth.reset_registration()

    print("\n[REGISTER] Press SPACE to capture your face (×5).")
    print("[REGISTER] Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        display = cv2.flip(frame, 1)
        fh, fw = display.shape[:2]
        n = face_auth.samples_collected

        # ── Dark top bar ────────────────────────────────────────────────────
        _bar(display, height=80, color=CLR_DARK)
        cv2.putText(display,
                    f"FACE REGISTRATION  —  {n}/{CAPTURE_COUNT} captured",
                    (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.70, CLR_WHITE, 1)
        cv2.putText(display,
                    "Look straight • Even lighting • SPACE to capture • Q to quit",
                    (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_GREY, 1)

        # ── Progress dots ───────────────────────────────────────────────────
        for i in range(CAPTURE_COUNT):
            cx = fw // 2 - (CAPTURE_COUNT * 24) // 2 + i * 24
            color = CLR_GREEN if i < n else (60, 60, 60)
            cv2.circle(display, (cx, 70), 7, color, -1)

        # ── Centre guide circle ─────────────────────────────────────────────
        cx, cy = fw // 2, fh // 2
        cv2.circle(display, (cx, cy), 100, CLR_GREY, 1)
        cv2.line(display, (cx-10, cy), (cx+10, cy), CLR_GREY, 1)
        cv2.line(display, (cx, cy-10), (cx, cy+10), CLR_GREY, 1)

        cv2.imshow("Activity Monitor", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if key == ord(" "):
            now = time.time()
            if now - last_capture_t < CAPTURE_COOLDOWN:
                continue
            last_capture_t = now

            ok = face_auth.capture_sample(frame)   # use original (not flipped) for accuracy
            if ok:
                print(f"  [REGISTER] Captured {face_auth.samples_collected}/{CAPTURE_COUNT}")
                # Quick green flash
                flash = display.copy()
                cv2.rectangle(flash, (0,0), (fw, fh), CLR_GREEN, 6)
                cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)
                cv2.imshow("Activity Monitor", display)
                cv2.waitKey(150)
            else:
                # No face detected — brief red border
                err = display.copy()
                cv2.rectangle(err, (0,0), (fw, fh), CLR_RED, 6)
                cv2.putText(err, "No face detected!", (14, fh-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_RED, 2)
                cv2.addWeighted(err, 0.5, display, 0.5, 0, display)
                cv2.imshow("Activity Monitor", display)
                cv2.waitKey(300)

            if face_auth.samples_collected >= CAPTURE_COUNT:
                if face_auth.finalize_registration():
                    print("[REGISTER] Embedding locked! Starting tracking...")
                    # Success splash
                    splash = np.full_like(display, 15)
                    cv2.putText(splash, "REGISTERED!", (fw//2-140, fh//2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, CLR_GREEN, 2)
                    cv2.putText(splash, "Starting tracking...", (fw//2-150, fh//2+50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_GREY, 1)
                    cv2.imshow("Activity Monitor", splash)
                    cv2.waitKey(1000)
                    return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

import concurrent.futures

def phase_infer(cap: cv2.VideoCapture,
                face_auth: FaceAuth,
                tracker: TargetTracker,
                activity: ActivityDetector):
    """
    Continuous face-gated activity detection.
    Runs FaceAuth in a background thread to guarantee high FPS (no stutter).
    Returns "quit" or "reregister".
    """
    print("[INFER] Tracking active. Q=Quit  R=Re-register\n")

    fps_time   = time.time()
    fps_val    = 0.0

    # Thread pool for non-blocking face recognition
    auth_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    auth_future = None

    def _run_auth_task(f):
        return face_auth.identify(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        display = cv2.flip(frame, 1)
        fh, fw  = display.shape[:2]

        # ── FPS calculation ──────────────────────────────────────────────────
        now      = time.time()
        fps_val  = 0.9 * fps_val + 0.1 * (1.0 / max(now - fps_time, 1e-9))
        fps_time = now

        # ── Non-Blocking Face Recognition ────────────────────────────────────
        # If no check is running, start one
        if auth_future is None:
            # Pass a COPY of the un-flipped frame so it's thread-safe
            auth_future = auth_pool.submit(_run_auth_task, frame.copy())
        
        # Check if the thread finished
        if auth_future is not None and auth_future.done():
            result = auth_future.result()
            if result is not None:
                raw_bbox, sim = result
                flipped_bbox = _flip_bbox(raw_bbox, fw)
                tracker.update(flipped_bbox, sim)
            else:
                tracker.update(None)
            auth_future = None  # Ready for the next check immediately

        # ── Activity Detection (every frame, very fast CPU pipeline) ─────────
        activity_result = activity.predict(frame)

        # ── If target lost → reset activity state ────────────────────────────
        if not tracker.is_locked:
            activity.reset()

        # ── RENDERING ────────────────────────────────────────────────────────
        if tracker.is_locked:
            _draw_locked(display, tracker, activity_result, fps_val)
        else:
            _draw_searching(display, tracker, fps_val)

        cv2.imshow("Activity Monitor", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # Clean up thread pool
            auth_pool.shutdown(wait=False)
            return "quit"
        elif key == ord("r"):
            auth_pool.shutdown(wait=False)
            activity.reset()
            tracker.reset()
            return "reregister"


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_locked(display, tracker: TargetTracker, res: dict, fps: float):
    fh, fw = display.shape[:2]
    bbox   = tracker.bbox
    sim    = tracker.similarity
    p_col  = tuple(res["posture_color"])
    i_col  = tuple(res["intake_color"])

    # Target bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        cv2.rectangle(display, (x1, y1), (x2, y2), CLR_GREEN, 2)
        cv2.putText(display, f"Registered User  {sim:.2f}",
                    (x1, max(y1-8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, CLR_GREEN, 1, cv2.LINE_AA)

    # ── Top bar ─────────────────────────────────────────────────────────────
    _bar(display, height=78, color=CLR_DARK)
    # Posture
    cv2.putText(display, "POSTURE", (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_GREY, 1)
    cv2.putText(display, f"{res['posture_icon']} {res['posture_label']}",
                (12, 58), cv2.FONT_HERSHEY_DUPLEX, 1.0, p_col, 2, cv2.LINE_AA)
    # FPS top-right
    cv2.putText(display, f"{fps:.0f} FPS", (fw-90, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)

    # Intake badge (right half)
    if res["is_intake"]:
        mid = fw // 2
        cv2.line(display, (mid, 0), (mid, 78), (50, 50, 50), 1)
        _tint_region(display, 0, 78, mid, fw, i_col)
        icon = "[EAT]" if res["intake"] == "EATING" else "[SIP]"
        cv2.putText(display, "INTAKE", (mid+12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(display, f"{icon} {res['intake']}",
                    (mid+12, 58), cv2.FONT_HERSHEY_DUPLEX, 1.0, i_col, 2, cv2.LINE_AA)
        cv2.putText(display, f"bites:{res['bite_count']}  bpm:{res['bpm']}",
                    (fw-240, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, i_col, 1)

    # ── Bottom bar ───────────────────────────────────────────────────────────
    bar_y = fh - 48
    _bar_bottom(display, bar_y)
    cv2.putText(display, res["posture_desc"],
                (12, bar_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
    cv2.putText(display, f"angle:{res['body_angle']:.1f}  Q=Quit  R=Re-register",
                (12, bar_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90,90,90), 1)

    # Left posture colour strip
    cv2.rectangle(display, (0, 78), (5, bar_y), p_col, -1)


def _draw_searching(display, tracker: TargetTracker, fps: float):
    fh, fw = display.shape[:2]
    _bar(display, height=78, color=CLR_DARK)
    blink_col = CLR_YELLOW if int(time.time() * 2) % 2 == 0 else CLR_GREY
    cv2.putText(display, "SEARCHING FOR REGISTERED USER...",
                (12, 48), cv2.FONT_HERSHEY_DUPLEX, 0.80, blink_col, 2, cv2.LINE_AA)
    cv2.putText(display, f"{fps:.0f} FPS  |  Q=Quit  R=Re-register",
                (fw-330, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)


# ── Low-level drawing utils ───────────────────────────────────────────────────

def _bar(frame, height: int, color: tuple):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)


def _bar_bottom(frame, y: int):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y), (frame.shape[1], frame.shape[0]), CLR_DARK, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, y), (frame.shape[1], y), (50, 50, 50), 1)


def _tint_region(frame, y1: int, y2: int, x1: int, x2: int, color: tuple):
    roi  = frame[y1:y2, x1:x2]
    over = roi.copy()
    cv2.rectangle(over, (0, 0), (x2-x1, y2-y1), color, -1)
    cv2.addWeighted(over, 0.12, roi, 0.88, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _flip_bbox(bbox, frame_width: int):
    """Mirror a bounding box horizontally (because display is flipped)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (frame_width - x2, y1, frame_width - x1, y2)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Face-Gated Activity Monitor")
    parser.add_argument("--cam", default=0, type=int, help="Camera index (default 0)")
    args = parser.parse_args()

    # ── Startup ──────────────────────────────────────────────────────────────
    print("=== Face-Gated Activity Detection System ===\n")

    print("[Init] Loading activity pipeline...")
    activity = ActivityDetector()

    print("[Init] Loading face engine...")
    face_auth = FaceAuth()

    tracker = TargetTracker()

    print(f"[Init] Opening camera {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}.")
        sys.exit(1)

    # ── High FPS Camera Config ───────────────────────────────────────────────
    # Read frames at 640x480 from the hardware -> much faster I/O
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Activity Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Activity Monitor", 1280, 720)

    # ── Main loop (supports R to re-register) ────────────────────────────────
    try:
        while True:
            # --- Phase 1: Registration ---
            ok = phase_register(cap, face_auth)
            if not ok:
                break   # user pressed Q

            # --- Phase 2: Inference ---
            action = phase_infer(cap, face_auth, tracker, activity)
            if action == "quit":
                break
            # action == "reregister" → loop back to phase 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[Bye] Camera released. Session ended.")


if __name__ == "__main__":
    main()
