"""
example_usage.py
=================
Demonstrates how to load and use activity_pipeline.pkl.

Usage:
    python example_usage.py              # Static image test (no camera)
    python example_usage.py --live       # Live webcam feed
    python example_usage.py --live --cam 1   # Alternate camera index

Requirements:
    pip install mediapipe>=0.10.0 numpy>=1.21 opencv-python>=4.5
"""

import pickle
import argparse
import time
import sys
import cv2
import numpy as np

# This import makes ActivityPipeline resolvable when pickle.load() is called.
# Keep this line — it is required even though ActivityPipeline isn't used directly.
from activity_pipeline import ActivityPipeline  # noqa: F401

PKL_PATH = "activity_pipeline.pkl"


def load_pipeline():
    """Load the pkl — this is all you need in any external app."""
    with open(PKL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print(f"[OK] Loaded ActivityPipeline v{pipeline.VERSION}")
    print(f"     Posture labels : {pipeline.get_labels()['posture_labels']}")
    print(f"     Intake labels  : {pipeline.get_labels()['intake_labels']}")
    return pipeline


def demo_static(pipeline):
    """
    Test with a synthetic black frame (no camera needed).
    Shows the full output dict.
    """
    print("\n── Static frame test ────────────────────────────────────")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.predict(dummy_frame)

    print("  Input : 640×480 black BGR frame")
    print("  Output:")
    for k, v in result.items():
        print(f"    {k:25s}: {v}")

    print("\n  (No pose landmarks detected in a black frame → 'unknown')")
    print("  This is correct behaviour.\n")


def demo_live(pipeline, cam_index=0):
    """
    Live webcam inference demo.
    Overlays posture + intake predictions on the camera feed.
    Press Q to quit, R to reset session.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n── Live webcam demo ─────────────────────────────────────")
    print("  Controls: Q=Quit  R=Reset session")
    cv2.namedWindow("ActivityPipeline Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ActivityPipeline Demo", 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            display = cv2.flip(frame, 1)

            # ── Core inference call ──────────────────────────────────────────
            result = pipeline.predict(frame)   # un-flipped frame for inference
            # ────────────────────────────────────────────────────────────────

            fh, fw = display.shape[:2]
            p_color = tuple(result["posture_color"])
            i_color = tuple(result["intake_color"])

            # ── Draw posture banner ──────────────────────────────────────────
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 78), (12, 12, 12), -1)
            cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)
            cv2.putText(display,
                        f"{result['posture_icon']} {result['posture_label']}",
                        (12, 58), cv2.FONT_HERSHEY_DUPLEX,
                        1.0, p_color, 2, cv2.LINE_AA)
            cv2.putText(display, "POSTURE", (12, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(display, f"{result['fps']:.0f} FPS",
                        (fw - 90, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(display,
                        f"angle: {result['body_angle']:.1f}°",
                        (fw - 160, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (100, 100, 100), 1, cv2.LINE_AA)

            # ── Draw intake badge ────────────────────────────────────────────
            if result["is_intake"]:
                mid = fw // 2
                cv2.line(display, (mid, 0), (mid, 78), (60, 60, 60), 1)
                roi = display[0:78, mid:fw]
                over = roi.copy()
                cv2.rectangle(over, (0, 0), (fw - mid, 78), i_color, -1)
                cv2.addWeighted(over, 0.14, roi, 0.86, 0, roi)
                display[0:78, mid:fw] = roi
                icon = "[EAT]" if result["intake"] == "EATING" else "[SIP]"
                cv2.putText(display,
                            f"{icon} {result['intake']}",
                            (mid + 12, 58), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, i_color, 2, cv2.LINE_AA)
                cv2.putText(display, "INTAKE", (mid + 12, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display,
                            f"bites:{result['bite_count']} bpm:{result['bpm']}",
                            (fw - 240, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            i_color, 1, cv2.LINE_AA)

            # ── Draw bottom stats bar ────────────────────────────────────────
            bar_y = fh - 60
            overlay2 = display.copy()
            cv2.rectangle(overlay2, (0, bar_y), (fw, fh), (12, 12, 12), -1)
            cv2.addWeighted(overlay2, 0.82, display, 0.18, 0, display)
            cv2.line(display, (0, bar_y), (fw, bar_y), (50, 50, 50), 1)
            cv2.putText(display, result["posture_desc"], (12, bar_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (200, 200, 200), 1, cv2.LINE_AA)
            horiz = f"  | on floor: {result['horizontal_duration']:.0f}s" \
                    if result["posture"] in ("sleeping", "lying") else ""
            cv2.putText(display,
                        f"body_angle: {result['body_angle']:.1f}°{horiz}",
                        (12, bar_y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(display, "Q=Quit  R=Reset",
                        (fw - 180, fh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (70, 70, 70), 1, cv2.LINE_AA)

            # ── Left colour strip ────────────────────────────────────────────
            cv2.rectangle(display, (0, 78), (6, bar_y), p_color, -1)

            cv2.imshow("ActivityPipeline Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                pipeline.reset()
                print("  [Reset] Session cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActivityPipeline usage example")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam demo")
    parser.add_argument("--cam", default=0, type=int,
                        help="Camera index (default 0)")
    args = parser.parse_args()

    pipeline = load_pipeline()

    if args.live:
        demo_live(pipeline, cam_index=args.cam)
    else:
        demo_static(pipeline)
        print("Run with --live to see the webcam demo.")
