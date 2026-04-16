"""
capture_demo.py
===============
Runs the full unified detection pipeline for 8 seconds,
then saves 6 annotated screenshots to output_results/demo_captures/
so you can see exactly what the live camera is outputting.

Run: python capture_demo.py
     python capture_demo.py --cam 1    # alternate camera index
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque

import mediapipe as mp

from detectors.posture_detector import PostureDetector
from detectors.intake_detector  import IntakeDetector
from display.hud import draw_skeleton, draw_mouth_box, draw_hud

OUTPUT = Path("output_results/demo_captures")
OUTPUT.mkdir(parents=True, exist_ok=True)

CAPTURE_AT = [2, 3, 4, 5, 6, 7]   # seconds after start to save a snapshot


def run_demo(cam_index: int = 0):
    mp_pose    = mp.solutions.pose
    mp_face    = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    # FIX: open camera inside function (not at module level) for proper lifecycle
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}. Try --cam 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    posture_det = PostureDetector()
    intake_det  = IntakeDetector()
    prev_gray   = None
    fps_buf     = deque(maxlen=30)

    saved = []
    start = time.time()

    print("Capturing demo frames for 8 seconds — please move in front of camera...")
    print("Try: standing, sitting, raising hand to mouth, tilting body\n")

    # FIX: wrap in try/finally to guarantee camera release
    try:
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          model_complexity=1) as pose_model, \
             mp_face.FaceMesh(max_num_faces=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,
                              refine_landmarks=True) as face_model:

            while True:
                t0  = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

                # Motion
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gblur = cv2.GaussianBlur(gray, (21, 21), 0)
                if prev_gray is None:
                    prev_gray = gblur
                motion    = float(np.sum(cv2.absdiff(prev_gray, gblur) > 25))
                prev_gray = gblur

                # MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pose_result = pose_model.process(rgb)
                face_result = face_model.process(rgb)
                rgb.flags.writeable = True

                # Detect
                posture = posture_det.update(pose_result, motion)
                intake  = intake_det.update(pose_result, face_result, frame.shape)

                # Draw
                draw_skeleton(frame, pose_result, mp_drawing, mp_pose)
                draw_mouth_box(frame, intake)

                fps_buf.append(time.time() - t0)
                fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0

                draw_hud(frame, posture, intake, fps, motion)

                # Auto-save snapshots at specific seconds
                elapsed = time.time() - start
                for sec in CAPTURE_AT:
                    if abs(elapsed - sec) < 0.12 and sec not in saved:
                        tag  = intake.label.replace(" ", "_")
                        fname = OUTPUT / f"demo_{sec:02d}s_{posture.action}_{tag}.jpg"
                        cv2.imwrite(str(fname), frame)
                        saved.append(sec)
                        print(f"  Saved: {fname.name}  [{posture.label} | {intake.label}]")

                if elapsed > 8:
                    break
    finally:
        cap.release()

    print(f"\nDone. {len(saved)} demo frames saved to: {OUTPUT}")
    print("Open that folder in File Explorer to see the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8-second annotated demo (no face auth)")
    parser.add_argument("--cam", default=0, type=int, help="Camera index (default 0)")
    args = parser.parse_args()
    run_demo(cam_index=args.cam)
