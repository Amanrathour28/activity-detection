"""
app/run_inference.py
======================
Unified Inference Pipeline combining Face Authentication and Activity Detection.

1. FaceAuthentication (InsightFace) ensures only the registered user is tracked.
2. ActivityPipeline (MediaPipe) runs Posture and Intake detection.
"""

import cv2
import sys
import time
import pickle
import numpy as np
from pathlib import Path

# Setup paths so we can import modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from face_auth.face_engine import FaceEngine
from face_auth import config as auth_config
from core.activity_pipeline import ActivityPipeline

# Paths
REG_FACE_PATH = project_root / "face_auth" / "registered_face.npy"
PIPELINE_PATH = project_root / "models" / "activity_pipeline.pkl"


class GatedInference:
    def __init__(self):
        print("[App] Initializing systems...")

        # 1. Load Face Engine
        if not REG_FACE_PATH.exists():
            print(f"\n[ERROR] Registered face not found: {REG_FACE_PATH}")
            print("Please run `python app/register_face.py` first.\n")
            sys.exit(1)
        self.registered_emb = np.load(str(REG_FACE_PATH))
        self.face_engine = FaceEngine()

        # 2. Load Activity Pipeline
        if not PIPELINE_PATH.exists():
            print(f"\n[ERROR] Pipeline model not found: {PIPELINE_PATH}")
            print("Please run `python core/build_pipeline.py` first.\n")
            sys.exit(1)
            
        with open(str(PIPELINE_PATH), "rb") as f:
            self.activity_pipeline = pickle.load(f)
        print(f"[App] Loaded ActivityPipeline v{self.activity_pipeline.VERSION}")

        # Auth State
        self.auth_consecutive = 0
        self.is_authenticated = False
        self.last_bbox = None
        self.target_fps = 30

    def run(self, cam_index=0):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {cam_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n── Live inference started ──────────────────────────────")
        print("  Controls: Q=Quit  R=Reset counters")
        
        cv2.namedWindow("Secure Activity Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Secure Activity Monitor", 1280, 720)

        # To limit FPS so we don't cook the CPU
        frame_time = 1.0 / self.target_fps
        
        try:
            while True:
                start_t = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                display = cv2.flip(frame, 1)

                # ── 1. FACE AUTHENTICATION ─────────────────────────────────────────
                # We only need to run full InsightFace if we are trying to authenticate 
                # or if we lost the track bounding box over the last few frames.
                # For simplicity and robustness, we run FaceEngine every frame 
                # to find the target face.
                
                faces = self.face_engine.detect_and_embed(frame)
                match_result = self.face_engine.best_match(faces, self.registered_emb)

                if match_result:
                    bbox, sim = match_result
                    self.last_bbox = bbox
                    self.auth_consecutive += 1
                    if self.auth_consecutive >= auth_config.CONSECUTIVE_FRAMES_REQ:
                        self.is_authenticated = True
                else:
                    self.auth_consecutive = 0
                    self.is_authenticated = False
                    self.last_bbox = None

                fh, fw = display.shape[:2]

                # ── 2. ACTIVITY DETECTION (only if auth passes) ────────────────────
                if self.is_authenticated:
                    # Run activity pipeline
                    res = self.activity_pipeline.predict(frame)
                    
                    # --- Overlay Rendering ---
                    p_color = res["posture_color"]
                    
                    # Top auth bar (Green)
                    cv2.rectangle(display, (0,0), (fw, 30), (40, 200, 40), -1)
                    cv2.putText(display, "AUTHENTICATED: Tracking User", (12, 22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    # Posture Banner
                    cv2.rectangle(display, (0, 30), (300, 100), (20,20,20), -1)
                    cv2.putText(display, f"{res['posture_icon']} {res['posture_label']}", 
                                (12, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, p_color, 2)
                    cv2.putText(display, "POSTURE", (12, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)
                                
                    # Intake Badge
                    if res["is_intake"]:
                        i_color = res["intake_color"]
                        cv2.rectangle(display, (fw-300, 30), (fw, 100), (20,20,20), -1)
                        icon = "[EAT]" if res["intake"] == "EATING" else "[SIP]"
                        cv2.putText(display, f"{icon} {res['intake']}", 
                                    (fw-288, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, i_color, 2)
                        cv2.putText(display, f"INTAKE  |  bites: {res['bite_count']}", 
                                    (fw-288, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

                    # Bottom stats bar
                    cv2.rectangle(display, (0, fh-40), (fw, fh), (20,20,20), -1)
                    cv2.putText(display, f"FPS: {res['fps']:.1f} | Angle: {res['body_angle']:.1f} | Activity: {res['posture_desc']}", 
                                (12, fh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                    # Draw Face bounding box on flipped display
                    if self.last_bbox is not None:
                        # Bbox coordinates must be flipped horizontally
                        x1, y1, x2, y2 = self.last_bbox
                        fx1 = fw - x2
                        fx2 = fw - x1
                        cv2.rectangle(display, (fx1, y1), (fx2, y2), (40,200,40), 2)

                else:
                    # Not authenticated rendering
                    self.activity_pipeline.reset()  # Reset history so new users start fresh
                    
                    # Top auth bar (Red)
                    prog = self.auth_consecutive / max(1, auth_config.CONSECUTIVE_FRAMES_REQ)
                    cv2.rectangle(display, (0,0), (fw, 30), (0, 0, 200), -1)
                    cv2.putText(display, "UNAUTHORIZED: Waiting for valid face...", (12, 22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                                
                    # Progress bar if partially authenticated
                    if prog > 0:
                        cv2.rectangle(display, (0,30), (int(fw * prog), 40), (0, 150, 255), -1)
                        
                    # Blur background for privacy
                    display = cv2.GaussianBlur(display, (51, 51), 0)

                cv2.imshow("Secure Activity Monitor", display)
                
                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.activity_pipeline.reset()
                    print("  [Action] Reset session counters.")
                    
                # FPS sleep
                elapsed = time.time() - start_t
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", default=0, type=int, help="Camera index")
    args = parser.parse_args()

    app = GatedInference()
    app.run(cam_index=args.cam)
