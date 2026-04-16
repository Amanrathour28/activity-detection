"""
build_pipeline.py
==================
Builds and saves activity_pipeline.pkl.

Imports ActivityPipeline from activity_pipeline.py so that the pkl
correctly stores the class path as `activity_pipeline.ActivityPipeline`.
This means the pkl is loadable in ANY project that has activity_pipeline.py.

Run:
    python build_pipeline.py

Output:
    activity_pipeline.pkl
"""

import pickle
import sys
import time
import numpy as np

# Import from the standalone module — this sets the correct class path in pkl
from activity_pipeline import ActivityPipeline, POSTURE_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TEST HELPERS (no camera needed)
# ══════════════════════════════════════════════════════════════════════════════

def _lm(x=0.5, y=0.5, z=0.0, vis=0.99):
    class L:
        pass
    o = L()
    o.x = x; o.y = y; o.z = z; o.visibility = vis
    return o


def _pose(lm_list):
    class PL:
        def __init__(self, lms): self.landmark = lms
    class PR:
        def __init__(self, lms): self.pose_landmarks = PL(lms)
    return PR(lm_list)


def _standing_pose():
    lms = [_lm() for _ in range(33)]
    lms[11] = _lm(0.44, 0.35);  lms[12] = _lm(0.56, 0.35)   # shoulders
    lms[23] = _lm(0.46, 0.60);  lms[24] = _lm(0.54, 0.60)   # hips
    lms[25] = _lm(0.46, 0.80);  lms[26] = _lm(0.54, 0.80)   # knees
    lms[9]  = _lm(0.48, 0.25);  lms[10] = _lm(0.52, 0.25)   # mouth
    lms[15] = _lm(0.20, 0.90);  lms[16] = _lm(0.80, 0.90)   # wrists (far)
    return lms


def _lying_pose():
    lms = [_lm() for _ in range(33)]
    lms[11] = _lm(0.20, 0.50);  lms[12] = _lm(0.20, 0.55)
    lms[23] = _lm(0.80, 0.50);  lms[24] = _lm(0.80, 0.55)
    lms[25] = _lm(0.90, 0.50);  lms[26] = _lm(0.90, 0.55)
    lms[9]  = _lm(0.10, 0.50);  lms[10] = _lm(0.10, 0.55)
    lms[15] = _lm(0.05, 0.50);  lms[16] = _lm(0.05, 0.55)
    return lms


def _intake_pose():
    lms = [_lm() for _ in range(33)]
    lms[11] = _lm(0.35, 0.35);  lms[12] = _lm(0.65, 0.35)   # shoulders (wide)
    lms[23] = _lm(0.46, 0.60);  lms[24] = _lm(0.54, 0.60)
    lms[25] = _lm(0.46, 0.80);  lms[26] = _lm(0.54, 0.80)
    lms[9]  = _lm(0.49, 0.25);  lms[10] = _lm(0.51, 0.25)   # mouth at (320,120)
    lms[16] = _lm(0.515, 0.27)                                 # right wrist near mouth
    lms[15] = _lm(0.10, 0.70)                                  # left wrist far
    return lms


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(pipeline: ActivityPipeline) -> bool:
    SHAPE = (480, 640, 3)
    all_ok = True

    print("\n  Validation tests")
    print("  " + "-" * 54)

    # Test 1: Standing
    pipeline.reset()
    pr = _pose(_standing_pose())
    for _ in range(10):
        out = pipeline.predict_landmarks(pr.pose_landmarks, SHAPE, 0)
    ok = out["posture"] in ("standing", "sitting")
    all_ok = all_ok and ok
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} Upright person  -> posture='{out['posture']}'")

    # Test 2: Lying
    pipeline.reset()
    pr = _pose(_lying_pose())
    for _ in range(10):
        out = pipeline.predict_landmarks(pr.pose_landmarks, SHAPE, 0)
    ok = out["posture"] in ("lying", "sleeping", "falling")
    all_ok = all_ok and ok
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} Lying down      -> posture='{out['posture']}'")

    # Test 3: Intake
    pipeline.reset()
    pr = _pose(_intake_pose())
    for _ in range(8):
        out = pipeline.predict_landmarks(pr.pose_landmarks, SHAPE, 0)
    ok = out["is_intake"] and out["intake"] in ("EATING", "DRINKING")
    all_ok = all_ok and ok
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} Wrist near mouth -> intake='{out['intake']}', is_intake={out['is_intake']}")

    # Test 4: Pickle round-trip
    import io
    buf = io.BytesIO()
    pickle.dump(pipeline, buf)
    buf.seek(0)
    loaded = pickle.load(buf)
    ok = isinstance(loaded, ActivityPipeline) and loaded.VERSION == ActivityPipeline.VERSION
    all_ok = all_ok and ok
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} Pickle round-trip -> class={type(loaded).__name__} v{loaded.VERSION}")

    # Test 5: Labels
    labels = loaded.get_labels()
    ok = "standing" in labels["posture_labels"] and "EATING" in labels["intake_labels"]
    all_ok = all_ok and ok
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} get_labels()    -> posture={labels['posture_labels']}")

    # Test 6: Module path is correct
    mod = type(pipeline).__module__
    ok = mod == "activity_pipeline"
    all_ok = all_ok and ok
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} Module path     -> '{mod}' (must be 'activity_pipeline')")

    print("  " + "-" * 54)
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT = "activity_pipeline.pkl"

    print("=" * 60)
    print("  Activity + Intake Detection Pipeline Builder")
    print("  Face recognition: EXCLUDED")
    print("=" * 60)

    # 1. Build
    print("\n[1/3] Building pipeline...")
    pipeline = ActivityPipeline(
        smoothing_frames        = 8,
        min_intake_frames       = 4,
        sitting_hip_y_threshold = 0.52,
        sitting_knee_diff_max   = 0.18,
        walk_motion_threshold   = 1500.0,
        shoulder_ratio          = 0.60,
    )
    print(f"      Version  : {pipeline.VERSION}")
    print(f"      Module   : {type(pipeline).__module__}.{type(pipeline).__name__}")
    print(f"      Labels   : {list(POSTURE_ACTIONS.keys())}")

    # 2. Validate
    print("\n[2/3] Validating (no camera needed)...")
    ok = validate(pipeline)

    if not ok:
        print("\n  [ERROR] Validation failed — pkl NOT saved.")
        sys.exit(1)

    print("\n  All tests passed.")

    # 3. Save (clean state)
    pipeline.reset()
    print(f"\n[3/3] Saving to {OUTPUT}...")
    with open(OUTPUT, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

    import os
    kb = os.path.getsize(OUTPUT) / 1024
    print(f"      Saved: {OUTPUT}  ({kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("  DONE")
    print()
    print("  To use in any project:")
    print("    1. Copy activity_pipeline.py  alongside your code")
    print("    2. Copy activity_pipeline.pkl alongside your code")
    print("    3. Then:")
    print()
    print("       import pickle")
    print("       from activity_pipeline import ActivityPipeline")
    print("       model = pickle.load(open('activity_pipeline.pkl','rb'))")
    print("       result = model.predict(bgr_frame)   # OpenCV BGR frame")
    print("=" * 60)
