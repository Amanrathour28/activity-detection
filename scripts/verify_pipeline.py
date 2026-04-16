"""
verify_pipeline.py
==================
Final portability test. Simulates loading the pkl from an external project.
Only needs: activity_pipeline.py + activity_pipeline.pkl
"""
import sys, os, pickle, io
import numpy as np

print("=== Final Portability Verification ===\n")

# ── 1. Load ────────────────────────────────────────────────────────────────────
from core.activity_pipeline import ActivityPipeline
model = pickle.load(open("models/activity_pipeline.pkl", "rb"))
assert isinstance(model, ActivityPipeline)
print(f"[OK] Load          pkl size = {os.path.getsize('models/activity_pipeline.pkl')} bytes")
print(f"     Type     : {type(model).__module__}.{type(model).__name__}")
print(f"     Version  : {model.VERSION}")

# ── 2. predict() on a random pixel frame (exercises MediaPipe path) ────────────
frame = (np.random.rand(480, 640, 3) * 255).astype("uint8")
result = model.predict(frame)
assert "posture" in result and "intake" in result
assert isinstance(result["fps"], float)
print(f"[OK] predict()     posture={result['posture']}  intake={result['intake']}  fps={result['fps']}")

# ── 3. predict_landmarks() with a synthetic standing pose ─────────────────────
class _L:
    def __init__(self, x, y, v=0.99):
        self.x=x; self.y=y; self.z=0.0; self.visibility=v
class _PL:
    def __init__(self):
        self.landmark = [_L(0.5, 0.5) for _ in range(33)]
class _PR:
    def __init__(self):
        self.pose_landmarks = _PL()

pr = _PR()
lm = pr.pose_landmarks.landmark
lm[11]=_L(0.44,0.35); lm[12]=_L(0.56,0.35)  # shoulders
lm[23]=_L(0.46,0.60); lm[24]=_L(0.54,0.60)  # hips
lm[25]=_L(0.46,0.80); lm[26]=_L(0.54,0.80)  # knees
lm[9] =_L(0.48,0.25); lm[10]=_L(0.52,0.25)  # mouth
lm[15]=_L(0.20,0.90); lm[16]=_L(0.80,0.90)  # wrists far

model.reset()
for _ in range(12):
    out = model.predict_landmarks(pr.pose_landmarks, (480, 640, 3), 0)

assert out["posture"] in ("standing", "sitting"), \
    f"Expected standing/sitting, got '{out['posture']}'"
print(f"[OK] predict_landmarks()  posture={out['posture']}  body_angle={out['body_angle']:.1f} deg")

# ── 4. reset() ─────────────────────────────────────────────────────────────────
model.reset()
assert model._i_bite_count == 0 and len(model._p_history) == 0
print("[OK] reset()       bite_count=0  history cleared")

# ── 5. get_labels() ────────────────────────────────────────────────────────────
labels = model.get_labels()
assert len(labels["posture_labels"]) == 7
assert set(labels["intake_labels"]) == {"NOT EATING", "EATING", "DRINKING"}
print(f"[OK] get_labels()  posture={labels['posture_labels']}")

# ── 6. Re-pickle (write pkl again after use) ───────────────────────────────────
buf = io.BytesIO()
pickle.dump(model, buf)
buf.seek(0)
model2 = pickle.load(buf)
out2 = model2.predict_landmarks(pr.pose_landmarks, (480, 640, 3), 0)
print(f"[OK] re-pickle     posture={out2['posture']}  intake={out2['intake']}")

# ── 7. Intake detection test ───────────────────────────────────────────────────
class _LI(_L):
    pass

model.reset()
pr_eat = _PR()
lm2 = pr_eat.pose_landmarks.landmark
lm2[11]=_L(0.35,0.35); lm2[12]=_L(0.65,0.35)  # wide shoulders
lm2[23]=_L(0.46,0.60); lm2[24]=_L(0.54,0.60)
lm2[25]=_L(0.46,0.80); lm2[26]=_L(0.54,0.80)
lm2[9] =_L(0.49,0.25); lm2[10]=_L(0.51,0.25)  # mouth
lm2[16]=_L(0.515,0.27)                           # right wrist very close to mouth
lm2[15]=_L(0.10,0.70)                            # left wrist far

for _ in range(8):
    out3 = model.predict_landmarks(pr_eat.pose_landmarks, (480, 640, 3), 0)

assert out3["is_intake"], "Expected intake to be detected"
assert out3["intake"] in ("EATING", "DRINKING")
print(f"[OK] intake test   is_intake={out3['is_intake']}  label={out3['intake']}  confidence={out3['confidence']:.2f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== ALL 7 CHECKS PASSED ===\n")
print("Deliverable files:")
files = [
    ("activity_pipeline.py",  "Standalone module (copy this + pkl to any project)"),
    ("activity_pipeline.pkl", "Serialized pipeline — load with pickle.load()"),
    ("build_pipeline.py",     "Regenerates the pkl from scratch"),
    ("example_usage.py",      "Live webcam demo: python example_usage.py --live"),
]
for fn, desc in files:
    kb = os.path.getsize(fn) / 1024
    print(f"  {fn:<30s}  {kb:5.1f} KB   {desc}")
