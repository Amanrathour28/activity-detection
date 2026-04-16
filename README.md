# Face-Gated Activity Detection System

Single-command system combining **face registration**, **face-locked tracking**, and **real-time activity + intake detection** in one unified pipeline.

## 📂 Project Structure

```text
run.py                      ← SINGLE ENTRY POINT (run this)
│
├── utils/
│   ├── face_auth.py        InsightFace wrapper (register + recognize)
│   ├── tracker.py          IoU bounding box tracker (single target lock)
│   └── activity.py         ActivityPipeline wrapper
│
├── core/
│   ├── activity_pipeline.py  MediaPipe rule engine
│   └── build_pipeline.py     Regenerates the .pkl
│
├── models/
│   └── activity_pipeline.pkl  Serialized pipeline
│
├── app/
│   └── run_inference.py    Activity-only viewer (no face auth)
│
└── scripts/
    └── verify_pipeline.py  Automated pipeline tests
```

---

## 🚀 Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the system (single command)**
```bash
python run.py
```
or for a different camera:
```bash
python run.py --cam 1
```

---

## 🔄 How It Works

### Phase 1 — Face Registration (no separate script needed)
- The webcam opens automatically
- Look at the camera and **press SPACE 5 times** to capture your face
- The embedding is stored in RAM (no file saved)
- The camera **stays open** — no restart required

### Phase 2 — Continuous Tracking + Activity Detection
- **Every 15 frames**: InsightFace runs to find and verify the target face
- **Every frame**: IoU tracker keeps the bounding box locked (zero extra cost)
- **Every frame**: MediaPipe Pose detects posture and intake
- If a **stranger** appears, only the registered user is tracked
- If the **target leaves frame**, detection pauses (grace period: 2 sec)
- On reappearance, tracking resumes automatically

---

## 🎛️ Live Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Re-register (restart Phase 1 without closing camera) |
| `SPACE` | Capture face sample (Phase 1 only) |

---

## 📊 Output Displayed

- **Bounding box** around registered user only
- **Posture**: Standing / Sitting / Lying / Sleeping / Falling / Walking
- **Intake**: Eating / Drinking + Bites Per Minute counter
- **Body angle** and **FPS**

---

## ⚡ Performance Design

| Optimization | Technique |
|---|---|
| Face recognition bottleneck | Only runs every **15 frames** |
| Detection downscaling | Frames **halved** (50%) for InsightFace input |
| Tracking | IoU bounding box tracker (< 1ms per frame) |
| Activity detection | MediaPipe Pose — CPU-optimized TFLite |
| Multi-person isolation | Only target bbox passed to activity pipeline |
