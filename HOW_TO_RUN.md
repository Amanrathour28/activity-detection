# How to Run — AI Activity Detection

## What This Project Does

This is a **real-time, face-gated human activity monitoring system** built in Python. It uses two AI libraries working together:

| Library | Role |
|---|---|
| **InsightFace** | Detects and recognises your face (authentication gate) |
| **MediaPipe Pose** | Tracks your body skeleton to classify activities |

### Full Pipeline

```
Webcam frame
    │
    ▼
[InsightFace] ─── Not you? → Show "AUTHENTICATION REQUIRED" screen
    │
    └── It's you! ↓
        │
        ▼
    [MediaPipe Pose]
        ├── Posture: Standing / Walking / Sitting / Falling / Lying / Sleeping
        └── Intake:  Eating / Drinking (wrist near mouth) + Bites Per Minute
        │
        ▼
    [HUD Overlay] → Display on screen in real-time
```

### Project Structure

```
activity-detection/
├── run_inference.py        ← Main entry point (live camera)
├── register_face.py        ← One-time face registration (REQUIRED first)
├── capture_demo.py         ← 8-second annotated demo, no face auth needed
├── requirements.txt        ← All Python dependencies
├── HOW_TO_RUN.md           ← This file
│
├── face_auth/
│   ├── face_engine.py      ← InsightFace wrapper (detect + embed + match)
│   └── config.py           ← Auth thresholds you can tune
│
├── detectors/
│   ├── posture_detector.py ← Body angle → posture classification
│   └── intake_detector.py  ← Wrist-to-mouth proximity → eating/drinking
│
├── display/
│   └── hud.py              ← All screen overlays (skeleton, badges, HUD)
│
├── model_code/             ← FIR thermal dataset model (training only)
│   └── model.py            ← DW-CNN2D / 3D-CNN / CNN-LSTM definitions
│
└── RGB_hand_labeled/       ← Sample YOLO-annotated frames (reference data)
```

---

## Quick Start

### Step 1 — Install Python 3.10+

Make sure Python 3.10 or newer is installed and on your PATH:

```powershell
python --version
```

### Step 2 — Create a virtual environment (recommended)

```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install all dependencies

```powershell
pip install -r requirements.txt
```

> **First run note:** `insightface` will automatically download the `buffalo_s`
> face model pack (~80 MB) from the internet. This happens once only.

---

## Running the System

### ① Register your face — run once before anything else

```powershell
python register_face.py
```

- Your webcam opens with a live preview.
- Press **SPACE** 5 times to capture photos of your face (vary angle slightly).
- Press **Q** to abort.
- On success, saves `face_auth/registered_face.npy`.

---

### ② Live detection — the main program

```powershell
python run_inference.py
```

**What you'll see:**

| Screen state | Meaning |
|---|---|
| Dark overlay + pulsing green text | Waiting — look at the camera |
| "AUTHORIZATION SUCCESSFUL" green banner | Your face matched ✓ |
| 3-second countdown | System getting ready |
| Full HUD with posture/intake labels | Fully active — detecting your activity |

**Keyboard shortcuts (while the window is focused):**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot → `output_results/screenshots/` |
| `R` | Reset session counters (bite count, posture history) |

**Optional command-line flags:**

```powershell
# Use a different camera (if your default cam is busy or wrong)
python run_inference.py --cam 1

# Browse the FIR thermal dataset (no face auth required)
python run_inference.py --dataset

# Browse a single video from the dataset folder
python run_inference.py --video video105

# Limit frames shown per video in dataset mode
python run_inference.py --max-frames 50
```

---

### ③ Quick demo — no face registration needed

```powershell
python capture_demo.py
```

Runs posture + intake detection for 8 seconds and auto-saves 6 annotated
screenshots to `output_results/demo_captures/`. Great for verifying that
MediaPipe and your camera are working correctly.

---

## Tuning Authentication (Optional)

Open `face_auth/config.py` and adjust these values:

| Parameter | Default | What it does |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.45` | Cosine similarity cut-off. Lower = more lenient |
| `AUTH_CONSECUTIVE_FRAMES` | `10` | Consecutive matches before unlocking |
| `IDENTITY_GRACE_SECONDS` | `2.0` | Stay authenticated this many seconds after face leaves frame |
| `FACE_SKIP_FRAMES` | `3` | Re-run face embedding every N frames (higher = less CPU) |
| `EMA_ALPHA` | `0.3` | Face-box smoothing (0 = frozen, 1 = no smoothing) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot open camera 0` | Try `python run_inference.py --cam 1` |
| `No registered face found` | Run `python register_face.py` first |
| `insightface is not installed` | `pip install insightface onnxruntime` |
| `mediapipe not installed` | `pip install mediapipe` |
| Very low FPS | Increase `FACE_SKIP_FRAMES` in `config.py`; use `model_complexity=0` |
| Face never authenticates | Improve lighting; lower `SIMILARITY_THRESHOLD` to `0.35` |
| `No face detected` during registration | Move closer to camera; improve lighting |
