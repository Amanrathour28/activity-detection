# Unified Face-Gated Activity Detection

This project uses a minimal, production-ready architecture to perform real-time face authentication (InsightFace) followed by posture and intake detection (MediaPipe).

## 📂 Project Structure

```text
├── app/
│   ├── register_face.py        # 1. Run this first to enroll authorized user
│   └── run_inference.py        # 2. Main Live Viewer (FaceAuth + Activity)
│
├── core/
│   ├── activity_pipeline.py    # Standalone engine for MediaPipe rules
│   └── build_pipeline.py       # Developer script to regenerate the .pkl
│   
├── face_auth/                  # Face Engine (InsightFace wrapper)
│   ├── config.py
│   ├── face_engine.py
│   └── registered_face.npy     # Saved enrollment data
│
├── models/
│   └── activity_pipeline.pkl   # Serialized pipeline / state config
│
├── scripts/
│   ├── example_usage.py        # Demo without Face Authentication
│   └── verify_pipeline.py      # Automated tests
│
└── requirements.txt
```

---

## 🚀 Quickstart

**1. Install minimal dependencies**
```bash
pip install -r requirements.txt
```

**2. Register your face (Required)**
```bash
python app/register_face.py
```
* Look at the camera until it captures your face geometry. This creates `face_auth/registered_face.npy`.

**3. Run the secure live monitor**
```bash
python app/run_inference.py
```
* If the camera sees someone else, the screen blurs and denies access.
* If it sees you, the Activity Engine activates, detecting:
  * **Posture**: Standing, Sitting, Falling, Lying, Sleeping.
  * **Intake**: Eating, Drinking, Bites Per Minute.

---

## 🔧 Features

* **Complete Privacy Isolation**: Face Auth is isolated from Activity Detection. The user history and bite count drops to zero if an unauthorized person steps in frame.
* **Dependency-Lite Pipeline**: Posture/Intake relies purely on `mediapipe` and runs extremely fast on CPU.
* **Portable Core**: The `.pkl` format allows exporting the activity logic to other projects easily. Just grab `core/activity_pipeline.py` and `models/activity_pipeline.pkl`.
