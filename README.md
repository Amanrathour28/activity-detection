# Activity & Intake Detection Monitor

This project uses a minimal, production-ready architecture to perform real-time posture and intake detection via MediaPipe Pose rules. 

**Note: Face Authentication has been completely stripped from this project for simplicity and speed.**

## 📂 Project Structure

```text
├── app/
│   └── run_inference.py        # The Live Webcam Monitor
│
├── core/
│   ├── activity_pipeline.py    # Standalone engine for MediaPipe rules
│   └── build_pipeline.py       # Developer script to regenerate the .pkl
│
├── models/
│   └── activity_pipeline.pkl   # Serialized pipeline / state config
│
├── scripts/
│   └── verify_pipeline.py      # Automated pipeline tests
│
└── requirements.txt
```

---

## 🚀 Quickstart

**1. Install minimal dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the secure live monitor**
```bash
python app/run_inference.py
```
* The pipeline will instantly track your skeleton and display:
  * **Posture**: Standing, Sitting, Falling, Lying, Sleeping.
  * **Intake**: Eating, Drinking, Bites Per Minute.

---

## 🔧 Features

* **Dependency-Lite Pipeline**: Relies purely on `mediapipe` and runs extremely fast on CPU.
* **Portable Core**: The `.pkl` format allows exporting the activity logic to other projects easily. Just grab `core/activity_pipeline.py` and `models/activity_pipeline.pkl`.
