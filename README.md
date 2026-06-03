# 🧠 CogniFlow

> **An Explainable, Real-Time, Hardware-Free Multimodal Framework for Cognitive Focus Monitoring**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
[![Accuracy](https://img.shields.io/badge/RF%20Accuracy-83.3%25-brightgreen.svg)]()

CogniFlow is a fully operational desktop application that computes a **personalised cognitive focus score (0–100) every 2 seconds** by fusing five independent behavioural and physiological signals — all without any specialised hardware beyond a standard webcam, keyboard, and mouse.

---

## 📌 Key Features

- 🎯 **Real-time focus scoring** every 2 seconds (score: 0–100)
- 👁️ **Eye Aspect Ratio (EAR)** — drowsiness & eye closure detection via MediaPipe FaceMesh
- 👄 **Mouth Aspect Ratio (MAR)** — yawn detection via 4-state FSM (zero false positives)
- ⌨️ **Keyboard flight-time variance** — typing rhythm as cognitive engagement proxy
- 🖱️ **Mouse movement activity** — supplementary behavioural signal
- 🪟 **Foreground window context** — application-level behavioural scoring
- 🔒 **Privacy-preserving** — fully on-device, no video/biometric data stored
- 🤖 **Random Forest classifier** — 83.3% test accuracy on 1,792 real session rows
- ⚡ **Lightweight** — ~12.7% CPU, <148 MB RAM at 30 fps

---

## 🏗️ System Architecture

```
Personal Calibration (30s)
        ↓
┌──────────────────────────────────────────────┐
│  Eye (EAR) │ Yawn (MAR) │ Keyboard │ Mouse │ Window │
└──────────────────────────────────────────────┘
        ↓                          ↓
  Rule-based scoring         ML Validation
  (per-frame analysis)    (Random Forest)
        ↓
  Weighted Fusion — 8-reading EWMA
  eye 35% · typing 25% · context 20% · yawn 15% · mouse 5%
        ↓
  Focus Label (every 2s) → System Tray Overlay
```

---

## 🎯 Focus Labels

| Score | Label       | Tray Color |
|-------|-------------|------------|
| 85–100 | Deep Focus  | 🟢 Green   |
| 70–84  | Focused     | 🟢 Green   |
| 55–69  | Moderate    | 🟡 Yellow  |
| 40–54  | Drifting    | 🟠 Orange  |
| 25–39  | Fatigued    | 🔴 Red     |
| 0–24   | Exhausted   | 🔴 Red     |

---

## 📁 Project Structure

```
CogniFlow/
├── main.py                 # Entry point & orchestrator
├── engine.py               # Core fusion brain
├── webcam_sensor.py        # EAR/MAR/FaceMesh extraction
├── keyboard_sensor.py      # Flight-time variance
├── mouse_sensor.py         # Movement & click activity
├── window_sensor.py        # Foreground app detection
├── face_mesh_viewer.py     # Live webcam visual display
├── tray.py                 # Animated system-tray icon
├── dashboard.py            # Session analytics (matplotlib)
├── models/
│   ├── cogniflow_rf_model.pkl
│   ├── le_state.pkl
│   └── le_activity.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- Webcam (720p/1080p, 30fps)
- Windows 10/11 (primary), macOS 12+, or Ubuntu 20.04+

### Setup

```bash
# Clone the repository
git clone https://github.com/Karthikeyan8490/CogniFlow.git
cd CogniFlow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run CogniFlow
python main.py
```

---

## 📦 Dependencies

```
mediapipe>=0.10.0
opencv-python>=4.8.0
pynput>=1.7.6
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tkinter          # built-in
pystray>=0.19.4
Pillow>=10.0.0
psutil>=5.9.0
joblib>=1.3.0
requests>=2.31.0
```

---

## 🧪 ML Performance

| Metric                  | Value  |
|-------------------------|--------|
| Training samples         | 1,792  |
| Test accuracy            | 83.3%  |
| 5-fold CV accuracy       | 75.4%  |
| Estimators               | 100    |
| Max depth                | 10     |
| Min samples per leaf     | 2      |

### Feature Importance (Gini)

| Feature                  | Importance |
|--------------------------|------------|
| Eye Aspect Ratio (EAR)   | 34.9%      |
| Yawn Count               | 34.4%      |
| Typing Variance          | 15.2%      |
| Application Context      | 10.5%      |
| Mouse Displacement       | 5.0%       |

---

## 🛡️ Privacy

- ❌ No raw video frames stored
- ❌ No keystrokes logged
- ❌ No biometric templates saved
- ✅ Only per-second aggregate feature vectors (7 values) written to AES-encrypted local log
- ✅ No internet connection required
- ✅ Fully on-device processing

---

## 👨‍💻 Authors

| Name | Roll No |
|------|---------|
| Bukka Karthikeyan | 2451-22-733-150 |
| Puliyoju Varshith | 2451-22-733-176 |
| Madishetty Abishek | 2451-22-733-177 |

**Guide:** P. Greeshma, Asst. Professor, Dept. of CSE  
**Institution:** MVSR Engineering College, Hyderabad (Affiliated to Osmania University)  
**Academic Year:** 2025–26

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
