# LCGC: Lightweight Continuous Gesture Control

[![Simulation Status](https://img.shields.io/badge/Simulation-Verified-brightgreen.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A self-contained, hardware-free software simulation framework for real-time continuous gesture recognition and differential-drive robot control using normalized RGB-only hand landmarks and an INT8 depthwise-separable 1D-CNN.

---

## 📌 Project Overview

**Paper Title:** *"A Lightweight RGB Vision Framework for Real-Time Continuous Gesture Recognition and Robot Control"*  
**System Name:** LCGC (Lightweight Continuous Gesture Control)  
**Target Platform:** Edge microcomputers (e.g., Raspberry Pi 4 / 5) with low-cost RGB webcams, eliminating bulky depth sensors.

### End-to-End Pipeline
```
[RGB Webcam / Synthetic Stream]
       │
       ▼
[MediaPipe 21 Hand Landmarks]  ──> (x, y) coordinates per joint
       │
       ▼
[Wrist-Relative Palm-Scale Normalization]  ──> p_tilde_i = (p_i - p_0) / ||p_9 - p_0||_2 (42 features)
       │
       ▼
[Sliding Window Buffer]  ──> 15 frames (~0.5 s at 30 FPS, shape: 15 × 42)
       │
       ▼
[Depthwise-Separable 1D-CNN (INT8)]  ──> 6 classes (Idle, Stop, Forward, Backward, Left, Right)
       │
       ▼
[Probability Smoothing (EMA, α=0.4)]  ──> Smoothed class probability vector
       │
       ▼
[Hysteresis Filter (Threshold=0.80)]  ──> m=3 consecutive frames (m=1 for Stop)
       │
       ▼
[Command Mapping & Proportional Speed]  ──> (v, ω) with palm-proximity speed scaling
       │
       ▼
[Safety Gate (Watchdog + 20cm Ultrasonic)]  ──> Emergency stop on obstacle or hand loss
       │
       ▼
[Differential-Drive Robot Simulation]  ──> v_R = v + ωL/2, v_L = v - ωL/2
```

---

## 🔬 Research Gaps Addressed

| Gap | Prior Limitation | LCGC Solution & Evidence |
| :--- | :--- | :--- |
| **G1: Depth Sensor Dependency** | Systems rely on Kinect/ToF [Kahlouche 2022, Li 2018] | RGB-only wrist-relative & palm-scale normalization invariant to hand position, distance, and uniform scaling. |
| **G2: Low RGB Accuracy / FPS** | Raw image RGB models achieve ~70-74% acc @ 15 FPS [Lee & Ahn 2020, Keshinro 2022] | Compact 42-feature landmark windows + 1D-CNN achieve high accuracy at >100 FPS classifier inference. |
| **G3: Static-Only Micro-Models** | Sub-10KB models support only static poses without tracking [Muhtadin 2025] | 15-frame temporal sliding window with continuous EMA smoothing, hysteresis, and palm-size speed scaling. |
| **G4: Heavy Temporal Edge Models** | LSTMs/Transformers/GPUs exceed micro-controller RAM/compute [Mavsar 2025, Bandi 2021] | Depthwise-separable 1D-CNN with full-integer INT8 quantization (~24.3 KB, <=50 KB edge budget). |
| **G5: False Triggers & Lost Tracking** | Inadvertent motions trigger commands; tracking loss causes run-aways [Zeng 2025] | Confidence threshold (0.80), m=3 hysteresis, 0.5s hand-lost watchdog, and 20cm ultrasonic gate. |

---

## 📂 Project Architecture

```
lcgc/
├── config.py                 # Central configuration for all fixed constants & hyper-parameters
├── synthetic_hands.py        # Parametric 21-point hand landmark generator with anatomical variations
├── models.py                 # Architectures: B1 (MLP), B2 (LSTM), B3 (LCGC DS-CNN), INT8 quantization
├── pipeline.py               # Streaming pipeline: normalization -> window -> model -> EMA -> hysteresis -> gate
├── robot_sim.py              # 2D differential-drive simulator, 4m x 4m arena, obstacles, ultrasonic sensor
├── collect_real.py           # Optional webcam bridge for live recording with MediaPipe Hands
├── experiments/
│   ├── e1_recognition.py     # Leave-One-Subject-Out recognition benchmark (G1, G2, G4)
│   ├── e2_ablation.py        # Continuous tracking ablation: raw vs EMA vs hysteresis (G3, G5)
│   ├── e3_robustness.py      # Noise, dropout, scale, rotation sweeps & invariance check (G1, G5)
│   ├── e4_safety.py          # Sudden obstacle and hand-lost watchdog evaluation (G5)
│   ├── e5_navigation.py      # 100 navigation trials per pipeline across 3 operator styles (G3, G5)
│   └── e6_latency.py         # Full real-time latency budget and 20 FPS threshold analysis
├── results/                  # Generated CSV metrics, JSON summaries, and publication-quality figures
├── report_tables.md          # Auto-generated publication-ready report tables and analysis
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## ⚙️ Installation & Requirements

### Prerequisites
- Python 3.11 or 3.12 (TensorFlow 2.15+ compatible)
- Windows PowerShell / Linux Bash

### Setup Virtual Environment
```powershell
# Clone or navigate to the directory
cd lcgc

# Create virtual environment
py -3.12 -m venv .venv

# Activate environment
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

All experiments run autonomously on synthetic data with fixed random seeds (`SEED = 42`):

```powershell
# 1. Leave-One-Subject-Out Recognition Benchmark (E1)
python experiments/e1_recognition.py

# 2. Continuous Tracking Ablation (E2)
python experiments/e2_ablation.py

# 3. Robustness Sweeps & Invariance Verification (E3)
python experiments/e3_robustness.py

# 4. Safety Gate & Watchdog Verification (E4)
python experiments/e4_safety.py

# 5. End-to-End Navigation Simulation (E5)
python experiments/e5_navigation.py

# 6. Real-Time Latency Budget Analysis (E6)
python experiments/e6_latency.py
```

---

## 📷 Optional Real-Data Bridge

To record live gesture windows from an ordinary webcam (requires MediaPipe):

```powershell
# Record 50 windows of 'Forward' gesture
python collect_real.py --gesture Forward --samples 50 --out real_data/forward.npz

# Inspect recorded data
python collect_real.py --inspect real_data/forward.npz
```

---

## ⚠️ Honesty & Simulation Disclosure

All experiments, benchmarks, trajectories, and latency numbers reported by this software suite were executed in software simulation on synthetic 21-point hand landmarks using the exact parameters defined in `config.py`. They do not claim physical robot deployment or embedded hardware board timings.
