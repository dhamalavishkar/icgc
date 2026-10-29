"""
LCGC Configuration — all fixed parameters for the simulation.
Every hyper-parameter, threshold, and physical constant lives here.
"""

import numpy as np

# ─── Random seeds ───────────────────────────────────────────────────────
SEED = 42

# ─── Camera / window ────────────────────────────────────────────────────
FPS = 30
WINDOW_T = 15               # frames in one window (~0.5 s at 30 FPS)
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 2   # 42 (x, y per landmark)

# ─── Landmark normalisation ─────────────────────────────────────────────
WRIST_IDX = 0                # p_0  (normalisation origin)
MIDDLE_BASE_IDX = 9          # p_9  (scale reference)
# s_t = ||p_9 - p_0||_2
# p_tilde_i = (p_i - p_0) / s_t

# MediaPipe hand landmark groups (0-indexed)
THUMB_IDXS    = [1, 2, 3, 4]
INDEX_IDXS    = [5, 6, 7, 8]
MIDDLE_IDXS   = [9, 10, 11, 12]
RING_IDXS     = [13, 14, 15, 16]
PINKY_IDXS    = [17, 18, 19, 20]

# ─── Gesture classes ────────────────────────────────────────────────────
GESTURE_NAMES = ["Idle", "Stop", "Forward", "Backward", "Left", "Right"]
NUM_CLASSES = len(GESTURE_NAMES)
GESTURE_TO_IDX = {name: i for i, name in enumerate(GESTURE_NAMES)}
IDX_TO_GESTURE = {i: name for i, name in enumerate(GESTURE_NAMES)}

# ─── Model architecture ─────────────────────────────────────────────────
# Conv1D(32,k=3,same,no_bias) + BN + ReLU
# SeparableConv1D(48,k=3) + BN + ReLU
# SeparableConv1D(64,k=3) + BN + ReLU
# GlobalAveragePooling1D
# Dense(32, relu) + Dropout(0.2) + Dense(6, softmax)
CONV1_FILTERS = 32
CONV1_KERNEL = 3
SEP_CONV2_FILTERS = 48
SEP_CONV2_KERNEL = 3
SEP_CONV3_FILTERS = 64
SEP_CONV3_KERNEL = 3
DENSE_HIDDEN = 32
DROPOUT_RATE = 0.2

# ─── Training ───────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 8
NUM_REPRESENTATIVE_WINDOWS = 200   # for INT8 quantisation

# ─── Smoothing / hysteresis ─────────────────────────────────────────────
EMA_ALPHA = 0.4              # probability EMA
HYSTERESIS_THRESHOLD = 0.80  # minimum probability to accept
HYSTERESIS_M_DEFAULT = 3     # consecutive frames required
HYSTERESIS_M_STOP = 1        # Stop is never delayed

# ─── Command mapping ────────────────────────────────────────────────────
V_MAX = 0.25                 # m/s
OMEGA_MAX = 1.0              # rad/s
PALM_GAMMA = 0.2             # minimum speed fraction
S_MIN = 0.08                 # palm-size fraction (min)
S_MAX = 0.30                 # palm-size fraction (max)

# ─── Differential-drive robot ───────────────────────────────────────────
WHEEL_SEPARATION_L = 0.20    # m

# ─── Safety gate ────────────────────────────────────────────────────────
ULTRASONIC_STOP_DIST = 0.20  # m  (20 cm)
HAND_LOST_TIMEOUT = 0.5      # seconds
ULTRASONIC_NOISE_STD = 0.02  # m  (sensor noise)
ULTRASONIC_MAX_RANGE = 2.0   # m

# ─── Data generation ────────────────────────────────────────────────────
NUM_SUBJECTS = 12
WINDOWS_PER_CLASS = 120      # per subject per class
SUBJECT_SPLIT = (0.70, 0.15, 0.15)  # train / val / test

# Subject variation
FINGER_LENGTH_VAR = 0.10     # ±10%
ROTATION_RANGE = 15.0        # degrees
LANDMARK_NOISE_STD = 0.015   # normalised units
LANDMARK_DROPOUT_PROB = 0.03 # probability per landmark per frame
TRANSITION_FRAMES_RANGE = (3, 8)
HAND_LOST_PROB = 0.02        # probability of hand-lost per frame in streams

# Continuous stream length per subject
STREAM_FRAMES = 900          # 30 seconds at 30 FPS

# ─── Robot simulation arena ─────────────────────────────────────────────
ARENA_WIDTH = 4.0            # m
ARENA_HEIGHT = 4.0           # m
ROBOT_RADIUS = 0.10          # m (circular approximation)
GOAL_RADIUS = 0.25           # m (reach tolerance)
SIM_DT = 1.0 / FPS           # simulation time step

# ─── Navigation experiment ──────────────────────────────────────────────
NAV_TRIALS = 100
NAV_MAX_STEPS = 1200         # 40 seconds at 30 FPS (allows realistic trajectory traversal)

# ─── Latency budget (E6) ────────────────────────────────────────────────
FRAME_INTERVAL_MS = 1000.0 / FPS  # 33.3 ms
SERIAL_TRANSFER_MS = 0.7
LANDMARK_BEST_MS = 17.1
LANDMARK_SWEEP_MS = np.arange(10, 81, 5).tolist()  # 10..80 ms
FPS_TARGET = 20              # target: 50 ms per frame
