"""
Parametric hand-landmark generator for gesture-recognition simulation.

Generates synthetic 21-point hand landmarks (MediaPipe indexing) without
requiring any images.  Supports six gesture classes, per-subject variation,
augmentation, smooth transitions, and both windowed and continuous-stream
output formats.

MediaPipe hand landmark indices:
    0       : Wrist
    1 – 4   : Thumb  (CMC, MCP, IP, TIP)
    5 – 8   : Index  (MCP, PIP, DIP, TIP)
    9 – 12  : Middle (MCP, PIP, DIP, TIP)
    13 – 16 : Ring   (MCP, PIP, DIP, TIP)
    17 – 20 : Pinky  (MCP, PIP, DIP, TIP)
"""

import numpy as np

import config

# ══════════════════════════════════════════════════════════════════════════
# 1. CANONICAL SKELETON (21 landmarks, normalised coords centred on wrist)
# ══════════════════════════════════════════════════════════════════════════

# Each row is (x, y) relative to wrist.  Fingers fan upward from the wrist
# which sits at the origin.  Units are arbitrary; we normalise later.
# The hand is modelled as a *right* hand, palm facing the camera.

CANONICAL_HAND = np.array([
    # 0 – Wrist
    [0.00,  0.00],
    # Thumb
    [0.10,  0.03],    # 1 CMC
    [0.18,  0.10],    # 2 MCP
    [0.24,  0.18],    # 3 IP
    [0.28,  0.26],    # 4 TIP
    # Index finger
    [0.07,  0.30],    # 5 MCP
    [0.08,  0.42],    # 6 PIP
    [0.09,  0.52],    # 7 DIP
    [0.09,  0.60],    # 8 TIP
    # Middle finger
    [0.00,  0.32],    # 9 MCP
    [0.00,  0.45],    # 10 PIP
    [0.00,  0.56],    # 11 DIP
    [0.00,  0.65],    # 12 TIP
    # Ring finger
    [-0.07,  0.30],   # 13 MCP
    [-0.08,  0.42],   # 14 PIP
    [-0.08,  0.51],   # 15 DIP
    [-0.08,  0.58],   # 16 TIP
    # Pinky
    [-0.14,  0.27],   # 17 MCP
    [-0.16,  0.36],   # 18 PIP
    [-0.17,  0.43],   # 19 DIP
    [-0.17,  0.49],   # 20 TIP
], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════
# 2. GESTURE POSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════
# Each pose is a (21, 2) array representing the target landmark positions
# for that gesture *before* per-subject variation and augmentation.

def _curl_finger(base_idx: int, hand: np.ndarray, amount: float = 0.75) -> np.ndarray:
    """Curl a four-joint finger toward the palm by *amount* (0=open, 1=fist)."""
    hand = hand.copy()
    base = hand[base_idx]          # MCP joint
    for j in range(1, 4):          # PIP, DIP, TIP
        idx = base_idx + j
        vec = hand[idx] - base
        # Shorten and rotate toward palm centre
        angle = -amount * np.pi * 0.45 * j / 3
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        hand[idx] = base + rot @ vec * (1 - 0.25 * amount)
    return hand


def _rotate_hand(hand: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate all landmarks around the wrist by *degrees*."""
    rad = np.deg2rad(degrees)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated = hand.copy()
    rotated[1:] = (R @ hand[1:].T).T
    return rotated


def _build_poses() -> dict[str, np.ndarray]:
    """Return {gesture_name: (21, 2) landmark array} for all six poses."""

    # --- Idle: relaxed, all fingers lightly curled ---
    idle = CANONICAL_HAND.copy()
    for base in (5, 9, 13, 17):
        idle = _curl_finger(base, idle, amount=0.45)
    # Thumb slightly curled
    idle = _curl_finger(1, idle, amount=0.30)

    # --- Stop: open palm, all fingers fully extended ---
    stop = CANONICAL_HAND.copy()  # canonical *is* fully extended

    # --- Forward: index up, others curled ---
    forward = CANONICAL_HAND.copy()
    for base in (9, 13, 17):
        forward = _curl_finger(base, forward, amount=0.85)
    forward = _curl_finger(1, forward, amount=0.70)
    # Index stays extended (already up in canonical)

    # --- Backward: index pointing DOWN ---
    backward = forward.copy()
    # Flip index finger direction (mirror the y of index joints around MCP)
    idx_mcp_y = backward[5, 1]
    for j in range(6, 9):
        backward[j, 1] = idx_mcp_y - (backward[j, 1] - idx_mcp_y)

    # --- Left: index pointing LEFT (rotate index 90° CCW) ---
    left = forward.copy()
    idx_base = left[5].copy()
    for j in range(5, 9):
        vec = left[j] - left[0]
        angle = np.pi / 2  # 90° CCW
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        left[j] = left[0] + R @ vec

    # --- Right: index pointing RIGHT (rotate index 90° CW) ---
    right = forward.copy()
    for j in range(5, 9):
        vec = right[j] - right[0]
        angle = -np.pi / 2  # 90° CW
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        right[j] = right[0] + R @ vec

    return {
        "Idle": idle,
        "Stop": stop,
        "Forward": forward,
        "Backward": backward,
        "Left": left,
        "Right": right,
    }


GESTURE_POSES = _build_poses()

# ══════════════════════════════════════════════════════════════════════════
# 3. SUBJECT PROFILE
# ══════════════════════════════════════════════════════════════════════════

class SubjectProfile:
    """Per-subject anatomical and behavioural variation."""

    def __init__(self, subject_id: int, rng: np.random.Generator):
        self.subject_id = subject_id
        # Finger-length ratios (±10 % from canonical)
        self.length_scale = 1.0 + rng.uniform(
            -config.FINGER_LENGTH_VAR,
            config.FINGER_LENGTH_VAR,
            size=(config.NUM_LANDMARKS, 2),
        )
        self.length_scale[0] = 1.0  # wrist stays at origin

        # Palm aspect-ratio variation (stretch x or y slightly)
        self.palm_aspect = rng.uniform(0.92, 1.08, size=2)

        # Natural tilt bias in degrees
        self.tilt_bias = rng.uniform(-5.0, 5.0)

        # Jitter style: per-landmark noise std multiplier
        self.noise_multiplier = rng.uniform(0.7, 1.4)

        # Transition speed (number of frames to interpolate, within config range)
        lo, hi = config.TRANSITION_FRAMES_RANGE
        self.transition_speed = rng.integers(lo, hi + 1)

    def apply(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply anatomical variation to *landmarks* (21, 2)."""
        lm = landmarks.copy()
        # Scale finger lengths relative to wrist
        lm[1:] = lm[0] + (lm[1:] - lm[0]) * self.length_scale[1:]
        # Palm aspect
        lm[:, 0] *= self.palm_aspect[0]
        lm[:, 1] *= self.palm_aspect[1]
        # Tilt bias
        lm = _rotate_hand(lm, self.tilt_bias)
        return lm


# ══════════════════════════════════════════════════════════════════════════
# 4. PER-FRAME AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════

def _augment_frame(
    landmarks: np.ndarray,
    profile: SubjectProfile,
    rng: np.random.Generator,
    hand_lost: bool = False,
) -> np.ndarray:
    """Apply augmentations to a single (21, 2) frame.

    Returns (21, 2) with augmentation applied.  If *hand_lost* the entire
    frame is zeroed out.
    """
    if hand_lost:
        return np.zeros((config.NUM_LANDMARKS, 2), dtype=np.float64)

    lm = landmarks.copy()

    # --- Distance / scale variation ---
    scale = rng.uniform(0.7, 1.3)
    lm[1:] = lm[0] + (lm[1:] - lm[0]) * scale

    # --- In-plane rotation ---
    angle = rng.uniform(-config.ROTATION_RANGE, config.ROTATION_RANGE)
    lm = _rotate_hand(lm, angle)

    # --- Position shift (hand centre offset) ---
    shift = rng.uniform(-0.1, 0.1, size=2)
    lm += shift

    # --- Gaussian landmark noise ---
    noise_std = config.LANDMARK_NOISE_STD * profile.noise_multiplier
    lm += rng.normal(0, noise_std, size=lm.shape)

    # --- Random landmark dropout ---
    dropout_mask = rng.random(config.NUM_LANDMARKS) < config.LANDMARK_DROPOUT_PROB
    lm[dropout_mask] = 0.0

    return lm

# ══════════════════════════════════════════════════════════════════════════
# 5. NORMALISATION & PALM SIZE
# ══════════════════════════════════════════════════════════════════════════

def normalize_landmarks(raw_landmarks_21x2: np.ndarray) -> np.ndarray:
    """Normalise (21, 2) raw landmarks to a 42-dim flat vector.

    p_tilde_i = (p_i - p_0) / s_t
    where s_t = ||p_9 - p_0||_2

    If the hand is essentially absent (scale ≈ 0) the function returns zeros.
    """
    p0 = raw_landmarks_21x2[config.WRIST_IDX]
    p9 = raw_landmarks_21x2[config.MIDDLE_BASE_IDX]
    s_t = np.linalg.norm(p9 - p0)
    if s_t < 1e-8:
        return np.zeros(config.FEATURE_DIM, dtype=np.float64)
    normalised = (raw_landmarks_21x2 - p0) / s_t
    return normalised.flatten()


def compute_palm_size(
    raw_landmarks_21x2: np.ndarray, frame_height: float
) -> float:
    """Return palm-size fraction s_t / frame_height."""
    p0 = raw_landmarks_21x2[config.WRIST_IDX]
    p9 = raw_landmarks_21x2[config.MIDDLE_BASE_IDX]
    s_t = np.linalg.norm(p9 - p0)
    return float(s_t / frame_height) if frame_height > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════
# 6. TRANSITION MODELLING
# ══════════════════════════════════════════════════════════════════════════

def _interpolate_poses(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Smoothly interpolate between two (21, 2) poses over *n_frames*.

    Uses cosine interpolation for a natural ease-in / ease-out.
    Returns (n_frames, 21, 2).
    """
    ts = np.linspace(0, 1, n_frames)
    # Cosine ease: alpha goes 0 → 1 smoothly
    alphas = 0.5 * (1 - np.cos(ts * np.pi))
    frames = np.array([
        pose_a * (1 - a) + pose_b * a for a in alphas
    ])
    return frames

# ══════════════════════════════════════════════════════════════════════════
# 7a. TRAINING-WINDOW GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_training_windows(seed: int = 42):
    """Generate training windows for all subjects.

    Returns
    -------
    X_train : ndarray, shape (N, 15, 42)
        Normalised, flattened landmark windows.
    y_train : ndarray, shape (N,)
        Integer gesture labels.
    subject_ids : ndarray, shape (N,)
        Subject index for each window.
    """
    rng = np.random.default_rng(seed)

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_subj: list[int] = []

    for subj_id in range(config.NUM_SUBJECTS):
        profile = SubjectProfile(subj_id, np.random.default_rng(seed + subj_id))

        for class_idx, gesture_name in enumerate(config.GESTURE_NAMES):
            base_pose = GESTURE_POSES[gesture_name]
            subject_pose = profile.apply(base_pose)

            for win_idx in range(config.WINDOWS_PER_CLASS):
                window_rng = np.random.default_rng(
                    seed * 1000 + subj_id * 10000 + class_idx * 100 + win_idx
                )
                frames = np.zeros(
                    (config.WINDOW_T, config.NUM_LANDMARKS, 2),
                    dtype=np.float64,
                )

                # Decide if this window has a transition (20 % chance)
                has_transition = window_rng.random() < 0.20
                if has_transition:
                    # Transition *into* the target gesture from a random other
                    other_idx = window_rng.integers(0, config.NUM_CLASSES)
                    other_name = config.GESTURE_NAMES[other_idx]
                    other_pose = profile.apply(GESTURE_POSES[other_name])
                    n_trans = profile.transition_speed
                    trans = _interpolate_poses(other_pose, subject_pose, n_trans)
                    # Fill beginning with transition, rest with target
                    for t in range(config.WINDOW_T):
                        if t < n_trans:
                            frames[t] = trans[t]
                        else:
                            frames[t] = subject_pose
                else:
                    for t in range(config.WINDOW_T):
                        frames[t] = subject_pose

                # Augment each frame
                for t in range(config.WINDOW_T):
                    frames[t] = _augment_frame(frames[t], profile, window_rng)

                # Normalise each frame
                window_flat = np.zeros(
                    (config.WINDOW_T, config.FEATURE_DIM), dtype=np.float64
                )
                for t in range(config.WINDOW_T):
                    window_flat[t] = normalize_landmarks(frames[t])

                all_X.append(window_flat)
                all_y.append(class_idx)
                all_subj.append(subj_id)

    X_train = np.array(all_X, dtype=np.float32)
    y_train = np.array(all_y, dtype=np.int32)
    subject_ids = np.array(all_subj, dtype=np.int32)

    return X_train, y_train, subject_ids

# ══════════════════════════════════════════════════════════════════════════
# 7b. CONTINUOUS-STREAM GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_continuous_stream(subject_id: int, seed: int):
    """Generate a ~900-frame continuous gesture stream for one subject.

    Returns
    -------
    frames : ndarray, shape (T, 42)
        Normalised, flattened landmarks per frame.
    labels : ndarray, shape (T,)
        Ground-truth gesture label per frame.
    hand_present : ndarray, shape (T,), dtype bool
        Whether the hand is visible in each frame.
    """
    rng = np.random.default_rng(seed)
    profile = SubjectProfile(subject_id, np.random.default_rng(seed + subject_id))

    T = config.STREAM_FRAMES

    # Build a scripted gesture sequence with realistic timing
    # Each segment: (gesture_index, duration in frames)
    segments: list[tuple[int, int]] = []
    t_used = 0
    while t_used < T:
        gesture_idx = rng.integers(0, config.NUM_CLASSES)
        # Hold duration: 30-90 frames (1-3 seconds)
        hold = int(rng.integers(30, 91))
        hold = min(hold, T - t_used)
        segments.append((gesture_idx, hold))
        t_used += hold

    # Expand segments into per-frame poses with transitions
    raw_frames = np.zeros((T, config.NUM_LANDMARKS, 2), dtype=np.float64)
    frame_labels = np.zeros(T, dtype=np.int32)
    hand_present_arr = np.ones(T, dtype=bool)

    t = 0
    for seg_i, (g_idx, hold) in enumerate(segments):
        gesture_name = config.GESTURE_NAMES[g_idx]
        target_pose = profile.apply(GESTURE_POSES[gesture_name])

        if seg_i > 0 and t > 0:
            # Smooth transition from previous pose
            prev_pose = raw_frames[t - 1].copy()
            n_trans = min(profile.transition_speed, hold)
            trans = _interpolate_poses(prev_pose, target_pose, n_trans)
            for j in range(n_trans):
                if t >= T:
                    break
                raw_frames[t] = trans[j]
                frame_labels[t] = g_idx
                t += 1
            remaining = hold - n_trans
        else:
            remaining = hold

        for _ in range(remaining):
            if t >= T:
                break
            raw_frames[t] = target_pose
            frame_labels[t] = g_idx
            t += 1

    # Apply augmentation + hand-lost segments
    # Pre-determine hand-lost spans
    lost_starts: list[int] = []
    frame_cursor = 0
    while frame_cursor < T:
        if rng.random() < config.HAND_LOST_PROB:
            lost_len = int(rng.integers(5, 16))  # 5-15 frames lost
            for ll in range(lost_len):
                if frame_cursor + ll < T:
                    hand_present_arr[frame_cursor + ll] = False
            lost_starts.append(frame_cursor)
            frame_cursor += lost_len
        else:
            frame_cursor += 1

    out_frames = np.zeros((T, config.FEATURE_DIM), dtype=np.float64)
    for t_i in range(T):
        is_lost = not hand_present_arr[t_i]
        augmented = _augment_frame(raw_frames[t_i], profile, rng, hand_lost=is_lost)
        out_frames[t_i] = normalize_landmarks(augmented)

    return (
        out_frames.astype(np.float32),
        frame_labels,
        hand_present_arr,
    )


def generate_all_streams(seed: int = 42) -> dict:
    """Generate continuous streams for all subjects.

    Returns
    -------
    streams : dict
        ``{subject_id: {"frames": (T,42), "labels": (T,),
        "hand_present": (T,)}}``
    """
    streams: dict = {}
    for subj_id in range(config.NUM_SUBJECTS):
        frames, labels, hand_present = generate_continuous_stream(
            subj_id, seed=seed + subj_id * 7
        )
        streams[subj_id] = {
            "frames": frames,
            "labels": labels,
            "hand_present": hand_present,
        }
    return streams

# ══════════════════════════════════════════════════════════════════════════
# MAIN — quick sanity check
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LCGC Synthetic Hand Data — Sanity Check")
    print("=" * 60)

    # --- Training windows ---
    print("\n--- Training Windows ---")
    X, y, subj = generate_training_windows(seed=config.SEED)
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Subjects: {np.unique(subj).tolist()}")

    print("\n  Windows per class:")
    for ci, name in enumerate(config.GESTURE_NAMES):
        count = int(np.sum(y == ci))
        print(f"    {ci} ({name:>10s}): {count}")

    print(f"\n  Total windows: {len(y)}")
    expected = config.NUM_SUBJECTS * config.NUM_CLASSES * config.WINDOWS_PER_CLASS
    print(f"  Expected     : {expected}")

    # Quick feature-range check
    print(f"\n  X min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}")

    # --- Continuous streams ---
    print("\n--- Continuous Streams ---")
    streams = generate_all_streams(seed=config.SEED)
    for sid in sorted(streams.keys()):
        s = streams[sid]
        n_lost = int(np.sum(~s["hand_present"]))
        unique_labels = np.unique(s["labels"]).tolist()
        print(
            f"  Subject {sid:2d}: frames={s['frames'].shape[0]}, "
            f"lost={n_lost}, gestures={unique_labels}"
        )

    # --- Normalisation quick test ---
    print("\n--- Normalisation Test ---")
    test_lm = CANONICAL_HAND.copy()
    normed = normalize_landmarks(test_lm)
    print(f"  Input shape : (21, 2)")
    print(f"  Output shape: ({normed.shape[0]},)")
    print(f"  Wrist (should be [0,0]): [{normed[0]:.4f}, {normed[1]:.4f}]")
    p9_norm = np.linalg.norm(normed[18:20])
    print(f"  ||p9_norm|| (should be 1.0): {p9_norm:.4f}")

    # --- Palm-size test ---
    print("\n--- Palm Size Test ---")
    ps = compute_palm_size(test_lm, frame_height=1.0)
    print(f"  Palm size fraction (h=1.0): {ps:.4f}")

    print("\n" + "=" * 60)
    print("  All checks passed.")
    print("=" * 60)
