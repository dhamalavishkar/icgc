"""
LCGC Streaming Gesture-Recognition Pipeline
=============================================
Processes raw hand landmarks frame-by-frame and outputs (v, omega)
commands for a differential-drive robot.

Pipeline modes
--------------
* ``raw``            – argmax of model softmax, no smoothing
* ``ema``            – EMA probability smoothing, then argmax
* ``ema_hysteresis`` – EMA + consecutive-frame hysteresis
* ``full``           – EMA + hysteresis + ultrasonic / hand-lost safety gate
"""

import time
import numpy as np

import config

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_frame(raw_landmarks_21x2: np.ndarray) -> np.ndarray:
    """Wrist-relative, palm-scale normalisation.

    Parameters
    ----------
    raw_landmarks_21x2 : ndarray of shape (21, 2)
        Raw (x, y) pixel / normalised coordinates for 21 hand landmarks.

    Returns
    -------
    ndarray of shape (42,)
        Flattened normalised landmark vector.

    Notes
    -----
    p_tilde_i = (p_i - p_0) / s_t   where  s_t = ||p_9 - p_0||_2
    """
    p0 = raw_landmarks_21x2[config.WRIST_IDX]          # wrist
    p9 = raw_landmarks_21x2[config.MIDDLE_BASE_IDX]    # middle-finger base
    s_t = np.linalg.norm(p9 - p0)
    if s_t < 1e-8:
        s_t = 1e-8  # avoid division by zero
    normalised = (raw_landmarks_21x2 - p0) / s_t
    return normalised.flatten().astype(np.float32)


def gesture_to_command(gesture_idx: int,
                       palm_size_fraction: float) -> tuple[float, float]:
    """Map a gesture class index + palm proximity to (v, omega).

    Speed scaling (Forward / Backward)::

        v_scaled = V_MAX * clip(gamma, (palm - s_min) / (s_max - s_min), 1)

    Parameters
    ----------
    gesture_idx : int
        Predicted gesture (0-5).
    palm_size_fraction : float
        Apparent palm bounding-box width divided by frame width.

    Returns
    -------
    (v, omega) : tuple of float
        Linear and angular velocity command.
    """
    if gesture_idx in (0, 1):
        # Idle / Stop → halt
        return 0.0, 0.0

    # Speed proportional to palm proximity
    frac = (palm_size_fraction - config.S_MIN) / (config.S_MAX - config.S_MIN)
    frac = min(1.0, max(config.PALM_GAMMA, frac))
    v_scaled = config.V_MAX * frac

    if gesture_idx == 2:    # Forward
        return +v_scaled, 0.0
    if gesture_idx == 3:    # Backward
        return -v_scaled, 0.0
    if gesture_idx == 4:    # Left
        return 0.0, +config.OMEGA_MAX
    if gesture_idx == 5:    # Right
        return 0.0, -config.OMEGA_MAX

    return 0.0, 0.0         # fallback


def diff_drive_wheels(v: float, omega: float) -> tuple[float, float]:
    """Convert (v, omega) to individual wheel velocities.

    Returns
    -------
    (v_right, v_left) : tuple of float
        Right and left wheel linear velocities (m/s).
    """
    v_r = v + omega * config.WHEEL_SEPARATION_L / 2
    v_l = v - omega * config.WHEEL_SEPARATION_L / 2
    return v_r, v_l


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class GesturePipeline:
    """Frame-by-frame streaming gesture recognition pipeline.

    Parameters
    ----------
    model : keras.Model or None
        A trained Keras model.  Ignored when *use_tflite* is ``True``.
    mode : {'raw', 'ema', 'ema_hysteresis', 'full'}
        Post-processing chain applied to the raw model probabilities.
    use_tflite : bool
        If ``True``, run inference through a TFLite interpreter instead of
        the Keras model.
    tflite_path : str or None
        Path to the ``.tflite`` file.  Required when *use_tflite* is True.
    """

    _VALID_MODES = ('raw', 'ema', 'ema_hysteresis', 'full')

    def __init__(self, model, mode='full', use_tflite=False, tflite_path=None):
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got {mode!r}")

        self.model = model
        self.mode = mode
        self.use_tflite = use_tflite
        self._is_mlp = False  # set True externally for B1 MLP baseline

        # ── TFLite interpreter ──────────────────────────────────────────
        self.interpreter = None
        self._tfl_input_details = None
        self._tfl_output_details = None
        if use_tflite:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self._tfl_input_details = self.interpreter.get_input_details()
            self._tfl_output_details = self.interpreter.get_output_details()

        # ── Sliding window buffer ───────────────────────────────────────
        self._window = np.zeros(
            (config.WINDOW_T, config.FEATURE_DIM), dtype=np.float32
        )

        # ── EMA state ──────────────────────────────────────────────────
        self._smoothed_probs = np.ones(config.NUM_CLASSES, dtype=np.float32) \
                               / config.NUM_CLASSES

        # ── Hysteresis state ───────────────────────────────────────────
        self._consecutive_class = 0   # class that has been top recently
        self._consecutive_count = 0   # how many consecutive frames

        # ── Safety gate state ──────────────────────────────────────────
        self._hand_lost_elapsed = 0.0  # seconds since hand last seen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_frame(self, norm_vec: np.ndarray) -> None:
        """Append a normalised 42-d vector to the sliding window, dropping
        the oldest frame."""
        self._window = np.roll(self._window, shift=-1, axis=0)
        self._window[-1] = norm_vec

    def _classify(self) -> np.ndarray:
        """Run the model on the current window and return raw probabilities
        of shape (NUM_CLASSES,)."""
        if self._is_mlp:
            # Baseline MLP: single frame only
            x = self._window[-1:][np.newaxis, ...]    # (1, 1, 42) → squeeze to (1, 42)
            x = x.reshape(1, config.FEATURE_DIM)
        else:
            x = self._window[np.newaxis, ...]          # (1, WINDOW_T, FEATURE_DIM)

        if self.use_tflite:
            in_dtype = self._tfl_input_details[0]['dtype']
            if in_dtype == np.int8:
                scale, zero_point = self._tfl_input_details[0]['quantization']
                x_quant = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
                self.interpreter.set_tensor(self._tfl_input_details[0]['index'], x_quant)
            else:
                self.interpreter.set_tensor(
                    self._tfl_input_details[0]['index'],
                    x.astype(np.float32),
                )
            self.interpreter.invoke()
            out_tensor = self.interpreter.get_tensor(
                self._tfl_output_details[0]['index']
            )
            out_dtype = self._tfl_output_details[0]['dtype']
            if out_dtype == np.int8:
                scale_o, zero_point_o = self._tfl_output_details[0]['quantization']
                probs = (out_tensor.astype(np.float32) - zero_point_o) * scale_o
            else:
                probs = out_tensor
            return probs[0].astype(np.float32)
        else:
            probs = self.model(x, training=False)
            return np.asarray(probs[0], dtype=np.float32)

    def _apply_ema(self, raw_probs: np.ndarray) -> np.ndarray:
        """Exponential moving average over probability vectors."""
        alpha = config.EMA_ALPHA
        self._smoothed_probs = (alpha * raw_probs
                                + (1 - alpha) * self._smoothed_probs)
        return self._smoothed_probs.copy()

    def _apply_hysteresis(self, probs: np.ndarray) -> int:
        """Consecutive-frame hysteresis filter.

        The predicted class must be the argmax for *m* consecutive frames
        **and** exceed ``HYSTERESIS_THRESHOLD`` before it is accepted.

        Returns
        -------
        gesture_idx : int
            Accepted gesture class index.  Falls back to Stop (1) when
            conditions are not met.
        """
        top_class = int(np.argmax(probs))
        top_prob = float(probs[top_class])

        # Update consecutive counter
        if top_class == self._consecutive_class:
            self._consecutive_count += 1
        else:
            self._consecutive_class = top_class
            self._consecutive_count = 1

        # Determine required number of consecutive frames
        if top_class == config.GESTURE_TO_IDX["Stop"]:
            m_required = config.HYSTERESIS_M_STOP
        else:
            m_required = config.HYSTERESIS_M_DEFAULT

        # Accept only if probability high enough AND enough consecutive frames
        if top_prob >= config.HYSTERESIS_THRESHOLD \
                and self._consecutive_count >= m_required:
            return top_class

        # Default safe output — Stop
        return config.GESTURE_TO_IDX["Stop"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, raw_landmarks_21x2_or_none, palm_size_frac=0.15,
             ultrasonic_dist=float('inf'), dt=1.0 / 30):
        """Process one camera frame.

        Parameters
        ----------
        raw_landmarks_21x2_or_none : ndarray (21, 2) or None
            Raw landmark positions.  ``None`` when no hand is detected.
        palm_size_frac : float
            Palm bounding-box width as a fraction of the frame width.
        ultrasonic_dist : float
            Current ultrasonic range reading (metres).
        dt : float
            Time elapsed since the previous frame (seconds).

        Returns
        -------
        (v, omega) : tuple of float
            Velocity command for the robot.
        gesture_name : str
            Human-readable gesture label.
        raw_probs : ndarray (NUM_CLASSES,)
            Raw model softmax output.
        smoothed_probs : ndarray (NUM_CLASSES,)
            Post-EMA probabilities (same as raw_probs if EMA is off).
        """
        # ── Handle hand-lost frames ─────────────────────────────────────
        if raw_landmarks_21x2_or_none is None:
            self._hand_lost_elapsed += dt
            # Push a zero frame into the window so the buffer keeps sliding
            self._push_frame(np.zeros(config.FEATURE_DIM, dtype=np.float32))
            raw_probs = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            raw_probs[config.GESTURE_TO_IDX["Idle"]] = 1.0
            smoothed_probs = self._apply_ema(raw_probs) \
                if self.mode in ('ema', 'ema_hysteresis', 'full') else raw_probs.copy()
            return (0.0, 0.0), "Idle", raw_probs, smoothed_probs

        # Hand detected — reset hand-lost timer
        self._hand_lost_elapsed = 0.0

        # ── Normalise & buffer ─────────────────────────────────────────
        raw_lm = np.asarray(raw_landmarks_21x2_or_none, dtype=np.float32)
        norm_vec = normalize_frame(raw_lm)
        self._push_frame(norm_vec)

        # ── Classification ─────────────────────────────────────────────
        raw_probs = self._classify()

        # ── Post-processing chain ──────────────────────────────────────
        if self.mode == 'raw':
            gesture_idx = int(np.argmax(raw_probs))
            smoothed_probs = raw_probs.copy()

        elif self.mode == 'ema':
            smoothed_probs = self._apply_ema(raw_probs)
            gesture_idx = int(np.argmax(smoothed_probs))

        elif self.mode == 'ema_hysteresis':
            smoothed_probs = self._apply_ema(raw_probs)
            gesture_idx = self._apply_hysteresis(smoothed_probs)

        elif self.mode == 'full':
            smoothed_probs = self._apply_ema(raw_probs)
            gesture_idx = self._apply_hysteresis(smoothed_probs)

        else:
            raise RuntimeError(f"Unknown mode {self.mode!r}")

        # ── Command mapping ────────────────────────────────────────────
        v, omega = gesture_to_command(gesture_idx, palm_size_frac)

        # ── Safety gate (full mode only) ───────────────────────────────
        if self.mode == 'full':
            if ultrasonic_dist < config.ULTRASONIC_STOP_DIST:
                v, omega = 0.0, 0.0
            if self._hand_lost_elapsed > config.HAND_LOST_TIMEOUT:
                v, omega = 0.0, 0.0

        gesture_name = config.IDX_TO_GESTURE[gesture_idx]
        return (v, omega), gesture_name, raw_probs, smoothed_probs

    def reset(self):
        """Clear all internal state (window, EMA, hysteresis, timers)."""
        self._window[:] = 0.0
        self._smoothed_probs[:] = 1.0 / config.NUM_CLASSES
        self._consecutive_class = 0
        self._consecutive_count = 0
        self._hand_lost_elapsed = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Quick-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(config.SEED)

    # ── Tiny dummy model (Keras) ─────────────────────────────────────────
    import tensorflow as tf
    tf.random.set_seed(config.SEED)

    dummy_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(config.WINDOW_T,
                                                config.FEATURE_DIM)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax'),
    ])

    # ── Run pipeline in each mode ────────────────────────────────────────
    for mode in ('raw', 'ema', 'ema_hysteresis', 'full'):
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")
        pipe = GesturePipeline(dummy_model, mode=mode)

        for t in range(20):
            # Simulate occasional hand loss
            if np.random.rand() < 0.1:
                lm = None
            else:
                lm = np.random.randn(config.NUM_LANDMARKS, 2).astype(np.float32)

            (v, omega), name, raw_p, sm_p = pipe.step(
                lm,
                palm_size_frac=np.random.uniform(0.05, 0.35),
                ultrasonic_dist=np.random.uniform(0.1, 2.0),
                dt=config.SIM_DT,
            )
            v_r, v_l = diff_drive_wheels(v, omega)
            print(f"  t={t:3d}  gesture={name:<10s}  v={v:+.3f}  "
                  f"omega={omega:+.3f}  wheels=({v_r:+.3f}, {v_l:+.3f})  "
                  f"top_raw={raw_p.max():.3f}  top_sm={sm_p.max():.3f}")

    print("\n✓ Pipeline smoke-test complete.")
