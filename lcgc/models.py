"""
LCGC Models — three neural-network classifiers for gesture recognition.

B1: Single-frame MLP        (static gesture baseline, Muhtadin-style)
B2: LSTM                    (heavy temporal baseline, Mavsar/Bandi-style)
B3: Depthwise-Separable 1D-CNN  (our LCGC model)

Also provides utility helpers for parameter counting, MAC estimation,
model serialisation (FP32 & INT8), and inference-time measurement.
"""

import os
import time
import pathlib

import numpy as np
import tensorflow as tf

import config

# ─── Reproducibility ────────────────────────────────────────────────────
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)


# ═══════════════════════════════════════════════════════════════════════
#  Model builders
# ═══════════════════════════════════════════════════════════════════════

def build_b1_mlp() -> tf.keras.Model:
    """B1 — Single-frame MLP (static gesture baseline).

    Input : (42,) — one frame of 42 landmark features.
    Output: (6,)  — softmax over gesture classes.
    Architecture:
        Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.2)
        → Dense(6, softmax)
    """
    inputs = tf.keras.Input(shape=(config.FEATURE_DIM,), name="frame_input")
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Dense(config.DENSE_HIDDEN, activation="relu",
                              name="dense_2")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax",
                                    name="output")(x)
    model = tf.keras.Model(inputs, outputs, name="B1_MLP")
    return model


def build_b2_lstm() -> tf.keras.Model:
    """B2 — LSTM (heavy temporal baseline).

    Input : (15, 42) — window of WINDOW_T frames × FEATURE_DIM features.
    Output: (6,)     — softmax over gesture classes.
    Architecture:
        LSTM(64) → Dense(32, relu) → Dropout(0.2) → Dense(6, softmax)
    """
    inputs = tf.keras.Input(
        shape=(config.WINDOW_T, config.FEATURE_DIM), name="window_input"
    )
    x = tf.keras.layers.LSTM(64, name="lstm_1")(inputs)
    x = tf.keras.layers.Dense(config.DENSE_HIDDEN, activation="relu",
                              name="dense_1")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax",
                                    name="output")(x)
    model = tf.keras.Model(inputs, outputs, name="B2_LSTM")
    return model


def build_b3_lcgc() -> tf.keras.Model:
    """B3 — LCGC Depthwise-Separable 1D-CNN (our model).

    Input : (15, 42)
    Output: (6,)

    Architecture:
        Conv1D(32, k=3, same, no bias) + BatchNorm + ReLU
        SeparableConv1D(48, k=3, same) + BatchNorm + ReLU
        SeparableConv1D(64, k=3, same) + BatchNorm + ReLU
        GlobalAveragePooling1D
        Dense(32, relu) + Dropout(0.2) + Dense(6, softmax)

    Expected: ~11,446 trainable params, ~135,440 MACs.
    """
    inputs = tf.keras.Input(
        shape=(config.WINDOW_T, config.FEATURE_DIM), name="window_input"
    )

    # --- Block 1: standard Conv1D (no bias) + BN + ReLU ------------------
    x = tf.keras.layers.Conv1D(
        filters=config.CONV1_FILTERS,
        kernel_size=config.CONV1_KERNEL,
        padding="same",
        use_bias=False,
        name="conv1d_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.ReLU(name="relu_1")(x)

    # --- Block 2: SeparableConv1D + BN + ReLU ----------------------------
    x = tf.keras.layers.SeparableConv1D(
        filters=config.SEP_CONV2_FILTERS,
        kernel_size=config.SEP_CONV2_KERNEL,
        padding="same",
        name="sep_conv1d_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.ReLU(name="relu_2")(x)

    # --- Block 3: SeparableConv1D + BN + ReLU ----------------------------
    x = tf.keras.layers.SeparableConv1D(
        filters=config.SEP_CONV3_FILTERS,
        kernel_size=config.SEP_CONV3_KERNEL,
        padding="same",
        name="sep_conv1d_3",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
    x = tf.keras.layers.ReLU(name="relu_3")(x)

    # --- Pooling + classifier head ----------------------------------------
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(config.DENSE_HIDDEN, activation="relu",
                              name="dense_head")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax",
                                    name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="B3_LCGC")
    return model


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def count_params(model: tf.keras.Model) -> int:
    """Return the number of *trainable* parameters in *model*."""
    return int(np.sum([
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    ]))


def estimate_macs(model: tf.keras.Model) -> int:
    """Estimate multiply-accumulate operations (MACs) for one inference.

    Formulae
    --------
    Dense             : input_dim × output_dim
    Conv1D            : output_length × filters × kernel_size × input_channels
    SeparableConv1D   : output_length × (in_ch × kernel + in_ch × filters)
    LSTM              : 4 × (input_size + hidden_size) × hidden_size × seq_len
    """
    total_macs = 0

    for layer in model.layers:
        layer_cfg = layer.get_config()

        # ── Dense ────────────────────────────────────────────────────
        if isinstance(layer, tf.keras.layers.Dense):
            # input shape: (..., input_dim)
            input_dim = layer.input.shape[-1]
            output_dim = layer.output.shape[-1]
            total_macs += int(input_dim) * int(output_dim)

        # ── Conv1D ───────────────────────────────────────────────────
        elif isinstance(layer, tf.keras.layers.Conv1D) and not isinstance(
            layer, tf.keras.layers.SeparableConv1D
        ):
            output_length = int(layer.output.shape[1])
            filters = int(layer_cfg["filters"])
            kernel_size = int(layer_cfg["kernel_size"][0]
                              if isinstance(layer_cfg["kernel_size"], (list, tuple))
                              else layer_cfg["kernel_size"])
            input_channels = int(layer.input.shape[-1])
            total_macs += output_length * filters * kernel_size * input_channels

        # ── SeparableConv1D ──────────────────────────────────────────
        elif isinstance(layer, tf.keras.layers.SeparableConv1D):
            output_length = int(layer.output.shape[1])
            filters = int(layer_cfg["filters"])
            kernel_size = int(layer_cfg["kernel_size"][0]
                              if isinstance(layer_cfg["kernel_size"], (list, tuple))
                              else layer_cfg["kernel_size"])
            input_channels = int(layer.input.shape[-1])
            # Depthwise + Pointwise
            total_macs += output_length * (
                input_channels * kernel_size + input_channels * filters
            )

        # ── LSTM ─────────────────────────────────────────────────────
        elif isinstance(layer, tf.keras.layers.LSTM):
            input_size = int(layer.input.shape[-1])
            hidden_size = int(layer_cfg["units"])
            seq_len = int(layer.input.shape[1])
            total_macs += 4 * (input_size + hidden_size) * hidden_size * seq_len

    return total_macs


# ─── Serialisation ──────────────────────────────────────────────────────

def save_fp32_model(model: tf.keras.Model, path: str) -> int:
    """Save *model* as a FP32 TFLite file and return file size in bytes."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
    except Exception:
        # Fallback for models with recurrent ops (like LSTM)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

    with open(path, "wb") as f:
        f.write(tflite_model)

    return os.path.getsize(path)


def quantize_int8(
    model: tf.keras.Model,
    representative_data: np.ndarray,
    path: str,
) -> int:
    """Quantise *model* to INT8 via full-integer quantisation."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _representative_dataset():
        for i in range(len(representative_data)):
            yield [representative_data[i : i + 1].astype(np.float32)]

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
    except Exception:
        # If full-integer INT8 fails (common with recurrent/LSTM layers), try dynamic range / default INT8
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = _representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
        except Exception:
            return -1

    with open(path, "wb") as f:
        f.write(tflite_model)

    return os.path.getsize(path)


# ─── Inference timing ──────────────────────────────────────────────────

def measure_inference_time(
    model: tf.keras.Model,
    sample_input: np.ndarray,
    n_runs: int = 100,
) -> float:
    """Measure mean Keras inference time in milliseconds.

    Warms up with 10 forward passes, then times *n_runs* passes.

    Parameters
    ----------
    model : tf.keras.Model
    sample_input : np.ndarray
        A single sample (no batch dim needed; it is added automatically).
    n_runs : int
        Number of timed inference passes.

    Returns
    -------
    float
        Mean inference time in milliseconds.
    """
    # Ensure batch dimension
    if sample_input.ndim == len(model.input_shape) - 1:
        x = np.expand_dims(sample_input, axis=0).astype(np.float32)
    else:
        x = sample_input.astype(np.float32)

    # Warm-up
    for _ in range(10):
        model(x, training=False)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x, training=False)
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0


def measure_tflite_inference_time(
    tflite_path: str,
    sample_input: np.ndarray,
    n_runs: int = 100,
) -> float:
    """Measure mean TFLite interpreter inference time in milliseconds.

    Warms up with 10 invocations, then times *n_runs* invocations.

    Parameters
    ----------
    tflite_path : str
        Path to a ``.tflite`` model file.
    sample_input : np.ndarray
        A single sample matching the model's expected input shape/type.
    n_runs : int
        Number of timed inference passes.

    Returns
    -------
    float
        Mean inference time in milliseconds.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Prepare input — match expected dtype and add batch dim if needed
    expected_shape = input_details["shape"]
    x = sample_input.copy()
    if x.ndim == len(expected_shape) - 1:
        x = np.expand_dims(x, axis=0)
    x = x.astype(input_details["dtype"])

    # Handle INT8 quantisation scaling
    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        x = (x.astype(np.float32) / scale + zero_point).astype(np.int8)

    # Warm-up
    for _ in range(10):
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0


# ═══════════════════════════════════════════════════════════════════════
#  Main — quick sanity check
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  LCGC Model Summary & Diagnostics")
    print("=" * 70)

    # ── Build all three models ───────────────────────────────────────
    b1 = build_b1_mlp()
    b2 = build_b2_lstm()
    b3 = build_b3_lcgc()

    models = {"B1_MLP": b1, "B2_LSTM": b2, "B3_LCGC": b3}

    # -- B3 detailed summary ------------------------------------------
    print("\n-- B3 (LCGC) model.summary() --")
    b3.summary()

    # -- Parameters & MACs --------------------------------------------
    print("\n-- Trainable parameters & estimated MACs --")
    for tag, m in models.items():
        params = count_params(m)
        macs = estimate_macs(m)
        print(f"  {tag:12s}  params={params:>8,}   MACs={macs:>12,}")

    # -- Random sample data -------------------------------------------
    rng = np.random.RandomState(config.SEED)
    sample_frame = rng.randn(config.FEATURE_DIM).astype(np.float32)
    sample_window = rng.randn(
        config.WINDOW_T, config.FEATURE_DIM
    ).astype(np.float32)

    # -- Keras inference time -----------------------------------------
    print("\n-- Keras inference time (100 runs) --")
    for tag, m, inp in [
        ("B1_MLP", b1, sample_frame),
        ("B2_LSTM", b2, sample_window),
        ("B3_LCGC", b3, sample_window),
    ]:
        t_ms = measure_inference_time(m, inp)
        print(f"  {tag:12s}  {t_ms:.3f} ms")

    # -- FP32 TFLite sizes -------------------------------------------
    print("\n-- FP32 TFLite model sizes --")
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    for tag, m in models.items():
        fp32_path = os.path.join(out_dir, f"{tag}_fp32.tflite")
        size_bytes = save_fp32_model(m, fp32_path)
        print(f"  {tag:12s}  {size_bytes:>8,} bytes  ({size_bytes / 1024:.1f} KB)")

    # -- INT8 quantised sizes -----------------------------------------
    print("\n-- INT8 TFLite model sizes --")
    rep_frames = rng.randn(
        config.NUM_REPRESENTATIVE_WINDOWS, config.FEATURE_DIM
    ).astype(np.float32)
    rep_windows = rng.randn(
        config.NUM_REPRESENTATIVE_WINDOWS, config.WINDOW_T, config.FEATURE_DIM
    ).astype(np.float32)

    for tag, m, rep in [
        ("B1_MLP", b1, rep_frames),
        ("B2_LSTM", b2, rep_windows),
        ("B3_LCGC", b3, rep_windows),
    ]:
        int8_path = os.path.join(out_dir, f"{tag}_int8.tflite")
        size_bytes = quantize_int8(m, rep, int8_path)
        if size_bytes > 0:
            print(f"  {tag:12s}  {size_bytes:>8,} bytes  ({size_bytes / 1024:.1f} KB)")
        else:
            print(f"  {tag:12s}  Unsupported (dynamic tensor ops)")

    # -- TFLite FP32 inference time -----------------------------------
    print("\n-- TFLite FP32 inference time (100 runs) --")
    for tag, inp in [
        ("B1_MLP", sample_frame),
        ("B2_LSTM", sample_window),
        ("B3_LCGC", sample_window),
    ]:
        fp32_path = os.path.join(out_dir, f"{tag}_fp32.tflite")
        if os.path.exists(fp32_path):
            try:
                t_ms = measure_tflite_inference_time(fp32_path, inp)
                print(f"  {tag:12s}  {t_ms:.3f} ms")
            except Exception as e:
                print(f"  {tag:12s}  measurement error: {e}")
        else:
            print(f"  {tag:12s}  file not found")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
