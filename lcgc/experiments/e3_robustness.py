"""
E3 — Robustness Sweeps & Invariance Verification (G1, G5).

Evaluates B1, B2, B3 accuracy across:
  1) Landmark Gaussian noise (sigma in 0.00 .. 0.10)
  2) Random landmark dropout probability (p in 0.00 .. 0.25)
  3) Distance / scale variation (scale in 0.4 .. 2.0)
  4) In-plane rotation (angle in -45 deg .. +45 deg)

Also empirically verifies algebraic scale and shift invariance:
  Max absolute difference between normalized features of p and (2.0 * p + shift).

Outputs (in results/):
  e3_invariance.json
  e3_sweeps.csv
  e3_robustness_plots.png
"""

import os
import sys
import json
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)

from synthetic_hands import (
    CANONICAL_HAND, GESTURE_POSES, normalize_landmarks,
    generate_training_windows, _rotate_hand
)
from models import build_b1_mlp, build_b2_lstm, build_b3_lcgc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def verify_scale_shift_invariance():
    """Verify that normalization makes features invariant to uniform scale and translation."""
    print("\n--- Verifying Scale and Translation Invariance ---")
    raw_hand = CANONICAL_HAND.copy()
    
    # Base normalized vector
    v_base = normalize_landmarks(raw_hand)
    
    # Scale by 2.5 and shift by (120.0, -45.0)
    scaled_shifted_hand = raw_hand * 2.5 + np.array([120.0, -45.0])
    v_transformed = normalize_landmarks(scaled_shifted_hand)
    
    max_diff = float(np.max(np.abs(v_base - v_transformed)))
    mean_diff = float(np.mean(np.abs(v_base - v_transformed)))
    
    print(f"  Base vector shape: {v_base.shape}")
    print(f"  Max absolute difference after 2.5x scale and shift: {max_diff:.3e}")
    print(f"  Mean absolute difference                         : {mean_diff:.3e}")
    
    invariance_record = {
        "scale_factor": 2.5,
        "shift_vector": [120.0, -45.0],
        "max_abs_difference": max_diff,
        "mean_abs_difference": mean_diff,
        "invariant": bool(max_diff < 1e-6),
    }
    with open(os.path.join(RESULTS_DIR, "e3_invariance.json"), "w") as f:
        json.dump(invariance_record, f, indent=2)
    return max_diff


def load_or_train_all_models():
    """Load or quickly train B1, B2, B3 models on synthetic data."""
    models = {}
    from synthetic_hands import generate_training_windows
    X, y, _ = generate_training_windows(seed=config.SEED)
    y_oh = tf.keras.utils.to_categorical(y, config.NUM_CLASSES)
    
    for tag, build_fn, is_mlp in [
        ("B1_MLP", build_b1_mlp, True),
        ("B2_LSTM", build_b2_lstm, False),
        ("B3_LCGC", build_b3_lcgc, False),
    ]:
        k_path = os.path.join(RESULTS_DIR, f"{tag.lower()}.keras")
        if os.path.exists(k_path):
            print(f"Loading {tag} from {k_path}...")
            models[tag] = tf.keras.models.load_model(k_path)
        else:
            print(f"Training {tag}...")
            m = build_fn()
            m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            inp = X[:, -1, :] if is_mlp else X
            m.fit(inp, y_oh, epochs=15, batch_size=config.BATCH_SIZE, verbose=0)
            m.save(k_path)
            models[tag] = m
            
    return models


def evaluate_model(model, X_test, y_test, is_mlp=False):
    inp = X_test[:, -1, :] if is_mlp else X_test
    preds = model.predict(inp, verbose=0)
    acc = float(np.mean(np.argmax(preds, axis=1) == y_test))
    return acc


def run_sweeps(models):
    print("\n--- Running Robustness Parameter Sweeps ---")
    rng = np.random.default_rng(config.SEED + 99)
    
    # Generate clean test windows across classes
    N_test_per_class = 60
    base_windows = []
    base_labels = []
    for c_idx, g_name in enumerate(config.GESTURE_NAMES):
        pose = GESTURE_POSES[g_name]
        for _ in range(N_test_per_class):
            win = np.tile(pose, (config.WINDOW_T, 1, 1))  # (15, 21, 2)
            base_windows.append(win)
            base_labels.append(c_idx)
    base_windows = np.array(base_windows)  # (360, 15, 21, 2)
    base_labels = np.array(base_labels)
    
    sweeps = {
        "noise_sigma": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
        "dropout_prob": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25],
        "distance_scale": [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0],
        "rotation_deg": [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0],
    }
    
    sweep_results = []
    
    # 1. Noise Sweep
    for val in sweeps["noise_sigma"]:
        noisy_wins = base_windows + rng.normal(0, val, base_windows.shape)
        # normalize
        X_norm = np.zeros((len(base_windows), config.WINDOW_T, config.FEATURE_DIM), dtype=np.float32)
        for i in range(len(base_windows)):
            for t in range(config.WINDOW_T):
                X_norm[i, t] = normalize_landmarks(noisy_wins[i, t])
        for tag in ["B1_MLP", "B2_LSTM", "B3_LCGC"]:
            acc = evaluate_model(models[tag], X_norm, base_labels, is_mlp=(tag == "B1_MLP"))
            sweep_results.append({"sweep": "noise_sigma", "value": val, "model": tag, "accuracy": acc})

    # 2. Dropout Sweep
    for val in sweeps["dropout_prob"]:
        drop_wins = base_windows.copy()
        mask = rng.random(drop_wins.shape[:3]) < val  # (N, 15, 21)
        drop_wins[mask] = 0.0
        X_norm = np.zeros((len(base_windows), config.WINDOW_T, config.FEATURE_DIM), dtype=np.float32)
        for i in range(len(base_windows)):
            for t in range(config.WINDOW_T):
                X_norm[i, t] = normalize_landmarks(drop_wins[i, t])
        for tag in ["B1_MLP", "B2_LSTM", "B3_LCGC"]:
            acc = evaluate_model(models[tag], X_norm, base_labels, is_mlp=(tag == "B1_MLP"))
            sweep_results.append({"sweep": "dropout_prob", "value": val, "model": tag, "accuracy": acc})

    # 3. Distance / Scale Sweep
    for val in sweeps["distance_scale"]:
        scale_wins = base_windows * val
        X_norm = np.zeros((len(base_windows), config.WINDOW_T, config.FEATURE_DIM), dtype=np.float32)
        for i in range(len(base_windows)):
            for t in range(config.WINDOW_T):
                X_norm[i, t] = normalize_landmarks(scale_wins[i, t])
        for tag in ["B1_MLP", "B2_LSTM", "B3_LCGC"]:
            acc = evaluate_model(models[tag], X_norm, base_labels, is_mlp=(tag == "B1_MLP"))
            sweep_results.append({"sweep": "distance_scale", "value": val, "model": tag, "accuracy": acc})

    # 4. Rotation Sweep
    for val in sweeps["rotation_deg"]:
        rot_wins = np.zeros_like(base_windows)
        for i in range(len(base_windows)):
            for t in range(config.WINDOW_T):
                rot_wins[i, t] = _rotate_hand(base_windows[i, t], val)
        X_norm = np.zeros((len(base_windows), config.WINDOW_T, config.FEATURE_DIM), dtype=np.float32)
        for i in range(len(base_windows)):
            for t in range(config.WINDOW_T):
                X_norm[i, t] = normalize_landmarks(rot_wins[i, t])
        for tag in ["B1_MLP", "B2_LSTM", "B3_LCGC"]:
            acc = evaluate_model(models[tag], X_norm, base_labels, is_mlp=(tag == "B1_MLP"))
            sweep_results.append({"sweep": "rotation_deg", "value": val, "model": tag, "accuracy": acc})

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "e3_sweeps.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "param_value", "model", "accuracy"])
        for r in sweep_results:
            w.writerow([r["sweep"], r["value"], r["model"], f"{r['accuracy']:.4f}"])
    print(f"Saved: {csv_path}")

    # Plot 4-panel publication-quality figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    colors = {"B1_MLP": "#C44E52", "B2_LSTM": "#DD8452", "B3_LCGC": "#4C72B0"}
    markers = {"B1_MLP": "s", "B2_LSTM": "^", "B3_LCGC": "o"}
    
    sweep_info = [
        ("noise_sigma", "Landmark Gaussian Noise (sigma)", axes[0, 0]),
        ("dropout_prob", "Landmark Dropout Probability", axes[0, 1]),
        ("distance_scale", "Palm Scale Multiplier", axes[1, 0]),
        ("rotation_deg", "In-Plane Rotation (degrees)", axes[1, 1]),
    ]
    
    for sw_name, label, ax in sweep_info:
        for tag in ["B1_MLP", "B2_LSTM", "B3_LCGC"]:
            sub = [r for r in sweep_results if r["sweep"] == sw_name and r["model"] == tag]
            xs = [r["value"] for r in sub]
            ys = [r["accuracy"] * 100.0 for r in sub]
            ax.plot(xs, ys, label=tag, color=colors[tag], marker=markers[tag], linewidth=1.8, markersize=5)
        ax.set_xlabel(label)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left" if sw_name in ["noise_sigma", "dropout_prob"] else "best")
        ax.set_title(label)
        
    plt.suptitle("E3: Robustness Sweeps across Models\n(SIMULATED — Synthetic Perturbations)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "e3_robustness_plots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")
    return sweep_results


def main():
    print("=" * 70)
    print("  E3: Robustness Sweeps and Invariance Verification")
    print("=" * 70)
    
    verify_scale_shift_invariance()
    models = load_or_train_all_models()
    run_sweeps(models)
    
    print("\n" + "=" * 70)
    print("  E3 Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
