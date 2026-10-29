"""
E2 — Continuous-Tracking Ablation (G3, G5).

On continuous gesture streams across all subjects, compare:
  1) Raw argmax
  2) EMA only
  3) EMA + hysteresis

Reports:
  - Per-frame accuracy (%)
  - False-trigger rate per minute
  - Command flicker count (rapid direction reversals / chatter)
  - Latency after gesture change (ms)

Outputs (in results/):
  e2_ablation.csv
  e2_summary.json
  e2_continuous_trace.png
  e2_metrics_bar.png
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

from synthetic_hands import generate_all_streams, CANONICAL_HAND
from models import build_b3_lcgc
from pipeline import GesturePipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_or_train_b3():
    """Load pre-trained B3 model from results/ or train on synthetic data."""
    keras_path = os.path.join(RESULTS_DIR, "b3_lcgc.keras")
    if os.path.exists(keras_path):
        print(f"Loading existing B3 model from {keras_path}...")
        return tf.keras.models.load_model(keras_path)
    
    print("Training B3 model for continuous stream ablation...")
    from synthetic_hands import generate_training_windows
    X, y, subj = generate_training_windows(seed=config.SEED)
    y_oh = tf.keras.utils.to_categorical(y, config.NUM_CLASSES)
    
    model = build_b3_lcgc()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=4, restore_best_weights=True, verbose=0
    )
    model.fit(X, y_oh, epochs=20, batch_size=config.BATCH_SIZE, callbacks=[early_stop], verbose=1)
    model.save(keras_path)
    return model


def compute_flicker_count(commands):
    """Count rapid command oscillations: A -> B -> A within 1-2 frames."""
    flickers = 0
    for i in range(1, len(commands) - 1):
        prev_cmd = commands[i - 1]
        curr_cmd = commands[i]
        next_cmd = commands[i + 1]
        if curr_cmd != prev_cmd and next_cmd == prev_cmd:
            flickers += 1
    return flickers


def compute_transition_latency(true_labels, predicted_labels, fps=config.FPS):
    """Measure delay (ms) from ground-truth gesture change to matching pipeline prediction."""
    latencies = []
    T = len(true_labels)
    i = 1
    while i < T:
        if true_labels[i] != true_labels[i - 1]:
            new_target = true_labels[i]
            change_frame = i
            # Look ahead up to 60 frames (2 seconds)
            matched = False
            for j in range(change_frame, min(change_frame + 60, T)):
                if predicted_labels[j] == new_target:
                    delay_frames = j - change_frame
                    latencies.append((delay_frames / fps) * 1000.0)
                    matched = True
                    break
            if not matched:
                latencies.append(60.0 / fps * 1000.0)  # capped
            # advance past this segment
            while i < T and true_labels[i] == new_target:
                i += 1
        else:
            i += 1
    return np.mean(latencies) if len(latencies) > 0 else 0.0


def compute_false_triggers(true_labels, predicted_labels, fps=config.FPS):
    """Count false action triggers (predicted motion when true was Idle/Stop or wrong action)."""
    idle_stop_idxs = {config.GESTURE_TO_IDX["Idle"], config.GESTURE_TO_IDX["Stop"]}
    false_triggers = 0
    in_false_trigger = False
    
    for y_true, y_pred in zip(true_labels, predicted_labels):
        if y_true in idle_stop_idxs and y_pred not in idle_stop_idxs:
            if not in_false_trigger:
                false_triggers += 1
                in_false_trigger = True
        elif y_pred in idle_stop_idxs or y_pred == y_true:
            in_false_trigger = False
            
    total_minutes = len(true_labels) / (fps * 60.0)
    rate_per_min = false_triggers / max(total_minutes, 1e-5)
    return false_triggers, rate_per_min


def evaluate_stream(pipeline, frames, labels, hand_present):
    """Feed frames through pipeline and collect predictions."""
    pipeline.reset()
    predicted_gestures = []
    predicted_indices = []
    commands = []
    
    for t in range(len(frames)):
        lm = frames[t].reshape(config.NUM_LANDMARKS, 2) if hand_present[t] else None
        (v, omega), g_name, raw_p, sm_p = pipeline.step(
            lm, palm_size_frac=0.18, ultrasonic_dist=1.5, dt=1.0/config.FPS
        )
        predicted_gestures.append(g_name)
        predicted_indices.append(config.GESTURE_TO_IDX[g_name])
        commands.append((round(v, 2), round(omega, 2)))
        
    return np.array(predicted_indices), commands


def main():
    print("=" * 70)
    print("  E2: Continuous-Tracking Ablation Evaluation")
    print("=" * 70)
    
    model = load_or_train_b3()
    streams = generate_all_streams(seed=config.SEED)
    
    modes = [
        ("Raw Argmax", "raw"),
        ("EMA Only", "ema"),
        ("EMA + Hysteresis", "ema_hysteresis"),
    ]
    
    results_by_mode = {name: {"acc": [], "false_rate": [], "flickers": [], "latency": []}
                       for name, _ in modes}
    
    # Store one trace for plotting (Subject 0)
    trace_data = {}
    
    for mode_name, mode_val in modes:
        print(f"\nEvaluating mode: {mode_name} ({mode_val})...")
        pipe = GesturePipeline(model, mode=mode_val)
        
        for subj_id, stream in streams.items():
            frames = stream["frames"]
            labels = stream["labels"]
            hand_present = stream["hand_present"]
            
            pred_idxs, cmds = evaluate_stream(pipe, frames, labels, hand_present)
            
            # Metrics
            acc = float(np.mean(pred_idxs == labels) * 100.0)
            _, f_rate = compute_false_triggers(labels, pred_idxs)
            flick = compute_flicker_count(cmds)
            lat = compute_transition_latency(labels, pred_idxs)
            
            results_by_mode[mode_name]["acc"].append(acc)
            results_by_mode[mode_name]["false_rate"].append(f_rate)
            results_by_mode[mode_name]["flickers"].append(flick)
            results_by_mode[mode_name]["latency"].append(lat)
            
            if subj_id == 0:
                trace_data[mode_name] = pred_idxs
                if "true" not in trace_data:
                    trace_data["true"] = labels
                    
        print(f"  Per-frame Accuracy   : {np.mean(results_by_mode[mode_name]['acc']):.2f}%")
        print(f"  False Triggers / min : {np.mean(results_by_mode[mode_name]['false_rate']):.2f}")
        print(f"  Command Flickers     : {np.mean(results_by_mode[mode_name]['flickers']):.1f}")
        print(f"  Transition Latency   : {np.mean(results_by_mode[mode_name]['latency']):.1f} ms")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "e2_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "accuracy_mean", "accuracy_std",
                     "false_triggers_per_min_mean", "false_triggers_per_min_std",
                     "command_flickers_mean", "command_flickers_std",
                     "latency_ms_mean", "latency_ms_std"])
        for mode_name, _ in modes:
            d = results_by_mode[mode_name]
            w.writerow([
                mode_name,
                f"{np.mean(d['acc']):.2f}", f"{np.std(d['acc']):.2f}",
                f"{np.mean(d['false_rate']):.2f}", f"{np.std(d['false_rate']):.2f}",
                f"{np.mean(d['flickers']):.1f}", f"{np.std(d['flickers']):.1f}",
                f"{np.mean(d['latency']):.1f}", f"{np.std(d['latency']):.1f}",
            ])
    print(f"\nSaved: {csv_path}")
    
    # ── Save JSON summary ─────────────────────────────────────────────
    summary = {}
    for mode_name, _ in modes:
        d = results_by_mode[mode_name]
        summary[mode_name] = {
            "accuracy_mean": float(np.mean(d["acc"])),
            "accuracy_std": float(np.std(d["acc"])),
            "false_triggers_per_min_mean": float(np.mean(d["false_rate"])),
            "false_triggers_per_min_std": float(np.std(d["false_rate"])),
            "flickers_mean": float(np.mean(d["flickers"])),
            "flickers_std": float(np.std(d["flickers"])),
            "latency_ms_mean": float(np.mean(d["latency"])),
            "latency_ms_std": float(np.std(d["latency"])),
        }
    with open(os.path.join(RESULTS_DIR, "e2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot 1: 15-second trace comparison ───────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    time_pts = np.arange(450) / float(config.FPS)  # first 15 seconds (450 frames)
    
    axes[0].step(time_pts, trace_data["true"][:450], label="Ground Truth", color="black", linewidth=2)
    axes[0].set_ylabel("Class")
    axes[0].set_title("Ground Truth Sequence")
    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels(config.GESTURE_NAMES)
    axes[0].grid(alpha=0.3)
    
    colors = ["#C44E52", "#EAA944", "#4C72B0"]
    for idx, (mode_name, _) in enumerate(modes):
        ax = axes[idx + 1]
        ax.step(time_pts, trace_data[mode_name][:450], label=mode_name, color=colors[idx], linewidth=1.5)
        ax.set_ylabel("Class")
        ax.set_title(f"Predicted: {mode_name}")
        ax.set_yticks(range(6))
        ax.set_yticklabels(config.GESTURE_NAMES)
        ax.grid(alpha=0.3)
        
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("E2: Continuous Gesture Stream Tracking Ablation\n(SIMULATED — Subject 0 Trace)", fontsize=13)
    plt.tight_layout()
    trace_path = os.path.join(RESULTS_DIR, "e2_continuous_trace.png")
    plt.savefig(trace_path, dpi=300)
    plt.close()
    print(f"Saved: {trace_path}")
    
    # ── Plot 2: Metrics bar chart ─────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    m_labels = [m[0] for m in modes]
    
    # (0,0) Accuracy
    acc_m = [np.mean(results_by_mode[m]["acc"]) for m in m_labels]
    acc_s = [np.std(results_by_mode[m]["acc"]) for m in m_labels]
    axs[0, 0].bar(m_labels, acc_m, yerr=acc_s, capsize=4, color="#4C72B0")
    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[0, 0].set_title("Per-Frame Tracking Accuracy")
    axs[0, 0].grid(axis="y", alpha=0.3)
    
    # (0,1) False Triggers
    ft_m = [np.mean(results_by_mode[m]["false_rate"]) for m in m_labels]
    ft_s = [np.std(results_by_mode[m]["false_rate"]) for m in m_labels]
    axs[0, 1].bar(m_labels, ft_m, yerr=ft_s, capsize=4, color="#DD8452")
    axs[0, 1].set_ylabel("False Triggers / min")
    axs[0, 1].set_title("False Trigger Rate (Lower is Better)")
    axs[0, 1].grid(axis="y", alpha=0.3)
    
    # (1,0) Flickers
    fl_m = [np.mean(results_by_mode[m]["flickers"]) for m in m_labels]
    fl_s = [np.std(results_by_mode[m]["flickers"]) for m in m_labels]
    axs[1, 0].bar(m_labels, fl_m, yerr=fl_s, capsize=4, color="#55A868")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_title("Command Flickers (Lower is Better)")
    axs[1, 0].grid(axis="y", alpha=0.3)
    
    # (1,1) Latency
    lat_m = [np.mean(results_by_mode[m]["latency"]) for m in m_labels]
    lat_s = [np.std(results_by_mode[m]["latency"]) for m in m_labels]
    axs[1, 1].bar(m_labels, lat_m, yerr=lat_s, capsize=4, color="#8172B2")
    axs[1, 1].set_ylabel("Latency (ms)")
    axs[1, 1].set_title("Gesture-Change Latency")
    axs[1, 1].grid(axis="y", alpha=0.3)
    
    plt.suptitle("E2: Continuous Tracking Ablation Metrics\n(SIMULATED — Synthetic Data, 12 Subjects)", fontsize=13)
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "e2_metrics_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")
    
    print("\n" + "=" * 70)
    print("  E2 Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
