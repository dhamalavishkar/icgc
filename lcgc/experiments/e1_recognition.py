"""
E1 — Leave-one-subject-out recognition evaluation (G1, G2, G4).

For each model (B1 MLP, B2 LSTM, B3 LCGC):
  - Train on 11 subjects, test on 1, repeat for all 12 subjects.
  - Report accuracy, macro-F1, confusion matrix, mean ± SD over subjects.
  - Model-size/latency comparison table.

Outputs (in results/):
  e1_metrics.csv         – per-subject accuracy & F1
  e1_confusion_b3.png    – confusion matrix for B3 (aggregated)
  e1_model_comparison.csv  – params, MACs, sizes, latency
  e1_bar_chart.png       – grouped bar chart of accuracy & F1
"""

import os
import sys
import json
import time
import numpy as np
import csv

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)

from synthetic_hands import generate_training_windows
from models import (build_b1_mlp, build_b2_lstm, build_b3_lcgc,
                    count_params, estimate_macs, save_fp32_model,
                    quantize_int8, measure_inference_time)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_evaluate(model_fn, X_all, y_all, subj_ids, is_mlp=False):
    """Leave-one-subject-out cross-validation.
    
    Returns: list of dicts with per-subject metrics, and aggregated confusion.
    """
    unique_subjects = np.unique(subj_ids)
    results = []
    all_y_true = []
    all_y_pred = []
    
    for test_subj in unique_subjects:
        # Split
        train_mask = subj_ids != test_subj
        test_mask = subj_ids == test_subj
        
        if is_mlp:
            # MLP uses only the last frame
            X_train = X_all[train_mask][:, -1, :]   # (N, 42)
            X_test = X_all[test_mask][:, -1, :]
        else:
            X_train = X_all[train_mask]
            X_test = X_all[test_mask]
        
        y_train = y_all[train_mask]
        y_test = y_all[test_mask]
        
        # One-hot encode
        y_train_oh = tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES)
        y_test_oh = tf.keras.utils.to_categorical(y_test, config.NUM_CLASSES)
        
        # Build fresh model
        tf.random.set_seed(config.SEED)
        model = model_fn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # Train
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=0,
        )
        
        model.fit(
            X_train, y_train_oh,
            validation_split=0.15,
            epochs=25,
            batch_size=config.BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )
        
        # Predict
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        results.append({
            "subject": int(test_subj),
            "accuracy": float(acc),
            "macro_f1": float(f1),
        })
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        print(f"  Subject {int(test_subj):2d}: acc={acc:.4f}  F1={f1:.4f}")
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    return results, cm, model


def main():
    print("=" * 70)
    print("  E1: Leave-One-Subject-Out Recognition Evaluation")
    print("=" * 70)
    
    # Generate data
    print("\nGenerating synthetic training data...")
    X, y, subj_ids = generate_training_windows(seed=config.SEED)
    print(f"  Data shape: X={X.shape}, y={y.shape}, subjects={np.unique(subj_ids)}")
    
    model_configs = [
        ("B1_MLP", build_b1_mlp, True),
        ("B2_LSTM", build_b2_lstm, False),
        ("B3_LCGC", build_b3_lcgc, False),
    ]
    
    all_results = {}
    all_cms = {}
    model_info = {}
    
    for model_name, model_fn, is_mlp in model_configs:
        print(f"\n{'-' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'-' * 60}")
        
        results, cm, last_model = train_and_evaluate(
            model_fn, X, y, subj_ids, is_mlp=is_mlp
        )
        
        all_results[model_name] = results
        all_cms[model_name] = cm
        
        # Save a reference trained model
        keras_path = os.path.join(RESULTS_DIR, f"{model_name.lower()}.keras")
        last_model.save(keras_path)
        print(f"  Saved trained model -> {keras_path}")
        
        accs = [r["accuracy"] for r in results]
        f1s = [r["macro_f1"] for r in results]
        print(f"\n  Mean accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  Mean F1:       {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        
        # Model info
        params = count_params(last_model)
        macs = estimate_macs(last_model)
        
        # Inference time
        if is_mlp:
            sample = np.random.randn(config.FEATURE_DIM).astype(np.float32)
        else:
            sample = np.random.randn(config.WINDOW_T, config.FEATURE_DIM).astype(np.float32)
        infer_ms = measure_inference_time(last_model, sample, n_runs=50)
        
        # Save FP32 and INT8
        fp32_path = os.path.join(RESULTS_DIR, f"{model_name}_fp32.tflite")
        fp32_size = save_fp32_model(last_model, fp32_path)
        
        if is_mlp:
            rep_data = np.random.randn(config.NUM_REPRESENTATIVE_WINDOWS, config.FEATURE_DIM).astype(np.float32)
        else:
            rep_data = np.random.randn(config.NUM_REPRESENTATIVE_WINDOWS, config.WINDOW_T, config.FEATURE_DIM).astype(np.float32)
        
        int8_path = os.path.join(RESULTS_DIR, f"{model_name}_int8.tflite")
        int8_size = quantize_int8(last_model, rep_data, int8_path)
        int8_kb = (int8_size / 1024.0) if int8_size > 0 else -1.0
        
        model_info[model_name] = {
            "params": params,
            "macs": macs,
            "fp32_kb": fp32_size / 1024.0,
            "int8_kb": int8_kb,
            "infer_ms": infer_ms,
        }
        print(f"  Params: {params:,}  MACs: {macs:,}")
        int8_str = f"{int8_kb:.1f} KB" if int8_kb > 0 else "Unsupported"
        print(f"  FP32: {fp32_size/1024:.1f} KB  INT8: {int8_str}")
        print(f"  Inference: {infer_ms:.3f} ms")
    
    # ── Save CSV: per-subject metrics ─────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "e1_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subject", "accuracy", "macro_f1"])
        for model_name, results in all_results.items():
            for r in results:
                w.writerow([model_name, r["subject"], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}"])
    print(f"\nSaved: {csv_path}")
    
    # ── Save CSV: model comparison ────────────────────────────────────
    comp_path = os.path.join(RESULTS_DIR, "e1_model_comparison.csv")
    with open(comp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "macs", "fp32_kb", "int8_kb", "infer_ms",
                     "mean_accuracy", "std_accuracy", "mean_f1", "std_f1"])
        for model_name in model_info:
            mi = model_info[model_name]
            accs = [r["accuracy"] for r in all_results[model_name]]
            f1s = [r["macro_f1"] for r in all_results[model_name]]
            int8_display = f"{mi['int8_kb']:.1f}" if mi['int8_kb'] > 0 else "N/A"
            w.writerow([
                model_name, mi["params"], mi["macs"],
                f"{mi['fp32_kb']:.1f}", int8_display, f"{mi['infer_ms']:.3f}",
                f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}",
                f"{np.mean(f1s):.4f}", f"{np.std(f1s):.4f}",
            ])
    print(f"Saved: {comp_path}")
    
    # ── Save JSON summary ─────────────────────────────────────────────
    summary = {}
    for model_name in all_results:
        accs = [r["accuracy"] for r in all_results[model_name]]
        f1s = [r["macro_f1"] for r in all_results[model_name]]
        summary[model_name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            **model_info[model_name],
        }
    json_path = os.path.join(RESULTS_DIR, "e1_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")
    
    # ── Plot: grouped bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35
    
    mean_accs = [np.mean([r["accuracy"] for r in all_results[m]]) for m in model_names]
    std_accs = [np.std([r["accuracy"] for r in all_results[m]]) for m in model_names]
    mean_f1s = [np.mean([r["macro_f1"] for r in all_results[m]]) for m in model_names]
    std_f1s = [np.std([r["macro_f1"] for r in all_results[m]]) for m in model_names]
    
    bars1 = ax.bar(x - width/2, mean_accs, width, yerr=std_accs,
                   label="Accuracy", capsize=5, color="#4C72B0")
    bars2 = ax.bar(x + width/2, mean_f1s, width, yerr=std_f1s,
                   label="Macro-F1", capsize=5, color="#DD8452")
    
    ax.set_ylabel("Score")
    ax.set_title("E1: Leave-One-Subject-Out Recognition\n(SIMULATED — Synthetic Data)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "e1_bar_chart.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")
    
    # ── Plot: confusion matrix for B3 ─────────────────────────────────
    cm = all_cms["B3_LCGC"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.GESTURE_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_yticklabels(config.GESTURE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("E1: B3 LCGC Confusion Matrix (Aggregated)\n(SIMULATED — Synthetic Data)")
    
    # Add text annotations
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "e1_confusion_b3.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")
    
    print("\n" + "=" * 70)
    print("  E1 Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
