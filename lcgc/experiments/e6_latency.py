"""
E6 — Real-Time Latency Budget and Timing Analysis.

End-to-end delay decomposition:
  End-to-End Delay = Frame Interval (33.3 ms)
                   + Landmark Extraction Stage (T_lm)
                   + Classifier Inference Time (T_cls)
                   + Hysteresis Delay: (m - 1) / 30 s
                   + Evidence Lag (EMA convergence)
                   + Serial / ROS 2 Transfer (0.7 ms)

Parameters:
  - Published best-case landmark extraction: 17.1 ms (MediaPipe Hands CPU)
  - Landmark extraction sweep: 10 to 80 ms
  - Target real-time throughput: 20 FPS (<= 50.0 ms per frame budget)
  - Hysteresis:
      General gestures (m=3): (3-1)/30 = 66.7 ms
      Stop gesture    (m=1): (1-1)/30 = 0.0 ms (Emergency stop is never delayed!)

Outputs (in results/):
  e6_latency_budget.csv
  e6_summary.json
  e6_latency_plot.png
"""

import os
import sys
import json
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  E6: Latency Budget & Real-Time Timing Analysis")
    print("=" * 70)
    
    # Check measured classifier time from E1 / models if available
    e1_json = os.path.join(RESULTS_DIR, "e1_summary.json")
    if os.path.exists(e1_json):
        with open(e1_json, "r") as f:
            e1_data = json.load(f)
            t_cls = e1_data.get("B3_LCGC", {}).get("infer_ms", 1.2)
    else:
        t_cls = 1.2  # default measured CPU inference time in ms
        
    t_frame_interval = config.FRAME_INTERVAL_MS  # 33.33 ms (30 FPS camera)
    t_serial = config.SERIAL_TRANSFER_MS         # 0.7 ms
    t_evidence_lag = (1.0 / config.FPS) * 1000.0 * 0.5  # ~16.7 ms (EMA alpha=0.4 reaches 0.8 in ~1 frame)
    
    hysteresis_general_ms = ((config.HYSTERESIS_M_DEFAULT - 1) / config.FPS) * 1000.0  # 66.7 ms
    hysteresis_stop_ms = ((config.HYSTERESIS_M_STOP - 1) / config.FPS) * 1000.0        # 0.0 ms
    
    fps_20_budget_ms = 1000.0 / config.FPS_TARGET  # 50.0 ms per frame
    
    # ── Benchmark at Baseline (T_lm = 17.1 ms) ─────────────────────────
    lm_best = config.LANDMARK_BEST_MS
    per_frame_compute_best = lm_best + t_cls + t_serial
    delay_steady_best = t_frame_interval + lm_best + t_cls + t_serial
    delay_stop_best = t_frame_interval + lm_best + t_cls + hysteresis_stop_ms + t_serial
    delay_change_best = t_frame_interval + lm_best + t_cls + hysteresis_general_ms + t_evidence_lag + t_serial
    
    # Landmark threshold where per-frame compute exceeds 50 ms (20 FPS limit)
    lm_cutoff_20fps = fps_20_budget_ms - t_cls - t_serial
    
    print("\n--- Latency Breakdown at Baseline (MediaPipe = 17.1 ms) ---")
    print(f"  Frame Interval (30 FPS)        : {t_frame_interval:.2f} ms")
    print(f"  Landmark Extraction            : {lm_best:.2f} ms")
    print(f"  Classifier Inference (B3 LCGC) : {t_cls:.2f} ms")
    print(f"  Serial / ROS Transfer          : {t_serial:.2f} ms")
    print(f"  Per-Frame Compute Time         : {per_frame_compute_best:.2f} ms (Target <= 50.0 ms -> {1000.0/per_frame_compute_best:.1f} FPS capable)")
    print(f"  Steady-State End-to-End Delay  : {delay_steady_best:.2f} ms")
    print(f"  Stop Gesture Reaction Delay    : {delay_stop_best:.2f} ms (m=1, no hysteresis lag)")
    print(f"  Gesture Change Delay (General) : {delay_change_best:.2f} ms (m=3, hysteresis + evidence)")
    print(f"  20 FPS Landmark Time Cutoff    : {lm_cutoff_20fps:.2f} ms")

    # ── Sweep Landmark Stage (10 to 80 ms) ─────────────────────────────
    sweep_lm = np.linspace(10, 80, 29)
    rows = []
    
    for lm in sweep_lm:
        frame_compute = lm + t_cls + t_serial
        steady_delay = t_frame_interval + lm + t_cls + t_serial
        stop_delay = t_frame_interval + lm + t_cls + hysteresis_stop_ms + t_serial
        change_delay = t_frame_interval + lm + t_cls + hysteresis_general_ms + t_evidence_lag + t_serial
        achievable_fps = 1000.0 / frame_compute
        meets_20fps = bool(frame_compute <= fps_20_budget_ms)
        
        rows.append({
            "landmark_time_ms": lm,
            "frame_compute_ms": frame_compute,
            "steady_delay_ms": steady_delay,
            "stop_delay_ms": stop_delay,
            "change_delay_ms": change_delay,
            "achievable_fps": achievable_fps,
            "meets_20fps": meets_20fps,
        })

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "e6_latency_budget.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["landmark_ms", "frame_compute_ms", "steady_delay_ms",
                     "stop_delay_ms", "gesture_change_delay_ms", "achievable_fps", "meets_20fps"])
        for r in rows:
            w.writerow([
                f"{r['landmark_time_ms']:.1f}", f"{r['frame_compute_ms']:.2f}",
                f"{r['steady_delay_ms']:.2f}", f"{r['stop_delay_ms']:.2f}",
                f"{r['change_delay_ms']:.2f}", f"{r['achievable_fps']:.1f}",
                r["meets_20fps"],
            ])
    print(f"\nSaved: {csv_path}")

    # ── Save JSON Summary ─────────────────────────────────────────────
    summary = {
        "frame_interval_ms": t_frame_interval,
        "classifier_time_ms": t_cls,
        "serial_transfer_ms": t_serial,
        "landmark_baseline_ms": lm_best,
        "per_frame_compute_baseline_ms": per_frame_compute_best,
        "steady_delay_baseline_ms": delay_steady_best,
        "stop_delay_baseline_ms": delay_stop_best,
        "gesture_change_baseline_ms": delay_change_best,
        "landmark_time_20fps_cutoff_ms": float(lm_cutoff_20fps),
        "target_fps": config.FPS_TARGET,
        "target_budget_ms": fps_20_budget_ms,
    }
    with open(os.path.join(RESULTS_DIR, "e6_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot Latency Sweep ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sweep_lm, [r["steady_delay_ms"] for r in rows],
            label="Steady-State Delay", color="#4C72B0", linewidth=2.2)
    ax.plot(sweep_lm, [r["stop_delay_ms"] for r in rows],
            label="Stop Gesture Delay (m=1, no hysteresis lag)", color="#55A868", linestyle="--", linewidth=2.2)
    ax.plot(sweep_lm, [r["change_delay_ms"] for r in rows],
            label="Gesture-Change Delay (m=3 + evidence lag)", color="#C44E52", linewidth=2.2)
    
    # Mark the baseline
    ax.axvline(lm_best, color="blue", linestyle=":", label=f"MediaPipe Baseline ({lm_best:.1f} ms)")
    
    # Mark 20 FPS threshold for compute stage
    ax.axvline(lm_cutoff_20fps, color="crimson", linestyle="-.",
               label=f"20 FPS Limit Cutoff (T_lm = {lm_cutoff_20fps:.1f} ms)")
    
    # Horizontal line for 50 ms budget
    ax.axhline(fps_20_budget_ms, color="gray", linestyle="--", alpha=0.6, label="20 FPS Frame Budget (50 ms)")
    
    ax.set_xlabel("Landmark Extraction Time T_lm (ms)")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title("E6: End-to-End Latency Budget & Real-Time Limits\n(SIMULATED — Parametric Timing Model)")
    ax.set_xlim(10, 80)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    
    # Annotate cutoff
    ax.annotate(
        f"20 FPS Lost if\nT_lm > {lm_cutoff_20fps:.1f} ms",
        xy=(lm_cutoff_20fps, 120),
        xytext=(lm_cutoff_20fps + 3, 140),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
    )
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "e6_latency_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")
    
    print("\n" + "=" * 70)
    print("  E6 Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
