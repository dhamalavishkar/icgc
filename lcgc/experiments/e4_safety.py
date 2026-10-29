"""
E4 — Safety Gate Evaluation (G5).

Evaluates the two critical safety mechanisms:
  1) Sudden Obstacle Stop:
     Robot moving Forward at V_MAX encounters an obstacle suddenly appearing
     at distance 0.18 m (< 0.20 m stop threshold).
     Compare: WITH Safety Gate vs WITHOUT Safety Gate.
     Reports: Time-to-stop (ms), Collision occurrences, Minimum stopping distance.

  2) Hand-Lost Watchdog:
     Robot moving Forward at V_MAX when user hand disappears from webcam FOV.
     Compare: WITH Watchdog (0.5 s timeout) vs WITHOUT Safety Gate.
     Reports: Time-to-stop (ms), Uncontrolled run-away distance (m).

Outputs (in results/):
  e4_safety.csv
  e4_summary.json
  e4_safety_comparison.png
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

from synthetic_hands import GESTURE_POSES
from models import build_b3_lcgc
from pipeline import GesturePipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_or_create_model():
    k_path = os.path.join(RESULTS_DIR, "b3_lcgc.keras")
    if os.path.exists(k_path):
        return tf.keras.models.load_model(k_path)
    m = build_b3_lcgc()
    m.compile(optimizer="adam", loss="categorical_crossentropy")
    return m


def run_sudden_obstacle_experiment(model, n_trials=50):
    """Evaluate response to sudden obstacle appearance."""
    dt = 1.0 / config.FPS
    fwd_pose = GESTURE_POSES["Forward"]
    b3_tflite = os.path.join(RESULTS_DIR, "B3_LCGC_fp32.tflite")
    use_tfl = os.path.exists(b3_tflite)
    
    records = []
    trace_with_gate = []
    trace_no_gate = []
    
    for trial in range(n_trials):
        rng = np.random.default_rng(config.SEED + trial)
        
        for use_gate in [False, True]:
            mode = "full" if use_gate else "ema_hysteresis"
            pipe = GesturePipeline(
                model if not use_tfl else None,
                mode=mode,
                use_tflite=use_tfl,
                tflite_path=b3_tflite if use_tfl else None,
            )
            
            # Warm up window with Forward gesture
            for _ in range(config.WINDOW_T):
                pipe.step(fwd_pose, palm_size_frac=0.25, ultrasonic_dist=1.5, dt=dt)
                
            # Robot state
            robot_x = 0.0
            robot_v = config.V_MAX
            obstacle_pos = 1.0  # 1 metre ahead
            obstacle_appeared = False
            collision = False
            time_to_stop = None
            
            trial_v = []
            trial_d = []
            
            for step in range(90):  # 3 seconds
                t_sec = step * dt
                
                # At t = 0.5 s, sudden obstacle appears at 0.15 m ahead
                if t_sec >= 0.5 and not obstacle_appeared:
                    obstacle_appeared = True
                    obstacle_pos = robot_x + 0.15
                    
                dist_to_obs = obstacle_pos - robot_x if obstacle_appeared else 1.5
                noisy_dist = dist_to_obs + rng.normal(0, config.ULTRASONIC_NOISE_STD)
                
                (v, omega), g_name, _, _ = pipe.step(
                    fwd_pose, palm_size_frac=0.25, ultrasonic_dist=noisy_dist, dt=dt
                )
                
                robot_x += v * dt
                trial_v.append(v)
                trial_d.append(dist_to_obs)
                
                # Check collision
                if obstacle_appeared and dist_to_obs <= 0.0:
                    collision = True
                    if time_to_stop is None:
                        time_to_stop = (t_sec - 0.5) * 1000.0
                    break
                    
                if obstacle_appeared and v == 0.0 and time_to_stop is None:
                    time_to_stop = (t_sec - 0.5) * 1000.0
                    
            if time_to_stop is None:
                time_to_stop = (step * dt - 0.5) * 1000.0
                
            records.append({
                "scenario": "sudden_obstacle",
                "use_gate": use_gate,
                "trial": trial,
                "collision": collision,
                "time_to_stop_ms": time_to_stop,
                "min_distance_m": min(trial_d),
            })
            
            if trial == 0:
                if use_gate:
                    trace_with_gate = (trial_v, trial_d)
                else:
                    trace_no_gate = (trial_v, trial_d)
                    
    return records, trace_with_gate, trace_no_gate


def run_hand_lost_experiment(model, n_trials=50):
    """Evaluate response when hand leaves the camera frame."""
    dt = 1.0 / config.FPS
    fwd_pose = GESTURE_POSES["Forward"]
    b3_tflite = os.path.join(RESULTS_DIR, "B3_LCGC_fp32.tflite")
    use_tfl = os.path.exists(b3_tflite)
    
    records = []
    trace_with_gate = []
    trace_no_gate = []
    
    for trial in range(n_trials):
        for use_gate in [False, True]:
            mode = "full" if use_gate else "raw"
            pipe = GesturePipeline(
                model if not use_tfl else None,
                mode=mode,
                use_tflite=use_tfl,
                tflite_path=b3_tflite if use_tfl else None,
            )
            
            # Warm up with forward gesture
            for _ in range(config.WINDOW_T):
                pipe.step(fwd_pose, palm_size_frac=0.25, ultrasonic_dist=2.0, dt=dt)
                
            robot_x = 0.0
            hand_lost = False
            time_to_stop = None
            trial_v = []
            trial_x = []
            
            for step in range(90):  # 3 seconds
                t_sec = step * dt
                
                # Hand lost at t = 0.5 s
                if t_sec >= 0.5:
                    hand_lost = True
                    lm = None
                else:
                    lm = fwd_pose
                    
                (v, omega), g_name, _, _ = pipe.step(
                    lm, palm_size_frac=0.25, ultrasonic_dist=2.0, dt=dt
                )
                
                robot_x += v * dt
                trial_v.append(v)
                trial_x.append(robot_x)
                
                if hand_lost and v == 0.0 and time_to_stop is None:
                    time_to_stop = (t_sec - 0.5) * 1000.0
                    
            if time_to_stop is None:
                time_to_stop = 2500.0  # full test duration
                
            runaway_dist = trial_x[-1] - trial_x[int(0.5 / dt)]
            
            records.append({
                "scenario": "hand_lost",
                "use_gate": use_gate,
                "trial": trial,
                "collision": False,
                "time_to_stop_ms": time_to_stop,
                "runaway_distance_m": runaway_dist,
            })
            
            if trial == 0:
                if use_gate:
                    trace_with_gate = (trial_v, trial_x)
                else:
                    trace_no_gate = (trial_v, trial_x)
                    
    return records, trace_with_gate, trace_no_gate


def main():
    print("=" * 70)
    print("  E4: Safety Gate Evaluation (G5)")
    print("=" * 70)
    
    model = load_or_create_model()
    
    print("\nRunning Sudden Obstacle Evaluation (50 trials)...")
    obs_records, obs_gate_trace, obs_nogate_trace = run_sudden_obstacle_experiment(model, 50)
    
    print("Running Hand-Lost Watchdog Evaluation (50 trials)...")
    lost_records, lost_gate_trace, lost_nogate_trace = run_hand_lost_experiment(model, 50)
    
    all_records = obs_records + lost_records
    
    # ── Summary Calculations ──────────────────────────────────────────
    summary = {}
    for sc in ["sudden_obstacle", "hand_lost"]:
        summary[sc] = {}
        for ug in [False, True]:
            sub = [r for r in all_records if r["scenario"] == sc and r["use_gate"] == ug]
            times = [r["time_to_stop_ms"] for r in sub]
            colls = sum([1 for r in sub if r["collision"]])
            gate_tag = "with_gate" if ug else "without_gate"
            summary[sc][gate_tag] = {
                "mean_time_to_stop_ms": float(np.mean(times)),
                "std_time_to_stop_ms": float(np.std(times)),
                "collision_count": int(colls),
                "collision_rate_pct": float(colls / len(sub) * 100.0),
            }
            if sc == "sudden_obstacle":
                min_ds = [r["min_distance_m"] for r in sub]
                summary[sc][gate_tag]["mean_min_distance_m"] = float(np.mean(min_ds))
            else:
                runaways = [r["runaway_distance_m"] for r in sub]
                summary[sc][gate_tag]["mean_runaway_m"] = float(np.mean(runaways))
                
    # Print results
    print("\n--- Summary Results ---")
    print("Sudden Obstacle:")
    print(f"  Without Gate : Collisions = {summary['sudden_obstacle']['without_gate']['collision_count']}/50 "
          f"({summary['sudden_obstacle']['without_gate']['collision_rate_pct']:.0f}%), "
          f"Time to stop = {summary['sudden_obstacle']['without_gate']['mean_time_to_stop_ms']:.1f} ms")
    print(f"  With Gate    : Collisions = {summary['sudden_obstacle']['with_gate']['collision_count']}/50 "
          f"({summary['sudden_obstacle']['with_gate']['collision_rate_pct']:.0f}%), "
          f"Time to stop = {summary['sudden_obstacle']['with_gate']['mean_time_to_stop_ms']:.1f} ms")
          
    print("\nHand Lost Watchdog:")
    print(f"  Without Gate : Runaway = {summary['hand_lost']['without_gate']['mean_runaway_m']:.3f} m, "
          f"Time to stop = {summary['hand_lost']['without_gate']['mean_time_to_stop_ms']:.1f} ms")
    print(f"  With Gate    : Runaway = {summary['hand_lost']['with_gate']['mean_runaway_m']:.3f} m, "
          f"Time to stop = {summary['hand_lost']['with_gate']['mean_time_to_stop_ms']:.1f} ms")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "e4_safety.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "condition", "mean_stop_time_ms", "std_stop_time_ms",
                     "collisions", "collision_pct", "extra_metric", "extra_val"])
        for sc in ["sudden_obstacle", "hand_lost"]:
            for ug in ["without_gate", "with_gate"]:
                d = summary[sc][ug]
                extra_m = "mean_min_dist_m" if sc == "sudden_obstacle" else "mean_runaway_m"
                extra_v = d.get("mean_min_distance_m", d.get("mean_runaway_m", 0.0))
                w.writerow([
                    sc, ug, f"{d['mean_time_to_stop_ms']:.1f}", f"{d['std_time_to_stop_ms']:.1f}",
                    d["collision_count"], f"{d['collision_rate_pct']:.1f}", extra_m, f"{extra_v:.3f}"
                ])
    print(f"\nSaved: {csv_path}")
    
    with open(os.path.join(RESULTS_DIR, "e4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot Comparison Figures ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    dt = 1.0 / config.FPS
    time_pts = np.arange(90) * dt
    
    # (0, 0) Obstacle: Velocity profile
    axes[0, 0].plot(time_pts[:len(obs_nogate_trace[0])], obs_nogate_trace[0], "r--", label="Without Gate (Collision)", linewidth=2)
    axes[0, 0].plot(time_pts[:len(obs_gate_trace[0])], obs_gate_trace[0], "g-", label="With Safety Gate (Halt)", linewidth=2)
    axes[0, 0].axvline(0.5, color="gray", linestyle=":", label="Obstacle Appears (t=0.5s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Linear Velocity v (m/s)")
    axes[0, 0].set_title("Sudden Obstacle: Robot Velocity")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # (0, 1) Obstacle: Distance profile
    axes[0, 1].plot(time_pts[:len(obs_nogate_trace[1])], obs_nogate_trace[1], "r--", label="Without Gate", linewidth=2)
    axes[0, 1].plot(time_pts[:len(obs_gate_trace[1])], obs_gate_trace[1], "g-", label="With Safety Gate", linewidth=2)
    axes[0, 1].axhline(config.ULTRASONIC_STOP_DIST, color="orange", linestyle="--", label="Stop Threshold (0.20m)")
    axes[0, 1].axhline(0.0, color="black", linestyle="-", label="Contact Boundary (0.0m)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Distance to Obstacle (m)")
    axes[0, 1].set_title("Sudden Obstacle: Proximity")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # (1, 0) Hand-Lost: Velocity profile
    axes[1, 0].plot(time_pts[:len(lost_nogate_trace[0])], lost_nogate_trace[0], "r--", label="Without Watchdog", linewidth=2)
    axes[1, 0].plot(time_pts[:len(lost_gate_trace[0])], lost_gate_trace[0], "g-", label="With Watchdog (0.5s timeout)", linewidth=2)
    axes[1, 0].axvline(0.5, color="gray", linestyle=":", label="Hand Leaves FOV (t=0.5s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Linear Velocity v (m/s)")
    axes[1, 0].set_title("Hand-Lost: Robot Velocity")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    
    # (1, 1) Hand-Lost: Robot displacement
    axes[1, 1].plot(time_pts[:len(lost_nogate_trace[1])], lost_nogate_trace[1], "r--", label="Without Watchdog", linewidth=2)
    axes[1, 1].plot(time_pts[:len(lost_gate_trace[1])], lost_gate_trace[1], "g-", label="With Watchdog", linewidth=2)
    axes[1, 1].axvline(0.5, color="gray", linestyle=":", label="Hand Lost (t=0.5s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Robot Position x (m)")
    axes[1, 1].set_title("Hand-Lost: Cumulative Displacement")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle("E4: Safety Gate & Hand-Lost Watchdog Performance\n(SIMULATED — 50 Independent Trials)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "e4_safety_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")
    
    print("\n" + "=" * 70)
    print("  E4 Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
