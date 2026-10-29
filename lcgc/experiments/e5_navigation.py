"""
E5 — End-to-End Navigation Simulation (G3, G5).

Evaluates 100+ simulated navigation trials per pipeline across 3 operator styles:
  P1: Single-frame model (B1 MLP), raw argmax, no safety gate
  P2: LCGC model (B3 DS-CNN), EMA + hysteresis, no safety gate
  P3: Full LCGC pipeline (B3 DS-CNN), EMA + hysteresis + safety gate

Reports:
  - Navigation success rate with Wilson 95% score confidence intervals
  - Mean completion time (steps / seconds)
  - Collision count
  - Animated GIFs of a successful trial and a failed trial

Outputs (in results/):
  e5_navigation.csv
  e5_summary.json
  e5_navigation_comparison.png
  e5_successful_trial.gif
  e5_failed_trial.gif
"""

import os
import sys
import json
import math
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)

from models import build_b1_mlp, build_b3_lcgc
from pipeline import GesturePipeline
from robot_sim import Arena, DiffDriveRobot, Simulation, save_animation, DEFAULT_OBSTACLES, DEFAULT_START, DEFAULT_GOAL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def wilson_score_interval(successes, total, confidence=0.95):
    """Calculate Wilson 95% score interval for a binomial proportion."""
    if total == 0:
        return 0.0, (0.0, 0.0)
    p_hat = successes / total
    z = 1.95996  # 95% confidence z-score
    denom = 1.0 + (z**2) / total
    centre = (p_hat + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2)))
    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)
    return p_hat, (ci_lower, ci_upper)


def load_or_train_models():
    """Load pre-trained models or create baseline models."""
    models = {}
    from synthetic_hands import generate_training_windows
    X, y, _ = generate_training_windows(seed=config.SEED)
    y_oh = tf.keras.utils.to_categorical(y, config.NUM_CLASSES)
    
    # B1
    b1_path = os.path.join(RESULTS_DIR, "b1_mlp.keras")
    if os.path.exists(b1_path):
        b1 = tf.keras.models.load_model(b1_path)
    else:
        b1 = build_b1_mlp()
        b1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        b1.fit(X[:, -1, :], y_oh, epochs=15, batch_size=config.BATCH_SIZE, verbose=0)
        b1.save(b1_path)
    models["B1"] = b1
    
    # B3
    b3_path = os.path.join(RESULTS_DIR, "b3_lcgc.keras")
    if os.path.exists(b3_path):
        b3 = tf.keras.models.load_model(b3_path)
    else:
        b3 = build_b3_lcgc()
        b3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        b3.fit(X, y_oh, epochs=15, batch_size=config.BATCH_SIZE, verbose=0)
        b3.save(b3_path)
    models["B3"] = b3
    
    return models


def run_navigation_evaluation(models, n_trials=100):
    """Run n_trials for P1, P2, P3 across 3 operator styles."""
    styles = ["normal", "aggressive", "cautious"]
    
    b1_tflite = os.path.join(RESULTS_DIR, "B1_MLP_fp32.tflite")
    b3_tflite = os.path.join(RESULTS_DIR, "B3_LCGC_fp32.tflite")
    use_tflite_b1 = os.path.exists(b1_tflite)
    use_tflite_b3 = os.path.exists(b3_tflite)
    
    pipelines = {
        "P1 (Single-Frame Raw)": {
            "model": models["B1"] if not use_tflite_b1 else None,
            "mode": "raw", "is_mlp": True,
            "use_tflite": use_tflite_b1, "tflite_path": b1_tflite if use_tflite_b1 else None,
        },
        "P2 (LCGC No Gate)": {
            "model": models["B3"] if not use_tflite_b3 else None,
            "mode": "ema_hysteresis", "is_mlp": False,
            "use_tflite": use_tflite_b3, "tflite_path": b3_tflite if use_tflite_b3 else None,
        },
        "P3 (Full LCGC)": {
            "model": models["B3"] if not use_tflite_b3 else None,
            "mode": "full", "is_mlp": False,
            "use_tflite": use_tflite_b3, "tflite_path": b3_tflite if use_tflite_b3 else None,
        },
    }
    
    results = {}
    successful_trial_data = None
    failed_trial_data = None
    arena_ref = Arena(obstacles=DEFAULT_OBSTACLES, goal=DEFAULT_GOAL)
    
    for p_name, p_cfg in pipelines.items():
        print(f"\nEvaluating {p_name} across {n_trials} trials...")
        p_results = []
        
        for trial_i in range(n_trials):
            style = styles[trial_i % len(styles)]
            seed = config.SEED + trial_i * 13
            
            # Instantiate pipeline
            pipe = GesturePipeline(
                p_cfg["model"], mode=p_cfg["mode"],
                use_tflite=p_cfg.get("use_tflite", False),
                tflite_path=p_cfg.get("tflite_path", None)
            )
            pipe._is_mlp = p_cfg["is_mlp"]
            
            # Setup simulation
            sim = Simulation(
                arena=Arena(obstacles=DEFAULT_OBSTACLES, goal=DEFAULT_GOAL),
                start=DEFAULT_START,
                goal=DEFAULT_GOAL,
                operator_style=style,
                pipeline=pipe,
                seed=seed,
            )
            
            trial_out = sim.run_trial(max_steps=config.NAV_MAX_STEPS)
            p_results.append({
                "trial": trial_i,
                "operator_style": style,
                "success": trial_out["success"],
                "steps": trial_out["steps"],
                "time_sec": trial_out["steps"] * config.SIM_DT,
                "collisions": trial_out["collision_count"],
                "min_obstacle_dist": trial_out["min_obstacle_dist"],
            })
            
            # Save candidate animations
            if trial_out["success"] and successful_trial_data is None and p_name == "P3 (Full LCGC)":
                successful_trial_data = trial_out
            if not trial_out["success"] and failed_trial_data is None and p_name == "P1 (Single-Frame Raw)":
                failed_trial_data = trial_out
                
        results[p_name] = p_results
        
    return results, successful_trial_data, failed_trial_data, arena_ref


def main():
    print("=" * 70)
    print("  E5: End-to-End Navigation Simulation (G3, G5)")
    print("=" * 70)
    
    models = load_or_train_models()
    n_trials = config.NAV_TRIALS  # 100 trials per pipeline
    
    results, success_data, fail_data, arena = run_navigation_evaluation(models, n_trials=n_trials)
    
    # ── Calculate Summary & Wilson CIs ────────────────────────────────
    summary = {}
    print("\n--- Summary Results ---")
    
    for p_name, p_trials in results.items():
        n_succ = sum(1 for t in p_trials if t["success"])
        total = len(p_trials)
        p_hat, (ci_lo, ci_hi) = wilson_score_interval(n_succ, total)
        
        succ_steps = [t["steps"] for t in p_trials if t["success"]]
        mean_steps = float(np.mean(succ_steps)) if len(succ_steps) > 0 else float(np.mean([t["steps"] for t in p_trials]))
        mean_time = mean_steps * config.SIM_DT
        
        all_colls = [t["collisions"] for t in p_trials]
        total_colls = sum(all_colls)
        mean_colls = float(np.mean(all_colls))
        
        summary[p_name] = {
            "successes": n_succ,
            "total_trials": total,
            "success_rate": float(p_hat),
            "wilson_ci_95": [float(ci_lo), float(ci_hi)],
            "mean_steps": mean_steps,
            "mean_time_sec": mean_time,
            "total_collisions": total_colls,
            "mean_collisions_per_trial": mean_colls,
        }
        
        print(f"\n{p_name}:")
        print(f"  Success Rate : {p_hat*100.0:.1f}% (95% CI: [{ci_lo*100.0:.1f}%, {ci_hi*100.0:.1f}%])")
        print(f"  Mean Time    : {mean_time:.2f} s ({mean_steps:.1f} steps)")
        print(f"  Collisions   : {total_colls} total ({mean_colls:.2f} / trial)")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "e5_navigation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "total_trials", "success_rate_pct", "wilson_ci_lower_pct",
                     "wilson_ci_upper_pct", "mean_time_sec", "mean_collisions_per_trial", "total_collisions"])
        for p_name, d in summary.items():
            w.writerow([
                p_name, d["total_trials"], f"{d['success_rate']*100.0:.1f}",
                f"{d['wilson_ci_95'][0]*100.0:.1f}", f"{d['wilson_ci_95'][1]*100.0:.1f}",
                f"{d['mean_time_sec']:.2f}", f"{d['mean_collisions_per_trial']:.2f}",
                d["total_collisions"],
            ])
    print(f"\nSaved: {csv_path}")
    
    with open(os.path.join(RESULTS_DIR, "e5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Render Animated GIFs ──────────────────────────────────────────
    if success_data is not None:
        succ_gif = os.path.join(RESULTS_DIR, "e5_successful_trial.gif")
        print(f"\nRendering successful trial animation -> {succ_gif}...")
        save_animation(success_data, arena, succ_gif, fps=12, frame_skip=3)
        
    if fail_data is not None:
        fail_gif = os.path.join(RESULTS_DIR, "e5_failed_trial.gif")
        print(f"Rendering failed trial animation -> {fail_gif}...")
        save_animation(fail_data, arena, fail_gif, fps=12, frame_skip=3)

    # ── Plot Comparative Bar Charts ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    p_labels = ["P1\n(Single-Frame)", "P2\n(LCGC No Gate)", "P3\n(Full LCGC)"]
    names = list(summary.keys())
    
    # 1. Success Rate with Wilson CIs
    succ_rates = [summary[n]["success_rate"] * 100.0 for n in names]
    yerr_lo = [max(0.0, succ_rates[i] - summary[names[i]]["wilson_ci_95"][0] * 100.0) for i in range(3)]
    yerr_hi = [max(0.0, summary[names[i]]["wilson_ci_95"][1] * 100.0 - succ_rates[i]) for i in range(3)]
    bars1 = axes[0].bar(p_labels, succ_rates, yerr=[yerr_lo, yerr_hi], capsize=5, color=["#C44E52", "#EAA944", "#55A868"])
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Navigation Success Rate\n(Wilson 95% Confidence Interval)")
    axes[0].grid(axis="y", alpha=0.3)
    for b in bars1:
        h = b.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h + 3),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # 2. Completion Time
    times = [summary[n]["mean_time_sec"] for n in names]
    bars2 = axes[1].bar(p_labels, times, color="#4C72B0")
    axes[1].set_ylabel("Mean Completion Time (s)")
    axes[1].set_title("Navigation Time\n(Successful Trials)")
    axes[1].grid(axis="y", alpha=0.3)
    for b in bars2:
        h = b.get_height()
        axes[1].annotate(f"{h:.1f}s", xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=9)

    # 3. Collision Count
    colls = [summary[n]["mean_collisions_per_trial"] for n in names]
    bars3 = axes[2].bar(p_labels, colls, color="#DD8452")
    axes[2].set_ylabel("Mean Collisions per Trial")
    axes[2].set_title("Obstacle Collisions\n(Lower is Better)")
    axes[2].grid(axis="y", alpha=0.3)
    for b in bars3:
        h = b.get_height()
        axes[2].annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=9)

    plt.suptitle("E5: Autonomous Gesture Navigation Benchmark\n(100 Trials per Pipeline, 3 Operators, SIMULATED)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "e5_navigation_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")
    
    print("\n" + "=" * 70)
    print("  E5 Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
