"""
run_all.py — Master runner for the complete LCGC simulation and evaluation suite.

Executes all benchmarks in sequence:
  1. E1: Leave-One-Subject-Out Recognition & Model Edge Complexity
  2. E2: Continuous-Tracking Ablation (Raw vs EMA vs Hysteresis)
  3. E3: Robustness Sweeps & Invariance Verification
  4. E4: Safety Gate & Watchdog Evaluation
  5. E5: Autonomous Navigation Simulation (100 trials x 3 pipelines)
  6. E6: Real-Time Latency Budget Breakdown
  7. Final Report Generation (report_tables.md)

Usage:
  python run_all.py
"""

import os
import sys
import subprocess
import time
import argparse

PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(step_name, script_path):
    print("\n" + "=" * 75)
    print(f"  [RUNNING] {step_name}")
    print(f"  Script  : {script_path}")
    print("=" * 75)
    t0 = time.time()
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    res = subprocess.run([PYTHON_EXE, script_path], cwd=BASE_DIR, env=env)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}.")
        return False
    else:
        print(f"\n[COMPLETED] {step_name} in {elapsed:.1f} s")
        return True


def main():
    parser = argparse.ArgumentParser(description="LCGC Master Experiment Runner")
    parser.add_argument("--skip-e1", action="store_true", help="Skip re-training E1 models if results already exist")
    args = parser.parse_args()

    print("=" * 75)
    print("  LCGC Complete Simulation & Evaluation Suite")
    print("=" * 75)
    total_t0 = time.time()

    experiments = []
    if not (args.skip_e1 and os.path.exists(os.path.join(BASE_DIR, "results", "e1_summary.json"))):
        experiments.append(("E1: Recognition & Complexity", os.path.join(BASE_DIR, "experiments", "e1_recognition.py")))
    else:
        print("\n[INFO] Skipping E1 (reusing trained models & metrics in results/).")

    experiments.extend([
        ("E2: Continuous-Tracking Ablation", os.path.join(BASE_DIR, "experiments", "e2_ablation.py")),
        ("E3: Robustness & Invariance", os.path.join(BASE_DIR, "experiments", "e3_robustness.py")),
        ("E4: Safety Gate Verification", os.path.join(BASE_DIR, "experiments", "e4_safety.py")),
        ("E5: Autonomous Navigation Benchmark", os.path.join(BASE_DIR, "experiments", "e5_navigation.py")),
        ("E6: Real-Time Latency Budget", os.path.join(BASE_DIR, "experiments", "e6_latency.py")),
        ("Report Synthesis", os.path.join(BASE_DIR, "generate_report.py")),
    ])

    for name, path in experiments:
        success = run_step(name, path)
        if not success:
            print(f"\nStopping run due to error in {name}.")
            sys.exit(1)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 75)
    print(f"  ALL EXPERIMENTS & REPORT GENERATION FINISHED in {total_elapsed/60.0:.1f} min.")
    print("=" * 75)


if __name__ == "__main__":
    main()
