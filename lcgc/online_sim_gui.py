"""
online_sim_gui.py — Desktop Interactive Real-Time Simulator for LCGC.

Runs a live 30 FPS interactive window using Tkinter and Matplotlib.
Allows manual gesture triggering, keyboard control (W/A/S/D/Space),
autonomous operator replay, real-time landmark visualization,
probability tracking, and ultrasonic safety gate triggering.

Usage:
  python online_sim_gui.py
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from pipeline import GesturePipeline
from robot_sim import Arena, DiffDriveRobot, SimulatedOperator, DEFAULT_OBSTACLES, DEFAULT_START, DEFAULT_GOAL
from synthetic_hands import GESTURE_POSES, CANONICAL_HAND

# Hand bone connections for 21 MediaPipe landmarks
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base
]


class OnlineSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LCGC — Lightweight Continuous Gesture Control (Online Simulation)")
        self.root.geometry("1200x820")
        self.root.minsize(1050, 720)

        # Simulation Backend State
        tflite_path = os.path.join(BASE_DIR, "results", "B3_LCGC_int8.tflite")
        if os.path.exists(tflite_path):
            self.pipeline = GesturePipeline(model=None, mode="full", use_tflite=True, tflite_path=tflite_path)
        else:
            keras_path = os.path.join(BASE_DIR, "results", "b3_lcgc.keras")
            import tensorflow as tf
            model = tf.keras.models.load_model(keras_path)
            self.pipeline = GesturePipeline(model=model, mode="full", use_tflite=False)
        self.arena = Arena(config.ARENA_WIDTH, config.ARENA_HEIGHT, DEFAULT_OBSTACLES, DEFAULT_GOAL)
        self.robot = DiffDriveRobot(DEFAULT_START[0], DEFAULT_START[1], 0.0)
        self.operator = SimulatedOperator(self.arena, DEFAULT_START, DEFAULT_GOAL, style="cautious")

        self.step_count = 0
        self.collisions = 0
        self.trajectory = [(float(self.robot.x), float(self.robot.y))]
        self.is_running = True
        self.auto_mode = False
        self.hand_present = True
        self.selected_gesture = "Idle"
        self.palm_size = 0.20
        self.operator_style = "cautious"

        self.current_cmd = (0.0, 0.0)
        self.current_gesture = "Idle"
        self.raw_probs = np.ones(6) / 6.0
        self.sm_probs = np.ones(6) / 6.0
        self.current_landmarks = GESTURE_POSES["Idle"].copy()
        self.ultrasonic_dist = 1.5
        self.safety_status_text = "Gate: CLEAR (Safe)"

        self._build_ui()
        self._bind_keys()
        self._schedule_loop()

    def _build_ui(self):
        # Master container: Left for Visualizations, Right for Controls & Telemetry
        main_frame = ttk.Frame(self.root, padding=6)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        right_frame = ttk.Frame(main_frame, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_frame.pack_propagate(False)

        # ─── LEFT: MATPLOTLIB CANVASES ───
        # Create multi-axis figure: Arena (Top-Left), Hand Landmarks (Top-Right), Probabilities (Bottom)
        self.fig = plt.Figure(figsize=(8, 7.5), dpi=100)
        self.fig.patch.set_facecolor("#1e222d")

        # Arena Axis
        self.ax_arena = self.fig.add_subplot(2, 2, (1, 3))
        self.ax_arena.set_facecolor("#151821")

        # Hand Axis
        self.ax_hand = self.fig.add_subplot(2, 2, 2)
        self.ax_hand.set_facecolor("#151821")

        # Probabilities Axis
        self.ax_probs = self.fig.add_subplot(2, 2, 4)
        self.ax_probs.set_facecolor("#151821")

        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ─── RIGHT: CONTROL DASHBOARD ───
        # Title Card
        title_box = ttk.LabelFrame(right_frame, text="System Info", padding=8)
        title_box.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(title_box, text="LCGC Real-Time Gesture Control", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(title_box, text="Depthwise 1D-CNN (INT8 Quantized, 26.8 KB)", font=("Arial", 8)).pack(anchor=tk.W)
        ttk.Label(title_box, text="Sliding Window: 15 frames @ 30 FPS", font=("Arial", 8)).pack(anchor=tk.W)

        # Telemetry Card
        telem_box = ttk.LabelFrame(right_frame, text="Live Telemetry", padding=8)
        telem_box.pack(fill=tk.X, pady=(0, 6))

        self.lbl_gesture = ttk.Label(telem_box, text="Classified: Idle", font=("Arial", 10, "bold"), foreground="#4ade80")
        self.lbl_gesture.pack(anchor=tk.W)

        self.lbl_cmd = ttk.Label(telem_box, text="Command: v = 0.00 m/s, ω = 0.00 rad/s", font=("Consolas", 9))
        self.lbl_cmd.pack(anchor=tk.W)

        self.lbl_wheels = ttk.Label(telem_box, text="Diff-Drive: v_L = 0.00, v_R = 0.00 m/s", font=("Consolas", 8))
        self.lbl_wheels.pack(anchor=tk.W)

        self.lbl_ultra = ttk.Label(telem_box, text="Ultrasonic Sensor: 1.50 m", font=("Consolas", 9))
        self.lbl_ultra.pack(anchor=tk.W)

        self.lbl_safety = ttk.Label(telem_box, text="Safety Gate: OK", font=("Arial", 9, "bold"), foreground="#22c55e")
        self.lbl_safety.pack(anchor=tk.W, pady=(2, 0))

        self.lbl_stats = ttk.Label(telem_box, text="Steps: 0 | Collisions: 0 | Goal Dist: 4.24m", font=("Arial", 8))
        self.lbl_stats.pack(anchor=tk.W, pady=(4, 0))

        # Pipeline Mode Card
        mode_box = ttk.LabelFrame(right_frame, text="Pipeline Stage Mode", padding=8)
        mode_box.pack(fill=tk.X, pady=(0, 6))

        self.mode_var = tk.StringVar(value="full")
        modes = [
            ("Full LCGC (EMA + Hyst + Gate)", "full"),
            ("EMA + Hysteresis (No Gate)", "ema_hysteresis"),
            ("EMA Only (alpha=0.4)", "ema"),
            ("Raw Argmax Baseline", "raw"),
        ]
        for text, val in modes:
            ttk.Radiobutton(mode_box, text=text, value=val, variable=self.mode_var,
                            command=self._on_mode_change).pack(anchor=tk.W)

        # Gesture Controller Card
        ctrl_box = ttk.LabelFrame(right_frame, text="Manual Gesture Trigger", padding=8)
        ctrl_box.pack(fill=tk.X, pady=(0, 6))

        btn_grid = ttk.Frame(ctrl_box)
        btn_grid.pack(fill=tk.X)

        ttk.Button(btn_grid, text="Forward (W)", command=lambda: self._set_gesture("Forward")).grid(row=0, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(btn_grid, text="Left (A)", command=lambda: self._set_gesture("Left")).grid(row=1, column=0, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(btn_grid, text="STOP (Space)", command=lambda: self._set_gesture("Stop")).grid(row=1, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(btn_grid, text="Right (D)", command=lambda: self._set_gesture("Right")).grid(row=1, column=2, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(btn_grid, text="Backward (S)", command=lambda: self._set_gesture("Backward")).grid(row=2, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Button(btn_grid, text="Idle (I)", command=lambda: self._set_gesture("Idle")).grid(row=2, column=2, sticky=tk.EW, padx=2, pady=2)
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)
        btn_grid.columnconfigure(2, weight=1)

        # Palm Size Proportional Speed Slider
        palm_frame = ttk.Frame(ctrl_box)
        palm_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(palm_frame, text="Palm Scale (Speed):", font=("Arial", 8)).pack(side=tk.LEFT)
        self.slider_palm = ttk.Scale(palm_frame, from_=0.08, to=0.30, value=0.20, command=self._on_palm_change)
        self.slider_palm.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Autonomous Operator & Safety Testing Card
        auto_box = ttk.LabelFrame(right_frame, text="Autonomous Operator & Watchdog", padding=8)
        auto_box.pack(fill=tk.X, pady=(0, 6))

        self.auto_btn = ttk.Button(auto_box, text="Start Autopilot Operator", command=self._toggle_auto)
        self.auto_btn.pack(fill=tk.X, pady=(0, 4))

        style_frame = ttk.Frame(auto_box)
        style_frame.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(style_frame, text="Operator Style:", font=("Arial", 8)).pack(side=tk.LEFT)
        self.cmb_style = ttk.Combobox(style_frame, values=["cautious", "aggressive", "novice"], state="readonly", width=12)
        self.cmb_style.set("cautious")
        self.cmb_style.bind("<<ComboboxSelected>>", self._on_style_change)
        self.cmb_style.pack(side=tk.RIGHT)

        self.hand_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_box, text="Hand In Camera Frame (Watchdog)", variable=self.hand_var,
                        command=self._on_hand_toggle).pack(anchor=tk.W)

        # Simulation Run Controls Card
        sim_box = ttk.LabelFrame(right_frame, text="Simulation Controls", padding=8)
        sim_box.pack(fill=tk.X, pady=(0, 6))

        sim_btns = ttk.Frame(sim_box)
        sim_btns.pack(fill=tk.X)
        self.pause_btn = ttk.Button(sim_btns, text="Pause (P)", command=self._toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(sim_btns, text="Reset Robot (R)", command=self._reset_sim).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

    def _bind_keys(self):
        self.root.bind("<w>", lambda e: self._set_gesture("Forward"))
        self.root.bind("<s>", lambda e: self._set_gesture("Backward"))
        self.root.bind("<a>", lambda e: self._set_gesture("Left"))
        self.root.bind("<d>", lambda e: self._set_gesture("Right"))
        self.root.bind("<space>", lambda e: self._set_gesture("Stop"))
        self.root.bind("<i>", lambda e: self._set_gesture("Idle"))
        self.root.bind("<p>", lambda e: self._toggle_pause())
        self.root.bind("<r>", lambda e: self._reset_sim())

    def _set_gesture(self, gesture):
        self.auto_mode = False
        self.auto_btn.config(text="Start Autopilot Operator")
        self.selected_gesture = gesture

    def _toggle_auto(self):
        self.auto_mode = not self.auto_mode
        self.auto_btn.config(text="Stop Autopilot Operator" if self.auto_mode else "Start Autopilot Operator")

    def _toggle_pause(self):
        self.is_running = not self.is_running
        self.pause_btn.config(text="Resume (P)" if not self.is_running else "Pause (P)")

    def _on_mode_change(self):
        self.pipeline.mode = self.mode_var.get()

    def _on_style_change(self, event=None):
        self.operator.style = self.cmb_style.get()

    def _on_palm_change(self, val):
        self.palm_size = float(val)

    def _on_hand_toggle(self):
        self.hand_present = self.hand_var.get()

    def _reset_sim(self):
        self.pipeline.reset()
        self.robot.x = float(DEFAULT_START[0])
        self.robot.y = float(DEFAULT_START[1])
        self.robot.theta = 0.0
        self.step_count = 0
        self.collisions = 0
        self.trajectory = [(float(self.robot.x), float(self.robot.y))]
        self.current_cmd = (0.0, 0.0)
        self.selected_gesture = "Idle"

    def _schedule_loop(self):
        self._update_simulation()
        # 30 FPS target -> 33 ms
        self.root.after(33, self._schedule_loop)

    def _update_simulation(self):
        if self.is_running:
            self.step_count += 1
            dt = 1.0 / config.FPS

            # 1. Ultrasonic reading
            self.ultrasonic_dist = self.robot.get_ultrasonic(self.arena)

            # 2. Gesture input & landmarks
            if self.auto_mode:
                lm, auto_palm, auto_name = self.operator.get_gesture(
                    self.robot.x, self.robot.y, self.robot.theta, self.step_count
                )
                self.palm_size = auto_palm
                self.selected_gesture = auto_name
            elif self.hand_present:
                base_pose = GESTURE_POSES.get(self.selected_gesture, GESTURE_POSES["Idle"])
                noise = np.random.normal(0, 0.012, base_pose.shape)
                lm = base_pose + noise
            else:
                lm = None

            self.current_landmarks = lm

            # 3. Pipeline inference
            (v, omega), g_name, raw_p, sm_p = self.pipeline.step(
                raw_landmarks_21x2_or_none=lm,
                palm_size_frac=self.palm_size,
                ultrasonic_dist=self.ultrasonic_dist,
                dt=dt
            )
            self.current_cmd = (v, omega)
            self.current_gesture = g_name
            self.raw_probs = raw_p if raw_p is not None else np.zeros(6)
            self.sm_probs = sm_p if sm_p is not None else np.zeros(6)

            # 4. Robot motion update
            self.robot.update(v, omega, dt)
            collided = self.arena.check_collision(self.robot.x, self.robot.y, self.robot.radius)
            if collided:
                self.collisions += 1

            self.trajectory.append((float(self.robot.x), float(self.robot.y)))
            if len(self.trajectory) > 600:
                self.trajectory.pop(0)

            # Safety text status
            if self.ultrasonic_dist < config.ULTRASONIC_STOP_DIST:
                self.safety_status_text = f"SAFETY STOP: Obstacle in {self.ultrasonic_dist*100:.1f} cm (< 20cm)"
                self.lbl_safety.config(text=self.safety_status_text, foreground="#ef4444")
            elif self.pipeline._hand_lost_elapsed > config.HAND_LOST_TIMEOUT:
                self.safety_status_text = f"WATCHDOG: Hand lost for {self.pipeline._hand_lost_elapsed:.2f} s (> 0.5s)"
                self.lbl_safety.config(text=self.safety_status_text, foreground="#f59e0b")
            else:
                self.safety_status_text = "Safety Gate: CLEAR (Nominal)"
                self.lbl_safety.config(text=self.safety_status_text, foreground="#22c55e")

            # Update Labels
            self.lbl_gesture.config(text=f"Classified: {self.current_gesture}")
            self.lbl_cmd.config(text=f"Command: v = {v:+.2f} m/s, ω = {omega:+.2f} rad/s")
            v_l = v - omega * config.WHEEL_SEPARATION_L / 2.0
            v_r = v + omega * config.WHEEL_SEPARATION_L / 2.0
            self.lbl_wheels.config(text=f"Diff-Drive: v_L = {v_l:+.2f}, v_R = {v_r:+.2f} m/s")
            self.lbl_ultra.config(text=f"Ultrasonic Sensor: {self.ultrasonic_dist:.2f} m")

            goal_dist = math.hypot(self.robot.x - self.arena.goal[0], self.robot.y - self.arena.goal[1])
            self.lbl_stats.config(text=f"Steps: {self.step_count} | Collisions: {self.collisions} | Goal Dist: {goal_dist:.2f}m")

        # Redraw plots (every step or every 2nd step)
        if self.step_count % 2 == 0 or not self.is_running:
            self._render_plots()

    def _render_plots(self):
        # 1. Arena Plot
        self.ax_arena.clear()
        self.ax_arena.set_title("2D Arena Simulation (4m x 4m)", color="#e2e8f0", fontsize=9, fontweight="bold")
        self.ax_arena.set_xlim(-0.2, self.arena.width + 0.2)
        self.ax_arena.set_ylim(-0.2, self.arena.height + 0.2)
        self.ax_arena.set_aspect("equal")
        self.ax_arena.tick_params(colors="#94a3b8", labelsize=7)
        self.ax_arena.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)

        # Boundary Walls
        wall_rect = patches.Rectangle((0, 0), self.arena.width, self.arena.height,
                                      linewidth=2, edgecolor="#64748b", facecolor="none")
        self.ax_arena.add_patch(wall_rect)

        # Obstacles
        for obs in self.arena.obstacles:
            hw, hh = obs["w"] / 2.0, obs["h"] / 2.0
            rect = patches.Rectangle((obs["x"] - hw, obs["y"] - hh), obs["w"], obs["h"],
                                     facecolor="#475569", edgecolor="#94a3b8", linewidth=1.5)
            self.ax_arena.add_patch(rect)

        # Goal
        gx, gy = self.arena.goal
        self.ax_arena.plot(gx, gy, marker="*", markersize=14, color="#eab308", label="Goal")
        self.ax_arena.text(gx + 0.1, gy + 0.1, "GOAL", color="#eab308", fontsize=8, fontweight="bold")

        # Trajectory trail
        if len(self.trajectory) > 1:
            trajx, trajy = zip(*self.trajectory)
            self.ax_arena.plot(trajx, trajy, color="#38bdf8", linewidth=1.2, alpha=0.7)

        # Robot chassis
        rx, ry, rth = self.robot.x, self.robot.y, self.robot.theta
        robot_color = "#22c55e" if self.current_gesture in ("Forward", "Backward", "Left", "Right") else "#94a3b8"
        if self.ultrasonic_dist < config.ULTRASONIC_STOP_DIST:
            robot_color = "#ef4444"

        robot_circle = patches.Circle((rx, ry), config.ROBOT_RADIUS, facecolor=robot_color, edgecolor="#ffffff", linewidth=1.5, zorder=5)
        self.ax_arena.add_patch(robot_circle)

        # Heading vector
        hx = rx + config.ROBOT_RADIUS * 1.5 * math.cos(rth)
        hy = ry + config.ROBOT_RADIUS * 1.5 * math.sin(rth)
        self.ax_arena.plot([rx, hx], [ry, hy], color="#ffffff", linewidth=2, zorder=6)

        # Ultrasonic beam ray
        beam_dist = min(self.ultrasonic_dist, config.ULTRASONIC_MAX_RANGE)
        bx = rx + beam_dist * math.cos(rth)
        by = ry + beam_dist * math.sin(rth)
        beam_color = "#ef4444" if self.ultrasonic_dist < config.ULTRASONIC_STOP_DIST else "#38bdf8"
        self.ax_arena.plot([rx, bx], [ry, by], color=beam_color, linestyle=":", linewidth=1.5, zorder=4)
        self.ax_arena.plot(bx, by, marker="x", color=beam_color, markersize=6, zorder=4)

        # 2. Hand Landmark Plot
        self.ax_hand.clear()
        self.ax_hand.set_title("MediaPipe 21 Landmarks", color="#e2e8f0", fontsize=8, fontweight="bold")
        self.ax_hand.set_aspect("equal")
        self.ax_hand.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax_hand.set_xlim(-0.35, 0.35)
        self.ax_hand.set_ylim(-0.15, 0.75)

        if self.current_landmarks is not None:
            lm = self.current_landmarks
            # Draw bones
            for u, v in HAND_BONES:
                self.ax_hand.plot([lm[u, 0], lm[v, 0]], [lm[u, 1], lm[v, 1]],
                                  color="#38bdf8", linewidth=1.8, alpha=0.8)
            # Draw landmark dots
            self.ax_hand.scatter(lm[:, 0], lm[:, 1], color="#f43f5e", s=25, zorder=5)
            # Highlight wrist (0) and middle MCP (9) for normalization
            self.ax_hand.scatter([lm[0, 0]], [lm[0, 1]], color="#fbbf24", s=50, zorder=6, label="p0 (wrist)")
            self.ax_hand.scatter([lm[9, 0]], [lm[9, 1]], color="#34d399", s=50, zorder=6, label="p9 (scale)")
        else:
            self.ax_hand.text(0, 0.3, "HAND NOT DETECTED\n(Watchdog active)",
                              color="#ef4444", ha="center", va="center", fontsize=9, fontweight="bold")

        # 3. Probability Distribution Bar Plot
        self.ax_probs.clear()
        self.ax_probs.set_title("Class Probabilities (EMA)", color="#e2e8f0", fontsize=8, fontweight="bold")
        classes = config.GESTURE_NAMES
        y_pos = np.arange(len(classes))
        probs = self.sm_probs

        colors = ["#38bdf8" if c != self.current_gesture else "#22c55e" for c in classes]
        self.ax_probs.barh(y_pos, probs, align="center", color=colors, height=0.6, alpha=0.85)
        self.ax_probs.set_yticks(y_pos)
        self.ax_probs.set_yticklabels(classes, color="#e2e8f0", fontsize=7)
        self.ax_probs.set_xlim(0.0, 1.05)
        self.ax_probs.axvline(x=config.PROB_THRESHOLD, color="#ef4444", linestyle="--", linewidth=1, label="0.80 thresh")
        self.ax_probs.tick_params(colors="#94a3b8", labelsize=7)
        self.ax_probs.invert_yaxis()

        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    app = OnlineSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
