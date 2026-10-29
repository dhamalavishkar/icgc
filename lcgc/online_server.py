"""
online_server.py — Real-time HTTP & WebSocket Simulation Server for LCGC.

Bridges the web-based simulation interface with the trained INT8 TFLite model
and physical differential-drive robot kinematics.

Usage:
  python online_server.py [--port 8080] [--tflite results/B3_LCGC_int8.tflite]
"""

import os
import sys
import json
import math
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from pipeline import GesturePipeline
from robot_sim import Arena, DiffDriveRobot, SimulatedOperator, DEFAULT_OBSTACLES, DEFAULT_START, DEFAULT_GOAL


class SimulationState:
    """Maintains state of the live online simulation."""
    def __init__(self, tflite_path=None):
        if tflite_path is None:
            tflite_path = os.path.join(BASE_DIR, "results", "B3_LCGC_int8.tflite")
        
        use_tflite = os.path.exists(tflite_path)
        if use_tflite:
            self.pipeline = GesturePipeline(model=None, mode="full", use_tflite=True, tflite_path=tflite_path)
        else:
            keras_path = os.path.join(BASE_DIR, "results", "b3_lcgc.keras")
            import tensorflow as tf
            model = tf.keras.models.load_model(keras_path)
            self.pipeline = GesturePipeline(model=model, mode="full", use_tflite=False)

        self.tflite_path = tflite_path
        self.arena = Arena(config.ARENA_WIDTH, config.ARENA_HEIGHT, DEFAULT_OBSTACLES, DEFAULT_GOAL)
        self.robot = DiffDriveRobot(DEFAULT_START[0], DEFAULT_START[1], 0.0)
        self.operator = SimulatedOperator(self.arena, DEFAULT_START, DEFAULT_GOAL, style="cautious")
        
        self.step_count = 0
        self.collisions = 0
        self.trajectory = [(float(self.robot.x), float(self.robot.y))]
        self.auto_mode = False
        self.selected_gesture = "Idle"
        self.hand_present = True
        self.palm_size = 0.20
        self.current_cmd = (0.0, 0.0)
        self.current_probs = [1.0/6.0] * 6
        self.sm_probs = [1.0/6.0] * 6
        self.current_gesture = "Idle"
        self.safety_reason = "Normal"

    def reset(self):
        self.pipeline.reset()
        self.robot.x = float(DEFAULT_START[0])
        self.robot.y = float(DEFAULT_START[1])
        self.robot.theta = 0.0
        self.step_count = 0
        self.collisions = 0
        self.trajectory = [(float(self.robot.x), float(self.robot.y))]
        self.current_cmd = (0.0, 0.0)
        self.current_gesture = "Idle"
        self.safety_reason = "Normal"

    def set_pipeline_mode(self, mode):
        if mode in ("raw", "ema", "ema_hysteresis", "full"):
            self.pipeline.mode = mode

    def step(self, custom_landmarks=None, custom_palm=None, custom_dist=None):
        self.step_count += 1
        dt = 1.0 / config.FPS

        # 1. Ultrasonic sensing
        if custom_dist is not None:
            dist = float(custom_dist)
        else:
            dist = float(self.robot.get_ultrasonic(self.arena))

        # 2. Hand landmarks & gesture input
        palm = float(custom_palm if custom_palm is not None else self.palm_size)
        if self.auto_mode:
            lm, auto_palm, auto_name = self.operator.get_gesture(
                self.robot.x, self.robot.y, self.robot.theta, self.step_count
            )
            palm = auto_palm
        elif custom_landmarks is not None:
            lm = np.array(custom_landmarks, dtype=np.float32).reshape(21, 2)
        elif self.hand_present:
            from synthetic_hands import GESTURE_POSES
            base_pose = GESTURE_POSES.get(self.selected_gesture, GESTURE_POSES["Idle"])
            noise = np.random.normal(0, 0.015, base_pose.shape)
            lm = base_pose + noise
        else:
            lm = None

        # 3. Pipeline step
        (v, omega), g_name, raw_p, sm_p = self.pipeline.step(
            raw_landmarks_21x2_or_none=lm,
            palm_size_frac=palm,
            ultrasonic_dist=dist,
            dt=dt
        )

        self.current_cmd = (float(v), float(omega))
        self.current_gesture = g_name
        self.current_probs = [float(p) for p in raw_p] if raw_p is not None else [0.0]*6
        self.sm_probs = [float(p) for p in sm_p] if sm_p is not None else [0.0]*6

        # Safety state diagnosis
        if dist < config.ULTRASONIC_STOP_DIST:
            self.safety_reason = f"Ultrasonic Stop (< {config.ULTRASONIC_STOP_DIST*100:.0f}cm)"
        elif self.pipeline._hand_lost_elapsed > config.HAND_LOST_TIMEOUT:
            self.safety_reason = f"Hand Lost Watchdog (> {config.HAND_LOST_TIMEOUT:.1f}s)"
        else:
            self.safety_reason = "Normal"

        # 4. Robot kinematics update
        self.robot.update(v, omega, dt)
        collided = self.arena.check_collision(self.robot.x, self.robot.y, self.robot.radius)
        if collided:
            self.collisions += 1

        self.trajectory.append((float(self.robot.x), float(self.robot.y)))
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)

        goal_dist = math.hypot(self.robot.x - self.arena.goal[0], self.robot.y - self.arena.goal[1])
        reached_goal = goal_dist < 0.25

        # Format 21 landmarks for client visualizer
        lm_list = lm.tolist() if lm is not None else None

        return {
            "step": self.step_count,
            "robot": {
                "x": float(self.robot.x),
                "y": float(self.robot.y),
                "theta": float(self.robot.theta),
                "v": float(v),
                "omega": float(omega),
                "v_left": float(v - omega * config.WHEEL_SEPARATION_L / 2.0),
                "v_right": float(v + omega * config.WHEEL_SEPARATION_L / 2.0),
                "ultrasonic_dist": float(dist),
                "reached_goal": reached_goal,
                "goal_dist": float(goal_dist),
                "collisions": self.collisions,
            },
            "pipeline": {
                "mode": self.pipeline.mode,
                "gesture": self.current_gesture,
                "raw_probs": self.current_probs,
                "smoothed_probs": self.sm_probs,
                "classes": config.GESTURE_NAMES,
                "safety_active": (v == 0.0 and omega == 0.0 and self.current_gesture in ("Forward", "Backward", "Left", "Right")),
                "safety_reason": self.safety_reason,
                "palm_size": palm,
            },
            "landmarks": lm_list,
            "trajectory": self.trajectory[-100:],
            "arena": {
                "width": self.arena.width,
                "height": self.arena.height,
                "obstacles": self.arena.obstacles,
                "goal": self.arena.goal.tolist(),
            }
        }


# Global simulation instance
SIM = None


class LCGCRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web interface and REST API endpoints."""

    def __init__(self, *args, **kwargs):
        web_dir = os.path.join(BASE_DIR, "web_sim")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            data = SIM.step()
            self._send_json(data)
        elif self.path == "/api/reset":
            SIM.reset()
            self._send_json({"status": "reset_ok"})
        elif self.path == "/api/config":
            self._send_json({
                "fps": config.FPS,
                "classes": config.GESTURE_NAMES,
                "v_max": config.V_MAX,
                "omega_max": config.OMEGA_MAX,
                "ultrasonic_stop": config.ULTRASONIC_STOP_DIST,
                "hand_lost_timeout": config.HAND_LOST_TIMEOUT,
            })
        else:
            super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        if self.path == "/api/step":
            lm = req.get("landmarks")
            palm = req.get("palm_size")
            dist = req.get("ultrasonic_dist")
            data = SIM.step(custom_landmarks=lm, custom_palm=palm, custom_dist=dist)
            self._send_json(data)
        elif self.path == "/api/control":
            if "gesture" in req:
                SIM.selected_gesture = req["gesture"]
            if "auto_mode" in req:
                SIM.auto_mode = bool(req["auto_mode"])
            if "operator_style" in req:
                SIM.operator.style = req["operator_style"]
            if "hand_present" in req:
                SIM.hand_present = bool(req["hand_present"])
            if "palm_size" in req:
                SIM.palm_size = float(req["palm_size"])
            if "mode" in req:
                SIM.set_pipeline_mode(req["mode"])
            self._send_json({"status": "control_updated"})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Suppress routine log flooding
        pass


def run_server(port=8080, tflite_path=None):
    global SIM
    SIM = SimulationState(tflite_path)
    server_address = ("", port)
    httpd = HTTPServer(server_address, LCGCRequestHandler)
    print("=" * 70)
    print(f"  LCGC Online Simulation Server Started")
    print(f"  Local Web UI  : http://localhost:{port}")
    print(f"  Pipeline Model: {SIM.tflite_path}")
    print(f"  Target Rate   : {config.FPS} FPS (33.3 ms/step)")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCGC Online Simulation Server")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--tflite", type=str, default=None, help="Path to B3 TFLite model")
    args = parser.parse_args()
    run_server(args.port, args.tflite)
