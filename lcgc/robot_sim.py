"""
LCGC Robot Simulation — 2-D differential-drive robot with arena,
obstacles, ultrasonic sensor, simulated operator, and animation.

All physical parameters are imported from ``config.py``.
"""

import math
import numpy as np

import config
from synthetic_hands import GESTURE_POSES, _rotate_hand

# ──────────────────────────────────────────────────────────────────────
#  Default arena layout
# ──────────────────────────────────────────────────────────────────────
DEFAULT_OBSTACLES = [
    {"x": 1.5, "y": 1.5, "w": 0.5, "h": 0.5},
    {"x": 2.5, "y": 2.8, "w": 0.4, "h": 0.6},
    {"x": 1.0, "y": 3.0, "w": 0.6, "h": 0.4},
    {"x": 3.0, "y": 1.2, "w": 0.5, "h": 0.5},
]
DEFAULT_START = (0.5, 0.5)
DEFAULT_GOAL = (3.5, 3.5)


# ======================================================================
#  Arena
# ======================================================================
class Arena:
    """Rectangular arena with axis-aligned box obstacles.

    Parameters
    ----------
    width, height : float
        Arena dimensions (metres).  Default from config.
    obstacles : list[dict]
        Each dict has keys ``x, y, w, h`` giving the centre and size of
        a rectangular obstacle.
    goal : tuple[float, float]
        Goal position ``(gx, gy)``.
    """

    def __init__(
        self,
        width: float = config.ARENA_WIDTH,
        height: float = config.ARENA_HEIGHT,
        obstacles: list | None = None,
        goal: tuple = DEFAULT_GOAL,
    ):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else list(DEFAULT_OBSTACLES)
        self.goal = np.array(goal, dtype=np.float64)

        # Wall line-segments (closed rectangle, counter-clockwise)
        self.walls = [
            ((0, 0), (width, 0)),       # bottom
            ((width, 0), (width, height)),  # right
            ((width, height), (0, height)),  # top
            ((0, height), (0, 0)),       # left
        ]

    # ------------------------------------------------------------------
    #  Collision detection
    # ------------------------------------------------------------------
    def _box_bounds(self, obs: dict):
        """Return (x_min, y_min, x_max, y_max) for an obstacle dict."""
        hw, hh = obs["w"] / 2.0, obs["h"] / 2.0
        return obs["x"] - hw, obs["y"] - hh, obs["x"] + hw, obs["y"] + hh

    def check_collision(self, x: float, y: float, radius: float) -> bool:
        """Return *True* if a circle at (x, y) with the given *radius*
        overlaps any wall or obstacle.
        """
        # Walls — distance from boundary
        if (x - radius < 0 or x + radius > self.width or
                y - radius < 0 or y + radius > self.height):
            return True

        # Box obstacles — circle vs AABB
        for obs in self.obstacles:
            xmin, ymin, xmax, ymax = self._box_bounds(obs)
            # Closest point on AABB to circle centre
            cx = max(xmin, min(x, xmax))
            cy = max(ymin, min(y, ymax))
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if dist_sq <= radius ** 2:
                return True
        return False

    # ------------------------------------------------------------------
    #  Ray-casting (ultrasonic beam)
    # ------------------------------------------------------------------
    @staticmethod
    def _ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by):
        """Return parametric *t* along ray for intersection with segment
        A→B, or ``None`` if no hit (t ≥ 0 and segment parameter in [0, 1]).
        """
        sx, sy = bx - ax, by - ay
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-12:
            return None
        t = ((ax - ox) * sy - (ay - oy) * sx) / denom
        u = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def _box_segments(self, obs: dict):
        """Yield 4 edge segments for a box obstacle."""
        xmin, ymin, xmax, ymax = self._box_bounds(obs)
        yield (xmin, ymin, xmax, ymin)  # bottom
        yield (xmax, ymin, xmax, ymax)  # right
        yield (xmax, ymax, xmin, ymax)  # top
        yield (xmin, ymax, xmin, ymin)  # left

    def ray_cast(self, x: float, y: float, theta: float,
                 max_range: float = config.ULTRASONIC_MAX_RANGE) -> float:
        """Cast a ray from ``(x, y)`` in direction *theta* and return the
        distance to the nearest wall or obstacle, capped at *max_range*.
        """
        dx = math.cos(theta)
        dy = math.sin(theta)
        t_min = max_range

        # Check walls
        for (ax, ay), (bx, by) in self.walls:
            t = self._ray_segment_intersect(x, y, dx, dy, ax, ay, bx, by)
            if t is not None and t < t_min:
                t_min = t

        # Check obstacle edges
        for obs in self.obstacles:
            for ax, ay, bx, by in self._box_segments(obs):
                t = self._ray_segment_intersect(x, y, dx, dy, ax, ay, bx, by)
                if t is not None and t < t_min:
                    t_min = t

        return t_min

    def obstacle_distance(self, x: float, y: float) -> float:
        """Minimum distance from point (x, y) to any obstacle surface."""
        min_d = float("inf")
        for obs in self.obstacles:
            xmin, ymin, xmax, ymax = self._box_bounds(obs)
            cx = max(xmin, min(x, xmax))
            cy = max(ymin, min(y, ymax))
            d = math.hypot(x - cx, y - cy)
            if d < min_d:
                min_d = d
        return min_d


# ======================================================================
#  Differential-Drive Robot
# ======================================================================
class DiffDriveRobot:
    """Simple kinematic 2-D diff-drive robot.

    State
    -----
    x, y   : position (metres)
    theta  : heading (radians, 0 = +x, π/2 = +y)

    Parameters come from ``config``: WHEEL_SEPARATION_L, ROBOT_RADIUS,
    ULTRASONIC_NOISE_STD.
    """

    def __init__(self, x: float = 0.5, y: float = 0.5, theta: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.radius = config.ROBOT_RADIUS
        self.wheel_sep = config.WHEEL_SEPARATION_L
        self.rng = rng if rng is not None else np.random.default_rng(config.SEED)

    def update(self, v: float, omega: float, dt: float = config.SIM_DT):
        """Advance pose by one time step using unicycle kinematics.

        x += v * cos(θ) * dt
        y += v * sin(θ) * dt
        θ += ω * dt
        """
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        # Keep θ in (-π, π]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def get_ultrasonic(self, arena: Arena) -> float:
        """Return forward-facing ultrasonic range with Gaussian noise."""
        true_range = arena.ray_cast(self.x, self.y, self.theta)
        noisy = true_range + self.rng.normal(0, config.ULTRASONIC_NOISE_STD)
        return max(0.0, min(noisy, config.ULTRASONIC_MAX_RANGE))


# ======================================================================
#  Simulated Operator
# ======================================================================
class SimulatedOperator:
    """Generates a sequence of ``(landmarks, palm_size, gesture_label)``
    tuples that steer the robot from *start* toward *goal*.

    Operator styles
    ---------------
    ``"normal"``     — balanced defaults.
    ``"aggressive"`` — faster reactions, fewer Idle pauses.
    ``"cautious"``   — more Idle pauses, slower reactions.
    ``"novice"``     — higher error rate, more hand-lost events.
    """

    _STYLE_PARAMS = {
        "normal":     {"error_prob": 0.05, "idle_prob": 0.10,
                       "lost_prob": 0.02, "reaction_delay": (1, 3)},
        "aggressive": {"error_prob": 0.03, "idle_prob": 0.04,
                       "lost_prob": 0.01, "reaction_delay": (0, 1)},
        "cautious":   {"error_prob": 0.04, "idle_prob": 0.20,
                       "lost_prob": 0.02, "reaction_delay": (2, 4)},
        "novice":     {"error_prob": 0.12, "idle_prob": 0.08,
                       "lost_prob": 0.06, "reaction_delay": (2, 5)},
    }

    # Canonical normalised landmark templates per gesture (21 × 2).
    # Built once and then perturbed per frame.
    _TEMPLATE_CACHE: dict | None = None

    def __init__(
        self,
        arena: Arena,
        start: tuple = DEFAULT_START,
        goal: tuple = DEFAULT_GOAL,
        style: str = "normal",
        rng: np.random.Generator | None = None,
    ):
        self.arena = arena
        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.style = style
        self.params = dict(self._STYLE_PARAMS.get(style, self._STYLE_PARAMS["normal"]))
        self.rng = rng if rng is not None else np.random.default_rng(config.SEED + 7)

        # Internal state for reaction delay
        self._pending_gesture: str | None = None
        self._delay_remaining = 0
        self._last_gesture = "Idle"
        self._lost_counter = 0  # frames remaining with hand lost

        # Operator route waypoints to bypass obstacles and reach goal
        self.waypoints = [
            np.array([1.5, 0.7], dtype=np.float64),
            np.array([3.0, 2.0], dtype=np.float64),
            self.goal,
        ]
        self._current_wp_idx = 0

    # ------------------------------------------------------------------
    #  Landmark template generation
    # ------------------------------------------------------------------
    @classmethod
    def _build_templates(cls) -> dict:
        """Return a dict mapping gesture name → (21, 2) template array.

        Templates are normalised (wrist at origin, scaled by
        ||p9 − p0||), matching the pipeline convention.
        """
        if cls._TEMPLATE_CACHE is not None:
            return cls._TEMPLATE_CACHE

        # A plausible resting hand (Idle / open palm facing camera)
        base = np.array([
            [0.00,  0.00],   # 0  wrist
            [0.18, -0.06],   # 1  thumb_cmc
            [0.30, -0.12],   # 2  thumb_mcp
            [0.40, -0.16],   # 3  thumb_ip
            [0.48, -0.20],   # 4  thumb_tip
            [0.22,  0.04],   # 5  index_mcp
            [0.36,  0.06],   # 6  index_pip
            [0.46,  0.08],   # 7  index_dip
            [0.52,  0.10],   # 8  index_tip
            [0.22,  0.14],   # 9  middle_mcp  (scale ref)
            [0.36,  0.16],   # 10 middle_pip
            [0.46,  0.18],   # 11 middle_dip
            [0.52,  0.20],   # 12 middle_tip
            [0.20,  0.24],   # 13 ring_mcp
            [0.32,  0.26],   # 14 ring_pip
            [0.40,  0.28],   # 15 ring_dip
            [0.46,  0.30],   # 16 ring_tip
            [0.16,  0.32],   # 17 pinky_mcp
            [0.24,  0.34],   # 18 pinky_pip
            [0.30,  0.36],   # 19 pinky_dip
            [0.34,  0.38],   # 20 pinky_tip
        ], dtype=np.float64)

        # Normalise so wrist=origin, scale=||p9-p0||
        scale = np.linalg.norm(base[config.MIDDLE_BASE_IDX] - base[config.WRIST_IDX])
        base = (base - base[config.WRIST_IDX]) / scale

        templates = {}
        # Idle — relaxed open hand
        templates["Idle"] = base.copy()

        # Stop — open palm, fingers spread (same as base but slightly wider)
        stop = base.copy()
        stop[config.THUMB_IDXS] *= np.array([1.0, 1.3])
        stop[config.PINKY_IDXS] *= np.array([1.0, 1.3])
        templates["Stop"] = stop

        # Forward — index + middle extended, others curled
        fwd = base.copy()
        for idx_list in [config.RING_IDXS, config.PINKY_IDXS, config.THUMB_IDXS]:
            for i in idx_list:
                fwd[i] = base[config.WRIST_IDX] + 0.15 * (base[i] - base[config.WRIST_IDX])
        templates["Forward"] = fwd

        # Backward — fist (all fingers curled)
        bwd = base.copy()
        for idx_list in [config.THUMB_IDXS, config.INDEX_IDXS, config.MIDDLE_IDXS,
                         config.RING_IDXS, config.PINKY_IDXS]:
            for i in idx_list:
                bwd[i] = base[config.WRIST_IDX] + 0.12 * (base[i] - base[config.WRIST_IDX])
        templates["Backward"] = bwd

        # Left — index extended pointing left (negative y direction)
        left = base.copy()
        for idx_list in [config.MIDDLE_IDXS, config.RING_IDXS,
                         config.PINKY_IDXS, config.THUMB_IDXS]:
            for i in idx_list:
                left[i] = base[config.WRIST_IDX] + 0.15 * (base[i] - base[config.WRIST_IDX])
        # Rotate index finger to point left
        rot_angle = math.radians(40)
        cos_r, sin_r = math.cos(rot_angle), math.sin(rot_angle)
        for i in config.INDEX_IDXS:
            ox, oy = left[i]
            left[i] = [ox * cos_r - oy * sin_r, ox * sin_r + oy * cos_r]
        templates["Left"] = left

        # Right — mirror of Left
        right = left.copy()
        right[:, 1] = -right[:, 1]  # mirror y
        templates["Right"] = right

        cls._TEMPLATE_CACHE = templates
        return templates

    # ------------------------------------------------------------------
    #  Decide which gesture to issue
    # ------------------------------------------------------------------
    def _desired_gesture(self, rx: float, ry: float, rtheta: float) -> str:
        """Operator guides the robot along waypoints toward the goal."""
        # Check current target waypoint
        target = self.waypoints[self._current_wp_idx]
        dx = target[0] - rx
        dy = target[1] - ry
        dist = math.hypot(dx, dy)

        # Reached current waypoint?
        if dist < 0.35 and self._current_wp_idx < len(self.waypoints) - 1:
            self._current_wp_idx += 1
            target = self.waypoints[self._current_wp_idx]
            dx = target[0] - rx
            dy = target[1] - ry
            dist = math.hypot(dx, dy)

        # Reached final goal?
        dist_to_goal = math.hypot(self.goal[0] - rx, self.goal[1] - ry)
        if dist_to_goal < config.GOAL_RADIUS:
            return "Stop"

        angle_to_target = math.atan2(dy, dx)
        angle_err = (angle_to_target - rtheta + math.pi) % (2 * math.pi) - math.pi

        # Align heading towards waypoint
        if abs(angle_err) > math.radians(18):
            return "Left" if angle_err > 0 else "Right"

        return "Forward"

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------
    def get_gesture(self, robot_x: float, robot_y: float,
                    robot_theta: float, step: int):
        """Return ``(raw_landmarks_21x2 | None, palm_size_fraction, gesture_label)``.

        The landmarks are pre-normalisation 21 × 2 arrays in [0, 1]² as
        if coming from a camera frame.  ``None`` simulates hand leaving
        the frame.
        """
        # --- hand-lost simulation ---
        if self._lost_counter > 0:
            self._lost_counter -= 1
            return None, 0.0, self._last_gesture
        if self.rng.random() < self.params["lost_prob"]:
            self._lost_counter = self.rng.integers(2, 8)
            return None, 0.0, self._last_gesture

        # --- decide gesture with reaction delay ---
        desired = self._desired_gesture(robot_x, robot_y, robot_theta)

        if desired != self._last_gesture:
            if self._pending_gesture is None:
                lo, hi = self.params["reaction_delay"]
                self._delay_remaining = int(self.rng.integers(lo, hi + 1))
                self._pending_gesture = desired
            if self._delay_remaining > 0:
                self._delay_remaining -= 1
                desired = self._last_gesture  # keep old gesture
            else:
                desired = self._pending_gesture
                self._pending_gesture = None

        # --- occasional wrong gesture (5 % default) ---
        if self.rng.random() < self.params["error_prob"]:
            wrong_choices = [g for g in config.GESTURE_NAMES if g != desired]
            desired = self.rng.choice(wrong_choices)

        # --- random Idle hesitation ---
        if desired != "Stop" and self.rng.random() < self.params["idle_prob"]:
            desired = "Idle"

        self._last_gesture = desired

        # --- build landmarks ---
        lm = GESTURE_POSES[desired].copy()

        # Add per-frame noise
        lm += self.rng.normal(0, config.LANDMARK_NOISE_STD, lm.shape)

        # Random slight rotation (subject variation)
        rot_deg = self.rng.normal(0, config.ROTATION_RANGE / 3.0)
        rot = math.radians(rot_deg)
        c, s = math.cos(rot), math.sin(rot)
        rot_mat = np.array([[c, -s], [s, c]])
        lm = lm @ rot_mat.T

        # Per-landmark dropout
        dropout_mask = self.rng.random(config.NUM_LANDMARKS) < config.LANDMARK_DROPOUT_PROB
        for i in range(config.NUM_LANDMARKS):
            if dropout_mask[i]:
                lm[i] = lm[max(0, i - 1)]

        # Map to camera-like [0, 1] range
        # Place "hand" comfortably in the camera field of view
        palm_size = self.rng.uniform(0.20, 0.28)
        cam_lm = lm * palm_size + np.array([0.5, 0.5])

        return cam_lm, palm_size, desired


# ======================================================================
#  Simulation
# ======================================================================
class Simulation:
    """Combines :class:`Arena`, :class:`DiffDriveRobot`,
    :class:`SimulatedOperator`, and a gesture pipeline to run a
    complete navigation trial.

    Parameters
    ----------
    arena : Arena
    start, goal : tuple[float, float]
    operator_style : str
    pipeline : callable or object with ``process_frame`` method.
        If *None*, a trivial always-Forward pipeline is used.
    seed : int
    """

    def __init__(
        self,
        arena: Arena | None = None,
        start: tuple = DEFAULT_START,
        goal: tuple = DEFAULT_GOAL,
        operator_style: str = "normal",
        pipeline=None,
        seed: int = config.SEED,
    ):
        self.rng = np.random.default_rng(seed)
        self.arena = arena if arena is not None else Arena(goal=goal)
        self.start = start
        self.goal = goal
        self.robot = DiffDriveRobot(
            x=start[0], y=start[1], theta=math.atan2(goal[1] - start[1],
                                                       goal[0] - start[0]),
            rng=np.random.default_rng(seed + 1),
        )
        self.operator = SimulatedOperator(
            self.arena, start=start, goal=goal, style=operator_style,
            rng=np.random.default_rng(seed + 2),
        )
        self.pipeline = pipeline  # may be None → use dummy

    # ------------------------------------------------------------------
    #  Gesture → (v, omega)  using config command mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _gesture_to_command(gesture: str, palm_frac: float):
        """Map a gesture label and palm-size fraction to ``(v, ω)``."""
        # Speed scaling via palm size
        if palm_frac <= 0.0:
            speed_scale = config.PALM_GAMMA
        else:
            t = np.clip((palm_frac - config.S_MIN) / (config.S_MAX - config.S_MIN),
                        0.0, 1.0)
            speed_scale = config.PALM_GAMMA + (1.0 - config.PALM_GAMMA) * t

        v, omega = 0.0, 0.0
        if gesture == "Forward":
            v = config.V_MAX * speed_scale
        elif gesture == "Backward":
            v = -config.V_MAX * speed_scale
        elif gesture == "Left":
            omega = config.OMEGA_MAX * speed_scale
        elif gesture == "Right":
            omega = -config.OMEGA_MAX * speed_scale
        # "Idle" / "Stop" → both zero
        return v, omega

    # ------------------------------------------------------------------
    #  Run one trial
    # ------------------------------------------------------------------
    def run_trial(self, max_steps: int = config.NAV_MAX_STEPS) -> dict:
        """Execute a full navigation trial and return result dict.

        Returns
        -------
        dict with keys: success, collision_count, steps, trajectory,
        commands, gestures, min_obstacle_dist, ultrasonic_dists.
        """
        trajectory = [(self.robot.x, self.robot.y, self.robot.theta)]
        commands = []
        gestures = []
        collisions = 0
        min_obs_dist = float("inf")
        ultrasonic_dists = []

        for step in range(max_steps):
            # 1. Operator generates raw landmarks + label
            raw_lm, palm_frac, true_label = self.operator.get_gesture(
                self.robot.x, self.robot.y, self.robot.theta, step)

            # Get ultrasonic distance
            us_dist = self.robot.get_ultrasonic(self.arena)
            ultrasonic_dists.append(us_dist)

            # 2. Pipeline processes (or use dummy)
            if self.pipeline is not None and hasattr(self.pipeline, "step"):
                (v, omega), gesture, raw_p, sm_p = self.pipeline.step(
                    raw_lm, palm_size_frac=palm_frac, ultrasonic_dist=us_dist, dt=config.SIM_DT
                )
            elif self.pipeline is not None and hasattr(self.pipeline, "process_frame"):
                gesture = self.pipeline.process_frame(raw_lm, palm_frac)
                if us_dist < config.ULTRASONIC_STOP_DIST and gesture == "Forward":
                    gesture = "Stop"
                v, omega = self._gesture_to_command(gesture, palm_frac)
            else:
                # Dummy pipeline — just use true label
                gesture = true_label if raw_lm is not None else "Idle"
                if us_dist < config.ULTRASONIC_STOP_DIST and gesture == "Forward":
                    gesture = "Stop"
                v, omega = self._gesture_to_command(gesture, palm_frac)

            gestures.append(gesture)
            commands.append((v, omega))

            # 5. Move robot
            self.robot.update(v, omega)

            # 6. Collision detection & correction
            if self.arena.check_collision(self.robot.x, self.robot.y,
                                          self.robot.radius):
                collisions += 1
                # Undo move
                self.robot.x = trajectory[-1][0]
                self.robot.y = trajectory[-1][1]
                self.robot.theta = trajectory[-1][2]

            trajectory.append((self.robot.x, self.robot.y, self.robot.theta))

            # Track minimum obstacle distance
            obs_d = self.arena.obstacle_distance(self.robot.x, self.robot.y)
            if obs_d < min_obs_dist:
                min_obs_dist = obs_d

            # Goal check
            dist_to_goal = math.hypot(self.robot.x - self.goal[0],
                                      self.robot.y - self.goal[1])
            if dist_to_goal < config.GOAL_RADIUS:
                return {
                    "success": True,
                    "collision_count": collisions,
                    "steps": step + 1,
                    "trajectory": trajectory,
                    "commands": commands,
                    "gestures": gestures,
                    "min_obstacle_dist": min_obs_dist,
                    "ultrasonic_dists": ultrasonic_dists,
                }

        return {
            "success": False,
            "collision_count": collisions,
            "steps": max_steps,
            "trajectory": trajectory,
            "commands": commands,
            "gestures": gestures,
            "min_obstacle_dist": min_obs_dist,
            "ultrasonic_dists": ultrasonic_dists,
        }


# ======================================================================
#  Animation (GIF via matplotlib + Pillow writer)
# ======================================================================
def save_animation(trial_data: dict, arena: Arena, filepath: str,
                   fps: int = 10, frame_skip: int = 3):
    """Render the trial trajectory as an animated GIF.

    Parameters
    ----------
    trial_data : dict
        Output of :meth:`Simulation.run_trial`.
    arena : Arena
        The arena used for the trial.
    filepath : str
        Output ``.gif`` path.
    fps : int
        Playback frames-per-second.
    frame_skip : int
        Render every *n*-th simulation frame (keeps file size small).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation

    traj = trial_data["trajectory"]
    cmds = trial_data["commands"]
    gest = trial_data["gestures"]
    us_dists = trial_data["ultrasonic_dists"]

    # Indices to render (every frame_skip-th frame)
    render_idxs = list(range(0, len(cmds), frame_skip))
    if render_idxs[-1] != len(cmds) - 1:
        render_idxs.append(len(cmds) - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.1, arena.width + 0.1)
    ax.set_ylim(-0.1, arena.height + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Draw arena boundary
    boundary = plt.Rectangle((0, 0), arena.width, arena.height,
                              linewidth=2, edgecolor="black",
                              facecolor="white")
    ax.add_patch(boundary)

    # Draw obstacles (gray boxes)
    for obs in arena.obstacles:
        rect = patches.Rectangle(
            (obs["x"] - obs["w"] / 2, obs["y"] - obs["h"] / 2),
            obs["w"], obs["h"],
            linewidth=1, edgecolor="dimgray", facecolor="lightgray",
        )
        ax.add_patch(rect)

    # Goal marker
    goal_circle = plt.Circle(arena.goal, config.GOAL_RADIUS, color="limegreen",
                              alpha=0.5, label="Goal")
    ax.add_patch(goal_circle)
    ax.plot(*arena.goal, "g*", markersize=14, zorder=5)

    # Dynamic artists
    trail_line, = ax.plot([], [], "b-", linewidth=1.0, alpha=0.5)
    robot_body = plt.Circle((0, 0), config.ROBOT_RADIUS, color="royalblue",
                             alpha=0.8, zorder=10)
    ax.add_patch(robot_body)
    heading_line, = ax.plot([], [], "w-", linewidth=2, zorder=11)
    us_beam_line, = ax.plot([], [], "r-", linewidth=0.8, alpha=0.4)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=9, verticalalignment="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="wheat", alpha=0.8))

    def init():
        trail_line.set_data([], [])
        heading_line.set_data([], [])
        us_beam_line.set_data([], [])
        info_text.set_text("")
        return trail_line, robot_body, heading_line, us_beam_line, info_text

    def update(frame_num):
        idx = render_idxs[frame_num]
        # Trail up to current step
        xs = [t[0] for t in traj[: idx + 2]]
        ys = [t[1] for t in traj[: idx + 2]]
        trail_line.set_data(xs, ys)

        # Robot pose
        rx, ry, rtheta = traj[idx + 1]
        robot_body.center = (rx, ry)
        hx = rx + config.ROBOT_RADIUS * math.cos(rtheta)
        hy = ry + config.ROBOT_RADIUS * math.sin(rtheta)
        heading_line.set_data([rx, hx], [ry, hy])

        # Ultrasonic beam
        us_d = us_dists[idx] if idx < len(us_dists) else 0
        bx = rx + us_d * math.cos(rtheta)
        by = ry + us_d * math.sin(rtheta)
        us_beam_line.set_data([rx, bx], [ry, by])

        # Info overlay
        g = gest[idx] if idx < len(gest) else ""
        v, omega = cmds[idx] if idx < len(cmds) else (0, 0)
        info_text.set_text(
            f"Step {idx:>4d}  Gesture: {g:<10s}\n"
            f"v={v:+.3f} m/s  ω={omega:+.3f} rad/s\n"
            f"US={us_d:.2f} m"
        )
        return trail_line, robot_body, heading_line, us_beam_line, info_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(render_idxs), interval=1000 // fps,
                         blit=True)
    anim.save(filepath, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Animation saved → {filepath}")


# ======================================================================
#  Main — quick smoke test
# ======================================================================
if __name__ == "__main__":
    np.random.seed(config.SEED)

    print("=" * 60)
    print("LCGC Robot Simulator — smoke test")
    print("=" * 60)

    # 1. Build arena
    arena = Arena(obstacles=DEFAULT_OBSTACLES, goal=DEFAULT_GOAL)
    print(f"Arena : {arena.width} × {arena.height} m, "
          f"{len(arena.obstacles)} obstacles")
    print(f"Start : {DEFAULT_START}")
    print(f"Goal  : {DEFAULT_GOAL}")

    # 2. Run trial with dummy pipeline (always uses true label)
    sim = Simulation(arena=arena, start=DEFAULT_START, goal=DEFAULT_GOAL,
                     operator_style="normal", pipeline=None, seed=config.SEED)
    result = sim.run_trial(max_steps=config.NAV_MAX_STEPS)

    print(f"\n--- Trial result ---")
    print(f"  Success        : {result['success']}")
    print(f"  Steps          : {result['steps']}")
    print(f"  Collisions     : {result['collision_count']}")
    print(f"  Min obs. dist  : {result['min_obstacle_dist']:.4f} m")

    # 3. Save test animation
    gif_path = "test_trial.gif"
    print(f"\nRendering animation ({gif_path}) …")
    save_animation(result, arena, gif_path, fps=10, frame_skip=3)

    print("\nDone.")
