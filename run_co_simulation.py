import os
import json
from pathlib import Path
import time
import numpy as np
try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:   # give a clear message instead of a bare traceback
    raise ImportError(
        "PyBullet is not installed. Install it with:  pip install pybullet\n"
        "(On Windows you may need the 'Microsoft C++ Build Tools', or use: "
        "conda install -c conda-forge pybullet)") from exc

from controllers import (
    LinearPIDController,
    NonlinearPIDController,
    SlidingModeController,
    FuzzyController,
    ControllerPreset
)

# ML model must be kept beside this script (along with its JSON metadata).
# Fail explicitly rather than claiming ML control when a fallback is used.
class PyBulletMLGainScheduler:
    def __init__(self, controller_type):
        import joblib
        from ml_gain_tuner import feature_vector, FEATURE_NAMES, GAIN_NAMES
        self.feature_vector = feature_vector
        self.gain_names = GAIN_NAMES
        self.controller_name = {
            "Linear PID": "pid", "Nonlinear PID": "nonlinear_pid",
            "Sliding Mode": "smc", "Fuzzy": "fuzzy",
        }[controller_type]
        root = Path(__file__).resolve().parent
        meta = json.loads((root / "ml_gain_model.json").read_text(encoding="utf-8"))
        if meta.get("feature_names") != FEATURE_NAMES or meta.get("gain_names") != GAIN_NAMES:
            raise RuntimeError("ML model metadata does not match the expected feature/gain schema")
        self.model = joblib.load(root / "ml_gain_model.joblib")
        self.bounds = [(0.60, 1.20), (0.35, 1.00), (0.75, 1.40),
                       (0.60, 1.20), (0.75, 1.35), (1.0, 1.80), (0.60, 1.25)]
        self.scales = {name: 1.0 for name in GAIN_NAMES}

    def update(self, speed_error, grade_deg, roll_deg, pitch_deg, lateral_speed, accel):
        x = self.feature_vector(speed_error, grade_deg, 0.0, roll_deg,
                                pitch_deg, lateral_speed, accel, self.controller_name)
        pred = np.asarray(self.model.predict(x.reshape(1, -1)), dtype=float)
        if pred.shape != (1, len(self.gain_names)) or not np.all(np.isfinite(pred)):
            raise RuntimeError("ML model returned invalid gains")
        for name, val, (lo, hi) in zip(self.gain_names, pred[0], self.bounds):
            bounded = float(np.clip(val, lo, hi))
            self.scales[name] = 0.8 * self.scales[name] + 0.2 * bounded
        return self.scales


# ---- Test-track layout (all x in metres, PyBullet world frame) -------------------
# Every surface joins the next one with NO vertical step: the drive torque is
# limited to 75 Nm (~250 N at the rim), which cannot push the front casters up
# even a 2 cm vertical lip. Humps and ramps are therefore built as sloped slabs.
HUMP_CENTERS = (2.5, 3.8)
HUMP_HEIGHT = 0.024        # m
HUMP_RAMP_RUN = 0.20       # horizontal length of each hump flank (~7 deg)
HUMP_TOP_LEN = 0.20        # flat top of each hump
GRADE_DEG = 10.0
INCLINE_START, INCLINE_END = 4.2, 7.2
CREST_END = 7.5
DECLINE_END = CREST_END + (INCLINE_END - INCLINE_START)   # back to z = 0
CREST_Z = (INCLINE_END - INCLINE_START) * float(np.tan(np.radians(GRADE_DEG)))
TRACK_HALF_WIDTH = 2.5


def track_segments():
    """Top-surface polyline of the track as ((x0, z0), (x1, z1), thickness_half, rgba)."""
    yellow = [0.8, 0.6, 0.1, 1.0]
    segs = []
    for xc in HUMP_CENTERS:
        x0 = xc - HUMP_TOP_LEN / 2 - HUMP_RAMP_RUN
        x1 = xc - HUMP_TOP_LEN / 2
        x2 = xc + HUMP_TOP_LEN / 2
        x3 = x2 + HUMP_RAMP_RUN
        segs += [((x0, 0.0), (x1, HUMP_HEIGHT), 0.012, yellow),
                 ((x1, HUMP_HEIGHT), (x2, HUMP_HEIGHT), 0.012, yellow),
                 ((x2, HUMP_HEIGHT), (x3, 0.0), 0.012, yellow)]
    segs += [((INCLINE_START, 0.0), (INCLINE_END, CREST_Z), 0.04, [0.55, 0.55, 0.58, 1.0]),
             ((INCLINE_END, CREST_Z), (CREST_END, CREST_Z), 0.04, [0.5, 0.5, 0.52, 1.0]),
             ((CREST_END, CREST_Z), (DECLINE_END, 0.0), 0.04, [0.45, 0.5, 0.45, 1.0])]
    return segs


def _add_slab(p0, p1, half_t, rgba, friction):
    """Static box whose TOP face runs exactly from p0 to p1 (x, z)."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    length = float(np.hypot(dx, dz))
    phi = float(np.arctan2(dz, dx))                      # rise angle of the surface
    normal = np.array([-np.sin(phi), np.cos(phi)])       # unit normal (x, z), pointing up
    mid = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]) - normal * half_t
    half = [length / 2, TRACK_HALF_WIDTH, half_t]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                             basePosition=[float(mid[0]), 0.0, float(mid[1])],
                             baseOrientation=p.getQuaternionFromEuler([0, -phi, 0]))
    p.changeDynamics(body, -1, lateralFriction=friction, rollingFriction=0.01)
    return body


def _visual_box(position, half_extents, color, orientation=None):
    """Decorative, collision-free scenery; does not bias stability measurements."""
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                             basePosition=position,
                             baseOrientation=orientation or [0, 0, 0, 1])


def _surface_height_and_angle(x):
    """Piecewise exact track surface, used to attach paint to ramp faces."""
    for a, b, _, _ in track_segments():
        if a[0] <= x <= b[0]:
            t = (x - a[0]) / max(b[0] - a[0], 1e-9)
            return a[1] + t * (b[1] - a[1]), np.arctan2(b[1] - a[1], b[0] - a[0])
    return 0.0, 0.0


def add_track_scenery():
    """Realistic visual road treatment, shoulders, safety markers and distance ticks.

    All objects here are visual-only so the original wheelchair dynamics,
    contact surfaces, and controller comparisons remain reproducible.
    """
    # Asphalt-colored lane strip, broken center line, and white edge markings.
    # Markings float 3 mm above the physical track to avoid z-fighting.
    for x in np.arange(-1.0, DECLINE_END + 3.0, 0.25):
        z, angle = _surface_height_and_angle(float(x))
        rot = p.getQuaternionFromEuler([0, -float(angle), 0])
        for y in (-1.15, 1.15):
            _visual_box([float(x), y, z + 0.004], [0.122, 0.025, 0.0015],
                        [0.94, 0.94, 0.88, 1], rot)
        if int((x + 1.0) / 0.25) % 8 < 4:
            _visual_box([float(x), 0, z + 0.005], [0.12, 0.018, 0.001],
                        [0.94, 0.82, 0.28, 1], rot)

    # Alternating shoulder markers, with hazard colors near the incline.
    for x in np.arange(0.5, DECLINE_END + 2.0, 0.75):
        z, _ = _surface_height_and_angle(float(x))
        for y in (-2.25, 2.25):
            _visual_box([float(x), y, z + 0.045], [0.15, 0.08, 0.045],
                        [0.96, 0.52, 0.08, 1] if x >= INCLINE_START else [0.88, 0.88, 0.8, 1])

    # Survey-style transverse markings make slope and speed visually legible.
    for x in (0.0, 2.0, INCLINE_START, INCLINE_END, CREST_END, DECLINE_END):
        z, angle = _surface_height_and_angle(x + 0.01)
        rot = p.getQuaternionFromEuler([0, -float(angle), 0])
        _visual_box([x + 0.01, 0, z + 0.007], [0.035, 1.05, 0.002],
                    [0.96, 0.92, 0.73, 1], rot)

    # Roadside low-poly grass patches, beyond the driving corridor.
    rng = np.random.default_rng(17)
    for _ in range(135):
        x = float(rng.uniform(-3, DECLINE_END + 5))
        y = float(rng.choice([-1, 1]) * rng.uniform(3.1, 10.0))
        radius = float(rng.uniform(0.12, 0.55))
        shade = float(rng.uniform(0.19, 0.35))
        _visual_box([x, y, 0.009], [radius, radius * 0.7, 0.008],
                    [shade * 0.65, shade, shade * 0.45, 1])


def create_uneven_terrain_track():
    """
    Creates an uneven terrain track for wheelchair stability testing:
    Flat Start -> Speed Humps -> 10-deg Incline -> Crest -> 10-deg Decline -> Flat Landing.
    All surfaces blend into each other (no vertical lips), see track_segments().
    """
    planeId = p.loadURDF("plane.urdf", [0, 0, 0])
    p.changeDynamics(planeId, -1, lateralFriction=0.85, rollingFriction=0.01)
    p.changeVisualShape(planeId, -1, rgbaColor=[0.2, 0.2, 0.22, 1.0])
    for p0, p1, half_t, rgba in track_segments():
        _add_slab(p0, p1, half_t, rgba, friction=0.9)
    add_track_scenery()


def terrain_pitch_deg(x):
    """Expected chair pitch (deg) if it sits flat on the track at world x.

    PyBullet convention: incline (nose-up) = -10, decline (nose-down) = +10.
    Corners are blended over ~0.3 m so the reference does not jump.
    """
    knots_x = [0.0, INCLINE_START, INCLINE_START + 0.3, INCLINE_END - 0.3, INCLINE_END,
               CREST_END, CREST_END + 0.3, DECLINE_END - 0.3, DECLINE_END, 99.0]
    knots_p = [0.0, 0.0, -GRADE_DEG, -GRADE_DEG, 0.0, 0.0, GRADE_DEG, GRADE_DEG, 0.0, 0.0]
    return float(np.interp(x, knots_x, knots_p))


# HUD layout (world units at the HUD depth). If the text spills outside the panel,
# raise HUD_CHAR_W; if the panel is too wide, lower it.
HUD_LINE_H = 0.09
HUD_CHAR_W = 0.042
HUD_PITCH_CAUTION_DEG = 6.0
HUD_PITCH_ALERT_DEG = 12.0
HUD_ROLL_CAUTION_DEG = 4.0
HUD_ROLL_ALERT_DEG = 8.0
HUD_MAX_CHARS = 64
HUD_PAD = 0.07
HUD_PANEL_W = HUD_MAX_CHARS * HUD_CHAR_W + 2 * HUD_PAD
HUD_PANEL_H = 9 * HUD_LINE_H + 2 * HUD_PAD


def _rotmat_to_quat(right, fwd, up):
    """Quaternion (x,y,z,w) for the rotation whose columns are right, fwd, up."""
    m = np.column_stack([right, fwd, up])
    tr = np.trace(m)
    if tr > 0:
        sc = np.sqrt(tr + 1.0) * 2
        return [(m[2, 1] - m[1, 2]) / sc, (m[0, 2] - m[2, 0]) / sc, (m[1, 0] - m[0, 1]) / sc, 0.25 * sc]
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    sc = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * sc
    q[j] = (m[j, i] + m[i, j]) / sc
    q[k] = (m[k, i] + m[i, k]) / sc
    q[3] = (m[k, j] - m[j, k]) / sc
    return q


def create_hud_panel(half_w=HUD_PANEL_W / 2, half_h=HUD_PANEL_H / 2):
    """Semi-transparent dark panel (no collision) drawn behind the HUD text."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_w, 0.004, half_h],
                              rgbaColor=[0.02, 0.03, 0.08, 0.82])
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, -50])


def draw_live_stability_monitor(text_ids, pitch_deg, roll_deg, pitch_rate, speed, torque, terrain_state, cam_target, panel_id=None, slope_deg=0.0):
    """
    Overlays a dynamic live HUD panel pinned firmly to the top-left corner 
    of the PyBullet visualizer window.
    """
    if not p.isConnected():
        return
    # Judge stability by tilt RELATIVE to the surface. On a 10 deg ramp the chair
    # is *supposed* to be pitched 10 deg; only the extra tilt means it is tipping.
    rel_pitch = pitch_deg - slope_deg
    abs_pitch = abs(rel_pitch)
    abs_roll = abs(roll_deg)
    
    if abs_pitch > HUD_PITCH_ALERT_DEG or abs_roll > HUD_ROLL_ALERT_DEG:
        status_text = "Display tilt threshold exceeded"
        status_color = [1, 0, 0]  # Red
    elif abs_pitch > HUD_PITCH_CAUTION_DEG or abs_roll > HUD_ROLL_CAUTION_DEG:
        status_text = "Near display tilt threshold"
        status_color = [1, 0.7, 0]  # Yellow/Orange
    else:
        status_text = "Within display thresholds"
        status_color = [0, 1, 0]  # Green

    hud_lines = [
        f"--- LIVE STABILITY MONITOR ---",
        f"Status: {status_text}",
        f"Terrain: {terrain_state}",
        f"Pitch: {pitch_deg:+.1f}° Slope: {slope_deg:+.1f}° Rel: {rel_pitch:+.1f}°",
        f"Roll Angle : {roll_deg:+.2f}° (display ±{HUD_ROLL_ALERT_DEG:g}°)",
        f"Display thresholds are not validated safety limits",
        f"Pitch Rate : {pitch_rate:.2f}°/s",
        f"Speed      : {speed:.2f} m/s",
        f"Torque     : {torque:+.2f} Nm"
    ]

    # Anchor the HUD to the camera's top-left screen corner (camera-relative),
    # so it stays in the corner and never overlaps the wheelchair.
    # Change HUD_CORNER to "top-left", "top-right", "bottom-left" or "bottom-right".
    HUD_CORNER = "top-left"
    cam = p.getDebugVisualizerCamera()
    width, height = cam[0], cam[1]
    fwd = np.array(cam[5], dtype=float)
    right = np.array(cam[6], dtype=float); right /= np.linalg.norm(right)
    up = np.array(cam[7], dtype=float); up /= np.linalg.norm(up)
    dist = cam[10]
    cam_pos = np.array(cam[11], dtype=float) - fwd * dist

    depth = dist * 0.9
    half_h = depth * np.tan(np.radians(30.0))   # default PyBullet fov = 60 deg
    half_w = half_h * (width / float(height))
    n_lines = len(hud_lines)
    block_h = HUD_PANEL_H
    margin_x, margin_y = 0.06 * half_w, 0.06 * half_h
    sx = -1 if "left" in HUD_CORNER else 1
    sy = 1 if "top" in HUD_CORNER else -1
    # text starts at its left edge; keep right-side corners inside the view
    x_off = -half_w + margin_x + HUD_PAD if sx < 0 else half_w - margin_x - HUD_PANEL_W + HUD_PAD
    y_off = (half_h - margin_y - HUD_PAD) if sy > 0 else (-half_h + margin_y + block_h - HUD_PAD)
    anchor = cam_pos + fwd * depth + right * x_off + up * y_off

    # Dark backing panel (fixed size, camera-facing) so text is readable on any background
    if panel_id is not None:
        p_center = (anchor
                    + right * (HUD_PANEL_W / 2 - HUD_PAD)
                    + up * (HUD_LINE_H * 0.7 + HUD_PAD - HUD_PANEL_H / 2)
                    + fwd * 0.03)            # slightly behind the text
        p.resetBasePositionAndOrientation(panel_id, p_center.tolist(),
                                          _rotmat_to_quat(right, fwd, up))

    for i, line in enumerate(hud_lines):
        line_pos = (anchor - up * (i * HUD_LINE_H)).tolist()
        color = status_color if i == 1 else ([1, 1, 1] if i == 0 else [0.85, 0.95, 1])
        text_scale = 1.15 if i in [0, 1] else 0.95
        
        if text_ids[i] is None:
            text_ids[i] = p.addUserDebugText(line, line_pos, textColorRGB=color, textSize=text_scale)
        else:
            text_ids[i] = p.addUserDebugText(
                line, 
                line_pos, 
                textColorRGB=color, 
                textSize=text_scale, 
                replaceItemUniqueId=text_ids[i]
            )


_ACTIVE_PHYSICS_CLIENT = None


def run_wheelchair_stability_analyzer(
    controller_type="Nonlinear PID",
    urdf_path=None,
    user_speed=1.0,
    sim_time=30.0,
    headless=False,
    output_dir="output/graphs",
    ml_adaptive=True,
):
    """Run PyBullet and guarantee client cleanup if setup or stepping fails."""
    global _ACTIVE_PHYSICS_CLIENT
    _ACTIVE_PHYSICS_CLIENT = None
    try:
        return _run_wheelchair_stability_analyzer(
            controller_type=controller_type, urdf_path=urdf_path,
            user_speed=user_speed, sim_time=sim_time, headless=headless,
            output_dir=output_dir, ml_adaptive=ml_adaptive)
    finally:
        client = _ACTIVE_PHYSICS_CLIENT
        if client is not None and p.isConnected(client):
            p.disconnect(client)
        _ACTIVE_PHYSICS_CLIENT = None


def _run_wheelchair_stability_analyzer(
    controller_type="Nonlinear PID", 
    urdf_path=None,
    user_speed=1.0,
    sim_time=30.0,
    headless=False,
    output_dir="output/graphs",
    ml_adaptive=True,
):
    """Run the PyBullet co-simulation.

    headless=True uses PyBullet DIRECT mode (no window, no real-time sleep) and
    still saves the stability graph - useful for a quick check or a remote PC.
    """
    if urdf_path is None:   # always find the URDF next to this file, whatever the cwd
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pybullet_wheelchair.urdf")
    if not np.isfinite(user_speed) or not 0.0 <= user_speed <= 1.2:
        raise ValueError("user_speed must be finite and between 0 and 1.2 m/s")
    if not np.isfinite(sim_time) or sim_time <= 0:
        raise ValueError("sim_time must be a positive finite number")
    if controller_type not in ("Linear PID", "Nonlinear PID", "Sliding Mode", "Fuzzy"):
        raise ValueError(f"Unknown controller '{controller_type}'. Choose Linear PID, Nonlinear PID, Sliding Mode, or Fuzzy.")
    # Use a non-interactive plotting backend to avoid Tkinter cleanup on worker threads.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    global _ACTIVE_PHYSICS_CLIENT
    physicsClient = p.connect(p.DIRECT if headless else p.GUI)
    if physicsClient < 0:
        raise RuntimeError("PyBullet could not connect to a physics server")
    _ACTIVE_PHYSICS_CLIENT = physicsClient
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    p.setPhysicsEngineParameter(numSolverIterations=50)
    if not headless:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)   # HUD panel must not cast a shadow
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    create_uneven_terrain_track()

    # Rear axle height equals the 0.30 m drive-wheel radius. The base origin
    # is the assumed occupied-body reference; see URDF inertial CoG offset.
    startPos = [0, 0, 0.30]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF(urdf_path, startPos, startOrientation)

    joint_indices = {}
    for i in range(p.getNumJoints(robotId)):
        info = p.getJointInfo(robotId, i)
        joint_indices[info[1].decode('utf-8')] = i

    left_joint = joint_indices["left_wheel_joint"]
    right_joint = joint_indices["right_wheel_joint"]

    p.setJointMotorControl2(robotId, left_joint, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robotId, right_joint, p.VELOCITY_CONTROL, force=0)

    # Caster swivel joints are passive and free to align with travel.
    if "caster_swivel_left_joint" in joint_indices:
        p.setJointMotorControl2(robotId, joint_indices["caster_swivel_left_joint"], p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robotId, joint_indices["caster_swivel_right_joint"], p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robotId, joint_indices["front_caster_left_joint"], p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(robotId, joint_indices["front_caster_right_joint"], p.VELOCITY_CONTROL, force=0)

    # Real total mass of the model (used for the gravity feedforward)
    total_mass = sum(p.getDynamicsInfo(robotId, i)[0] for i in range(-1, p.getNumJoints(robotId)))

    scheduler = PyBulletMLGainScheduler(controller_type) if ml_adaptive else None
    gain_log = []
    ml_state_log = []   # ML inputs at each gain update (for the CSV)
    preset = ControllerPreset(max_speed=1.2, max_torque=75.0, max_accel=0.6)
    
    if controller_type == "Linear PID":
        controller = LinearPIDController(kp=110.0, ki=10.0, kd=4.0, preset=preset)
    elif controller_type == "Nonlinear PID":
        controller = NonlinearPIDController(kp=120.0, ki=14.0, kd=5.0, alpha=1.2, preset=preset)
    elif controller_type == "Sliding Mode":
        controller = SlidingModeController(lambda_param=4.0, K_gain=40.0, preset=preset)
    elif controller_type == "Fuzzy":
        controller = FuzzyController(error_range=0.6, output_range=60.0, ki=10.0, preset=preset)
    else:
        raise ValueError(f"Unknown controller '{controller_type}'. Use 'Linear PID', 'Nonlinear PID', 'Sliding Mode' or 'Fuzzy'.")

    dt = 1.0 / 240.0
    p.setTimeStep(dt)

    time_log = []
    pitch_log, roll_log, slope_log, rel_pitch_log = [], [], [], []
    pitch_rate_log, roll_rate_log = [], []
    velocity_log, target_log, torque_log = [], [], []
    pos_log, vel_xyz_log, terrain_z_log = [], [], []   # for the summary graph
    
    tau_filtered = 0.0
    decelerating = False
    hud_text_ids = [None] * 9
    hud_panel_id = None if headless else create_hud_panel()

    # --- Stability tuning ---
    PITCH_FILTER = 0.10        # low-pass on pitch used for slope logic (~0.04 s)
    SLEW_NM_PER_S = 400.0      # max torque change rate (limits jerk -> less pitching)
    DECLINE_SPEED = 0.4        # target speed on a decline (m/s)
    SLOW_LOOKAHEAD_M = 0.8     # start slowing this far BEFORE the decline begins
    YAW_KP = 60.0              # heading hold: Nm of wheel-torque difference per rad of yaw error
    YAW_KD = 40.0              # ... per rad/s of yaw rate (damps the side-to-side wobble)
    YAW_DIFF_MAX = 10.0        # Nm cap on the steering torque difference
    YAW_REF = 0.0              # keep driving straight along +x
    pitch_f = 0.0              # filtered pitch (deg)
    v_ref = 0.0                # rate-limited speed reference
    tau_cmd = 0.0              # slew-limited torque
    camera_target = np.array([0.0, 0.0, 0.40])

    print(f"\n========================================================")
    print(f"   WHEELCHAIR STABILITY ANALYZER [{controller_type}]   ")
    print(f"   ML adaptive gains: {'ON (trained model)' if scheduler else 'OFF (fixed gains)'}")
    print(f"========================================================\n")

    steps = int(sim_time / dt)
    
    try:
        for step in range(steps):
            if not p.isConnected(physicsClient):
                print("PyBullet window closed; ending the live run.")
                break
            t = step * dt

            pos, orientation = p.getBasePositionAndOrientation(robotId)
            lin_vel, ang_vel = p.getBaseVelocity(robotId)
        
            rpy = p.getEulerFromQuaternion(orientation)
            roll_deg = np.degrees(rpy[0])
            pitch_deg = np.degrees(rpy[1])
            yaw_rad = rpy[2]

            rot_matrix = p.getMatrixFromQuaternion(orientation)
            forward_vec = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
            current_v = np.dot(np.array(lin_vel), forward_vec)

            # Smooth Camera Tracking
            desired_camera_target = np.array([pos[0] + 0.2, pos[1], pos[2] + 0.1])
            camera_target += 0.08 * (desired_camera_target - camera_target)
            cam_target = camera_target.tolist()
            if not headless and step % 4 == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.2,
                    cameraYaw=np.degrees(yaw_rad) - 50.0,
                    cameraPitch=-20.0,
                    cameraTargetPosition=cam_target
                )

            # Terrain State Regions
            if pos[0] < 2.0:
                terrain_state = "Flat Ground Start"
            elif 2.0 <= pos[0] < 4.5:
                terrain_state = "Speed Humps Zone"
            elif 4.5 <= pos[0] < 7.0:
                terrain_state = "10° Incline Climb"
            elif 7.0 <= pos[0] < 7.8:
                terrain_state = "Crest Platform"
            elif 7.8 <= pos[0] < 11.2:
                terrain_state = "10° Decline (Braking)"
            else:
                terrain_state = "Ground Landing / Controlled Stop"

            slope_ref = terrain_pitch_deg(pos[0])
            rel_pitch = pitch_deg - slope_ref

            # The test-track geometry is known exactly. Filter its terrain angle
            # for feedforward; body pitch is used only for the ride-attitude report.
            pitch_f += PITCH_FILTER * (slope_ref - pitch_f)

            # Slope-aware speed target: blend smoothly from user_speed to DECLINE_SPEED
            # between 1 deg and 4 deg of downhill pitch (no hard switch -> no chatter).
            slope_ahead = terrain_pitch_deg(pos[0] + SLOW_LOOKAHEAD_M)   # look ahead: already slow at the crest
            k = np.clip((max(pitch_f, slope_ahead) - 1.0) / 3.0, 0.0, 1.0)
            k = k * k * (3.0 - 2.0 * k)
            v_slope = user_speed + (DECLINE_SPEED - user_speed) * k

            if pos[0] >= 11.2:
                if not decelerating:
                    print(f"[INFO] Cleared decline ramp at X = {pos[0]:.2f}m. Bringing wheelchair to a complete stop...")
                    decelerating = True
                v_cmd = 0.0
            elif t >= 1.0:
                v_cmd = v_slope
            else:
                v_cmd = 0.0

            # Rate-limit the reference (uses the preset's max_accel)
            dv = preset.max_accel * dt
            v_ref += np.clip(v_cmd - v_ref, -dv, dv)

            # Update trained ML gains at 10 Hz from live PyBullet state.
            # Reuse the last prediction between updates to avoid model overhead.
            if scheduler is not None:
                if step == 0:
                    previous_speed = current_v
                if step % 24 == 0:
                    lateral_vec = np.array([-forward_vec[1], forward_vec[0], 0.0])
                    lateral_speed = float(np.dot(np.asarray(lin_vel), lateral_vec))
                    accel = (current_v - previous_speed) / (24 * dt) if step else 0.0
                    scales = scheduler.update(v_ref - current_v, -pitch_f,
                                              roll_deg, pitch_deg, lateral_speed, accel)
                    gain_log.append((t, *[scales[k] for k in scheduler.gain_names]))
                    ml_state_log.append((pos[0], current_v, -pitch_f, roll_deg, pitch_deg, lateral_speed, accel))
                    previous_speed = current_v
                    if controller_type in ("Linear PID", "Nonlinear PID"):
                        base = (110.0, 10.0, 4.0) if controller_type == "Linear PID" else (120.0, 14.0, 5.0)
                        controller.kp = base[0] * scales["pid_kp_scale"]
                        controller.ki = base[1] * scales["pid_ki_scale"]
                        controller.kd = base[2] * scales["pid_kd_scale"]
                    elif controller_type == "Fuzzy":
                        controller.output_scale = scales["fuzzy_output_scale"]
                    else:
                        controller.lambda_param = 4.0 * scales["smc_lambda_scale"]
                        controller.K_gain = 40.0 * scales["smc_k_scale"]
                        controller.phi = 0.1 * scales["smc_phi_scale"]

            # Calculate Control Torque
            if controller_type == "Nonlinear PID":
                tau_raw = controller.compute(v_ref, current_v, dt, pitch_rad=np.radians(pitch_f), m_eq=total_mass, R_rr=12.0)
            elif controller_type in ("Sliding Mode", "Fuzzy"):
                tau_raw = controller.compute(v_ref, current_v, dt, pitch_rad=np.radians(pitch_f), m_eq=total_mass, R_rr=12.0)
            else:
                tau_raw = controller.compute(v_ref, current_v, dt)

            # Slew-rate limit, then smooth
            max_step = SLEW_NM_PER_S * dt
            tau_cmd += np.clip(tau_raw - tau_cmd, -max_step, max_step)
            tau_filtered = 0.85 * tau_filtered + 0.15 * tau_cmd
        
            # Heading hold: positive yaw = turning left (+y). Speeding up the right wheel
            # and easing the left one turns the chair left, and vice versa.
            yaw_err = (YAW_REF - yaw_rad + np.pi) % (2 * np.pi) - np.pi
            d = np.clip(YAW_KP * yaw_err - YAW_KD * ang_vel[2], -YAW_DIFF_MAX, YAW_DIFF_MAX)
            p.setJointMotorControl2(robotId, left_joint, p.TORQUE_CONTROL, force=-(tau_filtered / 2.0 - d))
            p.setJointMotorControl2(robotId, right_joint, p.TORQUE_CONTROL, force=-(tau_filtered / 2.0 + d))

            # Update Top-Left HUD Monitor
            if not headless and step % 4 == 0:
                draw_live_stability_monitor(
                    hud_text_ids, 
                    pitch_deg, 
                    roll_deg, 
                    np.degrees(ang_vel[1]), 
                    current_v, 
                    tau_filtered, 
                    terrain_state,
                    cam_target,
                    hud_panel_id,
                    slope_ref
                )

            p.stepSimulation()
            if not headless:
                time.sleep(dt)

            # (total ground contact force is derived in the plot as m*(a - g))
            pos_log.append(pos); vel_xyz_log.append(lin_vel)
            terrain_z_log.append(_surface_height_and_angle(float(pos[0]))[0])

            time_log.append(t)
            pitch_log.append(pitch_deg)
            roll_log.append(roll_deg)
            slope_log.append(slope_ref)
            rel_pitch_log.append(rel_pitch)
            pitch_rate_log.append(np.degrees(ang_vel[1]))
            roll_rate_log.append(np.degrees(ang_vel[0]))
            velocity_log.append(current_v)
            target_log.append(v_ref)
            torque_log.append(tau_filtered)

            # Stop simulation when fully stopped on flat ground past the ramp
            if decelerating and current_v < 0.05 and pos[0] >= 11.4:
                print(f"[SUCCESS] Wheelchair landed smoothly and came to a stop at X = {pos[0]:.2f}m!")
                if not headless:
                    time.sleep(1.0)
                break
    except p.error as exc:   # window closed mid-run: keep what was logged so far
        print(f"PyBullet connection lost ({exc}); ending the live run and saving results so far.")

    if p.isConnected(physicsClient):
        p.disconnect(physicsClient)

    if not time_log:
        print("No simulation samples were collected; the PyBullet window closed before the first step.")
        return {"time": [], "pitch": [], "roll": [], "velocity": [], "target": [], "torque": []}

    # Generate Report & Plots
    max_pitch = np.max(np.abs(rel_pitch_log))   # tilt relative to the surface
    max_abs_pitch = np.max(np.abs(pitch_log))
    max_roll = np.max(np.abs(roll_log))
    max_pitch_rate = np.max(np.abs(pitch_rate_log))
    
    print("\n--- STABILITY ANALYSIS PERFORMANCE ---")
    print(f"Max Pitch vs Surface:     {max_pitch:.2f}° (display threshold: {HUD_PITCH_ALERT_DEG:g}°)")
    print(f"Max Absolute Pitch:       {max_abs_pitch:.2f}° (10° ramps give about 10° by design)")
    print(f"Max Roll Tilt Angle:      {max_roll:.2f}° (display threshold: {HUD_ROLL_ALERT_DEG:g}°)")
    print("Tilt thresholds are visualization aids, not validated safety limits.")
    print(f"Max Pitch Angular Rate:   {max_pitch_rate:.2f}°/s (Lower = Smoother Ride)")
    print("---------------------------------------\n")

    fig = Figure(figsize=(12, 8))
    FigureCanvasAgg(fig)
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time_log, pitch_log, 'r-', label='Pitch (world)')
    ax1.plot(time_log, slope_log, 'k:', label='Terrain slope')
    ax1.plot(time_log, rel_pitch_log, 'm-', linewidth=2, label='Pitch relative to surface')
    ax1.plot(time_log, roll_log, 'b-', label='Roll Angle (Side Tilt)')
    ax1.axhline(y=HUD_PITCH_ALERT_DEG, color='r', linestyle='--', alpha=0.5, label='Display threshold')
    ax1.axhline(y=-HUD_PITCH_ALERT_DEG, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Angle (deg)')
    ax1.set_title(f'Terrain Traversal Diagnostic ({controller_type})')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time_log, pitch_rate_log, 'm-', label='Pitch Rate (deg/s)')
    ax2.set_ylabel('Angular Rate (deg/s)')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time_log, velocity_log, 'g-', label='Actual Speed (m/s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"stability_hud_analysis_{controller_type.replace(' ', '_')}.png")
    fig.savefig(out_png, dpi=150)
    fig.clear()
    print(f"Dashboard saved: {out_png}")
    if scheduler is not None and gain_log:
        import csv
        csv_dir = "output/csv"            # CSV files live in output/csv, graphs stay in output/graphs
        os.makedirs(csv_dir, exist_ok=True)
        gain_csv = os.path.join(csv_dir, f"pybullet_ml_gains_{controller_type.replace(' ', '_')}.csv")
        used = {"Sliding Mode": scheduler.gain_names[3:6],
                "Fuzzy": scheduler.gain_names[6:7]}.get(controller_type, scheduler.gain_names[0:3])
        idx = [scheduler.gain_names.index(n) for n in used]
        with open(gain_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "position_x_m", "speed_mps", "terrain_grade_deg", "roll_deg",
                             "pitch_deg", "lateral_speed_mps", "accel_mps2", *used])
            for row, state in zip(gain_log, ml_state_log):
                writer.writerow([f"{row[0]:.2f}", *[f"{v:.4f}" for v in state],
                                 *[f"{row[1 + i]:.4f}" for i in idx]])
        print(f"ML gain history saved: {gain_csv}")
    try:   # summary graph for this terrain run; must never break the simulation itself
        from plots import plot_pybullet_summary
        saved = plot_pybullet_summary(
            {"time": time_log, "position": pos_log, "velocity": vel_xyz_log, "mass": total_mass,
             "drive_force": np.asarray(torque_log) / 0.30,
             "roll": roll_log, "pitch": pitch_log,
             "terrain_slope": slope_log, "terrain_z": terrain_z_log},
            controller_type, ml_adaptive=scheduler is not None)
        print(f"Summary graph saved: {saved}")
    except Exception as exc:
        print(f"[WARNING] summary graph not saved: {exc}")
    return {"time": time_log, "pitch": pitch_log, "roll": roll_log,
            "velocity": velocity_log, "target": target_log, "torque": torque_log}

if __name__ == "__main__":
    run_wheelchair_stability_analyzer(controller_type="Nonlinear PID")