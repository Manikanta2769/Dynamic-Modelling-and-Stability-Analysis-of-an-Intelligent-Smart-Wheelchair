import numpy as np
from dynamics import terrain

class ControllerPreset:
    """Safety limits and preset user parameters."""
    def __init__(self, max_speed=1.2, max_torque=75.0, max_accel=1.0):
        self.max_speed = max_speed      # m/s
        self.max_torque = max_torque    # Nm aggregate
        self.max_accel = max_accel      # m/s^2


class LinearPIDController:
    """Standard Linear PID Speed Controller."""
    def __init__(self, kp=110.0, ki=10.0, kd=4.0, preset=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_v = None
        self.accel_f = 0.0
        self.preset = preset or ControllerPreset()

    def reset(self):
        self.integral = 0.0
        self.prev_v = None
        self.accel_f = 0.0

    def compute(self, target_v, current_v, dt):
        target_v = np.clip(target_v, -self.preset.max_speed, self.preset.max_speed)
        error = target_v - current_v

        # Derivative on measurement = -d(v)/dt (filtered acceleration).
        if self.prev_v is not None and dt > 0:
            self.accel_f = 0.9 * self.accel_f + 0.1 * (current_v - self.prev_v) / dt
        self.prev_v = current_v
        derivative = -self.accel_f

        u_unsat = self.kp * error + self.ki * self.integral + self.kd * derivative
        tau = np.clip(u_unsat, -self.preset.max_torque, self.preset.max_torque)
        # Anti-windup: only integrate while not saturated
        if abs(u_unsat) < self.preset.max_torque:
            self.integral = np.clip(self.integral + error * dt, -1.0, 1.0)
        return tau


class NonlinearPIDController:
    """Nonlinear PID speed controller with slope-aware gravity feedforward.

    PyBullet convention: pitch is rotation about +Y, so NOSE-UP (incline) is
    NEGATIVE pitch and NOSE-DOWN (decline) is POSITIVE pitch.
        incline -> gravity pulls back  -> positive drive torque needed
        decline -> gravity pushes fwd  -> negative (braking) torque needed
    """
    def __init__(self, kp=120.0, ki=14.0, kd=6.0, alpha=1.2, preset=None,
                 wheel_radius=0.30):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.wheel_radius = wheel_radius
        self.integral = 0.0
        self.preset = preset or ControllerPreset()
    

    def reset(self):
        self.integral = 0.0

    def compute(self, target_v, current_v, dt, pitch_rad=0.0, m_eq=82.0, R_rr=12.0):
        target_v = np.clip(target_v, -self.preset.max_speed, self.preset.max_speed)
        error = target_v - current_v

        nl_kp = self.kp * (1.0 + self.alpha * np.abs(error))

        # Gravity force along the travel direction (+ = resists forward motion).
        f_gravity = -m_eq * 9.81 * np.sin(pitch_rad)
        # Rolling resistance, smooth in speed (no chatter around v = 0).
        f_rolling = R_rr * np.tanh(current_v / 0.1)
        tau_feedforward = (f_gravity + f_rolling) * self.wheel_radius

        u_unsat = nl_kp * error + self.ki * self.integral + tau_feedforward
        tau = np.clip(u_unsat, -self.preset.max_torque, self.preset.max_torque)

        # Anti-windup: integrate only when not saturated; keep it small since
        # the feedforward already carries the slope load.
        if abs(u_unsat) < self.preset.max_torque:
            self.integral = np.clip(self.integral + error * dt, -1.0, 1.0)
        return tau


class SlidingModeController:
    """Sliding Mode Control (SMC) for robust speed tracking."""
    def __init__(self, lambda_param=4.0, K_gain=40.0, boundary_layer=0.1, preset=None):
        self.lambda_param = lambda_param
        self.K_gain = K_gain
        self.phi = boundary_layer
        self.preset = preset or ControllerPreset()

    def reset(self):
        pass

    def compute(self, target_v, current_v, dt=None, pitch_rad=0.0, m_eq=82.0, R_rr=12.0,
                wheel_radius=0.30):
        """Boundary-layer sliding mode with gravity/rolling feedforward.

        Without the feedforward the switching term (K_gain = 40 Nm) cannot hold a
        10 deg slope (~42 Nm needed), so the chair stalls going uphill.
        pitch_rad uses the PyBullet convention (nose-down = positive).
        """
        target_v = np.clip(target_v, -self.preset.max_speed, self.preset.max_speed)
        s = current_v - target_v
        sat_s = np.clip(s / self.phi, -1.0, 1.0)
        tau_ff = (-m_eq * 9.81 * np.sin(pitch_rad) + R_rr * np.tanh(current_v / 0.1)) * wheel_radius
        u_control = tau_ff - self.K_gain * sat_s
        tau = np.clip(u_control, -self.preset.max_torque, self.preset.max_torque)
        return tau


class FuzzyController:
    """Fuzzy speed controller for the PyBullet co-simulation.

    Same rule base as the NumPy model's Fuzzy3D (five triangular error sets
    NB, NS, Z, PS, PB with singleton outputs, weighted-average defuzzification),
    plus the same slope/rolling feedforward used by the Nonlinear PID and SMC
    controllers so it can hold the 10 deg climb and brake on the decline.
    `output_scale` is the ML-scheduled gain (fuzzy_output_scale).
    PyBullet convention: nose-up pitch is NEGATIVE.
    """
    SETS = [(-1.5, -1.0, -0.5, -1.0), (-1.0, -0.5, 0.0, -0.5), (-0.5, 0.0, 0.5, 0.0),
            (0.0, 0.5, 1.0, 0.5), (0.5, 1.0, 1.5, 1.0)]

    def __init__(self, error_range=0.6, output_range=60.0, ki=10.0, preset=None,
                 wheel_radius=0.30):
        self.error_range = error_range      # m/s of speed error that maps to +-1
        self.output_range = output_range    # Nm of fuzzy torque at +-1
        self.ki = ki
        self.output_scale = 1.0
        self.wheel_radius = wheel_radius
        self.integral = 0.0
        self.preset = preset or ControllerPreset()

    def reset(self):
        self.integral = 0.0

    @staticmethod
    def _tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x < b else (c - x) / (c - b)

    def compute(self, target_v, current_v, dt, pitch_rad=0.0, m_eq=82.0, R_rr=12.0):
        target_v = np.clip(target_v, -self.preset.max_speed, self.preset.max_speed)
        error = target_v - current_v
        e = float(np.clip(error / self.error_range, -1.0, 1.0))
        num = den = 0.0
        for a, b, c, centre in self.SETS:
            mu = self._tri(e, a, b, c)
            num += mu * centre
            den += mu
        fuzzy_out = (num / den if den > 1e-9 else 0.0) * self.output_range * self.output_scale

        f_gravity = -m_eq * 9.81 * np.sin(pitch_rad)
        f_rolling = R_rr * np.tanh(current_v / 0.1)
        tau_feedforward = (f_gravity + f_rolling) * self.wheel_radius

        u_unsat = fuzzy_out + self.ki * self.integral + tau_feedforward
        tau = np.clip(u_unsat, -self.preset.max_torque, self.preset.max_torque)
        if abs(u_unsat) < self.preset.max_torque:
            self.integral = np.clip(self.integral + error * dt, -1.0, 1.0)
        return tau


# =====================================================================
# 3-D model interface (used by simulation.py)
# ---------------------------------------------------------------------
# Each controller tracks a ramped speed reference along the terrain
# tangent, adds slope/rolling/drag feedforward, and returns a torque
# vector for the four wheels [FR, FL, RR, RL]. Only driven wheels are
# used by the plant. Gains come from parameters.py. These are provisional
# tuning values, not identified from hardware.
#
# Slope convention in THIS model: slope > 0 = uphill along +x, so gravity
# opposes motion (feedforward is +m g sin(slope)). This is the opposite of
# the PyBullet script, where nose-up pitch is negative.
# =====================================================================
import parameters as _P

CRUISE_SPEED = 1.2        # m/s reference speed
REF_ACCEL = 0.5           # m/s^2 reference ramp (kept below the plant limit)
START_TIME = 0.0


def reference_speed(t):
    """Ramped speed reference (m/s)."""
    return float(np.clip(REF_ACCEL * (t - START_TIME), 0.0, CRUISE_SPEED))


def reference_position(t):
    """Integral of the ramped speed reference (m along the terrain)."""
    t = max(t - START_TIME, 0.0)
    t_ramp = CRUISE_SPEED / REF_ACCEL
    if t <= t_ramp:
        return 0.5 * REF_ACCEL * t * t
    return 0.5 * REF_ACCEL * t_ramp**2 + CRUISE_SPEED * (t - t_ramp)


def reference_accel(t):
    """Acceleration of the ramped speed reference (m/s^2)."""
    return REF_ACCEL if START_TIME <= t < START_TIME + CRUISE_SPEED / REF_ACCEL else 0.0


class _Base3D:
    name = "base"

    def __init__(self, ml_adaptive=False):
        self.last_t = None
        self.integral = 0.0
        self.prev_vf = None
        self.accel_f = 0.0
        
        self.ml_adaptive = bool(ml_adaptive)

        # Limit ML gain predictions to once every 0.1 seconds.
        self.ml_update_interval = 0.1
        self.last_ml_update = float("-inf")

        self.ml_gains = {
            "pid_kp_scale": 1.0, "pid_ki_scale": 1.0, "pid_kd_scale": 1.0,
            "smc_lambda_scale": 1.0, "smc_k_scale": 1.0,
            "smc_phi_scale": 1.0, "fuzzy_output_scale": 1.0,
        }

        self.gain_history = []
        self.state_history = []
        self.ml_gain_tuner = None
        if self.ml_adaptive:
            # Lazy import keeps baseline simulations independent of joblib/model files.
            from ml_gain_tuner import MLGainTuner
            self.ml_gain_tuner = MLGainTuner()

    def _schedule_ml_gains(self, t, v, euler, slope_deg, cross_deg):
        """Predict bounded gain scales from the current simulated state."""
        if not self.ml_adaptive or self.ml_gain_tuner is None:
            return
        v_f = float(np.dot(v, self._forward(slope_deg)))
        v_ref = reference_speed(t)
        _, _, lateral_axis = terrain(slope_deg, cross_deg)
        lateral_speed = float(np.dot(v, lateral_axis))
        features = self.ml_gain_tuner_features(
            v_ref - v_f, slope_deg, cross_deg,
            np.rad2deg(float(euler[0])), np.rad2deg(float(euler[1])),
            lateral_speed, self.accel_f, self.name)
        self.ml_gains = self.ml_gain_tuner.predict(features)
        # Inputs the model saw at this update (written next to the gains in the CSV
        # so every change in a gain can be traced to a change in the state).
        self.state_history.append([float(t), float(v_ref - v_f), float(slope_deg),
                                   float(np.rad2deg(float(euler[0]))), float(np.rad2deg(float(euler[1]))),
                                   float(lateral_speed), float(self.accel_f)])
        # Keep a trace for verification and plots/reports.
        self.gain_history.append([float(t)] + [self.ml_gains[k] for k in (
            "pid_kp_scale", "pid_ki_scale", "pid_kd_scale", "smc_lambda_scale",
            "smc_k_scale", "smc_phi_scale", "fuzzy_output_scale")])

    @staticmethod
    def ml_gain_tuner_features(speed_error, slope_deg, cross_slope_deg, roll_deg,
                               pitch_deg, lateral_speed, acceleration,
                               controller_name):
        from ml_gain_tuner import feature_vector
        return feature_vector(speed_error, slope_deg, cross_slope_deg, roll_deg,
                              pitch_deg, lateral_speed, acceleration,
                              controller_name)

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _forward(slope_deg):
        a = np.deg2rad(slope_deg)
        return np.array([np.cos(a), 0.0, np.sin(a)])

    def _feedforward(self, v_f, slope_deg, cross_deg):
        m, g = _P.mass_total, _P.g
        grade = m * g * np.sin(np.deg2rad(slope_deg))
        normal = m * g * np.cos(np.deg2rad(slope_deg)) * np.cos(np.deg2rad(cross_deg))
        rolling = _P.rolling_resistance * normal * np.tanh(v_f / 0.1)
        drag = 0.5 * _P.rho_air * _P.drag_coefficient * _P.frontal_area * v_f * abs(v_f)
        return grade + rolling + drag

    def _dt_and_accel(self, t, v_f):
        dt = 0.0 if self.last_t is None else max(t - self.last_t, 0.0)
        self.last_t = t
        if self.prev_vf is not None and dt > 0:
            self.accel_f = 0.9 * self.accel_f + 0.1 * (v_f - self.prev_vf) / dt
        self.prev_vf = v_f
        return dt

    def _force_limit(self):
        # Aggregate force at the driven wheels, matching simulation.py's
        # per-wheel wheel-side torque cap and equal torque allocation.
        return _P.motor_torque_max * _P.gear_ratio * _P.motor_efficiency / _P.wheel_radius

    def _force_to_torques(self, F):
        F = float(np.clip(F, -self._force_limit(), self._force_limit()))
        n_driven = max(int(np.sum(_P.driven_wheels)), 1)
        tau = np.zeros(4)
        tau[_P.driven_wheels] = F * _P.wheel_radius / n_driven
        return tau

    def _integrate(self, err, dt, F_unsat, limit):
        # anti-windup: only integrate while the force demand is not saturated
        if abs(F_unsat) < limit:
            self.integral = float(np.clip(self.integral + err * dt, -2.0, 2.0))

    def force(self, t, v_f, v_ref, slope_deg, cross_deg):
        raise NotImplementedError


    def command_3d(self, t, p, v, euler, w, ref, slope_deg, cross_deg):
        # Update ML-scheduled gains every 0.1 seconds of simulation time.
        if t - self.last_ml_update >= self.ml_update_interval - 1e-9:
            self._schedule_ml_gains(t, v, euler, slope_deg, cross_deg)
            self.last_ml_update = t

        v_f = float(np.dot(v, self._forward(slope_deg)))
        v_ref = reference_speed(t)

        F = self.force(t, v_f, v_ref, slope_deg, cross_deg)

        return self._force_to_torques(F)



class PID3D(_Base3D):
    name = "pid"

    def force(self, t, v_f, v_ref, slope_deg, cross_deg):
        dt = self._dt_and_accel(t, v_f)
        err = v_ref - v_f
        ff = self._feedforward(v_f, slope_deg, cross_deg)
        F = ff + _P.mass_total * reference_accel(t) + (_P.KP * self.ml_gains["pid_kp_scale"]) * err + (_P.KI * self.ml_gains["pid_ki_scale"]) * self.integral - (_P.KD * self.ml_gains["pid_kd_scale"]) * self.accel_f
        self._integrate(err, dt, F, self._force_limit())
        return F


class NonlinearPID3D(_Base3D):
    name = "nonlinear_pid"
    alpha = 1.2

    def force(self, t, v_f, v_ref, slope_deg, cross_deg):
        dt = self._dt_and_accel(t, v_f)
        err = v_ref - v_f
        ff = self._feedforward(v_f, slope_deg, cross_deg)
        kp = _P.KP * self.ml_gains["pid_kp_scale"] * (1.0 + self.alpha * abs(err))  # gain grows with error
        ki = _P.KI * self.ml_gains["pid_ki_scale"]
        kd = _P.KD * self.ml_gains["pid_kd_scale"]
        F = ff + _P.mass_total * reference_accel(t) + kp * err + ki * self.integral - 0.5 * kd * self.accel_f
        self._integrate(err, dt, F, self._force_limit())
        return F


class SlidingMode3D(_Base3D):
    name = "smc"

    def force(self, t, v_f, v_ref, slope_deg, cross_deg):
        dt = self._dt_and_accel(t, v_f)
        err = v_f - v_ref
        lambda_eff = _P.lambda_smc * self.ml_gains["smc_lambda_scale"]
        k_eff = _P.k_smc * self.ml_gains["smc_k_scale"]
        phi_eff = _P.boundary_layer * self.ml_gains["smc_phi_scale"]
        s = err + lambda_eff * 0.1 * self.integral  # integral sliding surface
        sat = float(np.clip(s / max(phi_eff, 1e-6), -1.0, 1.0))
        ff = self._feedforward(v_f, slope_deg, cross_deg)
        # equivalent control + boundary-layer switching term (provisional gains)
        F = ff + _P.mass_total * (REF_ACCEL if v_ref < CRUISE_SPEED else 0.0) \
            - _P.mass_total * (lambda_eff * err) - k_eff * 5.0 * sat
        self._integrate(err, dt, F, self._force_limit())
        return F


class Fuzzy3D(_Base3D):
    """Singleton weighted-average fuzzy speed controller (Sugeno-like output)."""
    name = "fuzzy"

    @staticmethod
    def _tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x < b else (c - x) / (c - b)

    def force(self, t, v_f, v_ref, slope_deg, cross_deg):
        dt = self._dt_and_accel(t, v_f)
        e_lo, e_hi = _P.fuzzy_error_range
        err = v_ref - v_f
        error_scale = max(abs(e_lo), abs(e_hi), 1e-9)
        e = float(np.clip(err / error_scale, -1.0, 1.0))
        # error sets: NB, NS, Z, PS, PB  -> output centres (normalised)
        sets = [(-1.5, -1.0, -0.5, -1.0), (-1.0, -0.5, 0.0, -0.5), (-0.5, 0.0, 0.5, 0.0),
                (0.0, 0.5, 1.0, 0.5), (0.5, 1.0, 1.5, 1.0)]
        num = den = 0.0
        for a, b, c, centre in sets:
            mu = self._tri(e, a, b, c)
            num += mu * centre
            den += mu
        out = num / den if den > 1e-9 else 0.0
        F_range = _P.mass_total * _P.max_acceleration * (abs(_P.fuzzy_output_range[1]) / 20.0) * self.ml_gains["fuzzy_output_scale"]
        ff = self._feedforward(v_f, slope_deg, cross_deg)
        F = ff + _P.mass_total * reference_accel(t) + out * F_range + _P.KI * self.integral
        self._integrate(err, dt, F, self._force_limit())
        return F


_CONTROLLERS_3D = {
    "pid": PID3D,
    "nonlinear_pid": NonlinearPID3D,
    "fuzzy": Fuzzy3D,
    "smc": SlidingMode3D,
}


def controller_names():
    return list(_CONTROLLERS_3D.keys())


def create_controller(name, ml_adaptive=False):
    key = str(name).lower().replace(" ", "_")
    if key not in _CONTROLLERS_3D:
        raise ValueError(f"Unknown controller '{name}'. Available: {controller_names()}")
    return _CONTROLLERS_3D[key](ml_adaptive=ml_adaptive)