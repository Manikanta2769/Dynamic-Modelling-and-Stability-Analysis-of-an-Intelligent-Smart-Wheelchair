"""Controller-specific ML gain scheduling for the 3-D simulator.

The model artifact is trained from simulator random-search labels and correlated
states drawn from designed controller/terrain rollouts. It is not independently
validated or evidence of real-world performance. Without a compatible trained
model, a documented bounded rule-based fallback is used.
"""
from pathlib import Path
import json
import numpy as np

MODEL_PATH = Path(__file__).with_name("ml_gain_model.joblib")
MODEL_METADATA_PATH = Path(__file__).with_name("ml_gain_model.json")
MODEL_VERSION = 5
CONTROLLER_NAMES = ["pid", "nonlinear_pid", "smc", "fuzzy"]
FEATURE_NAMES = [
    "abs_speed_error", "grade_deg", "cross_slope_deg",
    "abs_roll_deg", "abs_pitch_deg", "lateral_speed", "abs_accel",
    *[f"controller_{name}" for name in CONTROLLER_NAMES],
]
GAIN_NAMES = [
    "pid_kp_scale", "pid_ki_scale", "pid_kd_scale",
    "smc_lambda_scale", "smc_k_scale", "smc_phi_scale",
    "fuzzy_output_scale"
]

def feature_vector(speed_error, slope_deg, cross_slope_deg, roll_deg,
                   pitch_deg, lateral_speed, acceleration, controller_name):
    """Bounded state features plus a one-hot controller identifier."""
    controller_name = str(controller_name).lower()
    if controller_name not in CONTROLLER_NAMES:
        raise ValueError(f"Unknown controller feature: {controller_name}")
    state_features = [
        np.clip(abs(speed_error), 0, 2.5),
        np.clip(slope_deg, -35, 35),
        np.clip(cross_slope_deg, -20, 20),
        np.clip(abs(roll_deg), 0, 30),
        np.clip(abs(pitch_deg), 0, 40),
        np.clip(abs(lateral_speed), 0, 2.5),
        np.clip(abs(acceleration), 0, 4.0),
    ]
    return np.asarray(state_features + [
        float(controller_name == name) for name in CONTROLLER_NAMES
    ], dtype=float)

def engineered_gain_targets(x):
    """Conservative bounded target policy used to bootstrap synthetic labels."""
    err, grade, cross, roll, pitch, lateral, accel = np.asarray(x, dtype=float)[:7]
    terrain_risk = np.clip((abs(grade) / 35 + abs(cross) / 20 + roll / 15 + pitch / 20) / 4, 0, 1)
    motion_risk = np.clip((lateral / 1.5 + accel / 2.0) / 2, 0, 1)
    risk = np.clip(0.7 * terrain_risk + 0.3 * motion_risk, 0, 1)
    kp = np.clip(0.85 + 0.30 * min(err / 1.2, 1) - 0.15 * risk, 0.65, 1.15)
    ki = np.clip(0.90 - 0.45 * risk, 0.40, 0.95)
    kd = np.clip(0.90 + 0.35 * risk, 0.85, 1.30)
    smc_lambda = np.clip(0.90 + 0.20 * min(err / 1.2, 1) - 0.15 * risk, 0.65, 1.10)
    smc_k = np.clip(0.90 + 0.30 * risk, 0.85, 1.25)
    smc_phi = np.clip(1.0 + 0.75 * risk, 1.0, 1.75)
    fuzzy = np.clip(0.95 + 0.25 * min(err / 1.2, 1) - 0.20 * risk, 0.65, 1.20)
    return np.array([kp, ki, kd, smc_lambda, smc_k, smc_phi, fuzzy])

class MLGainTuner:
    def __init__(self):
        self.model = None
        self.model_source = "bounded rule-based fallback"
        try:
            if MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
                metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
                if (metadata.get("model_version") != MODEL_VERSION
                        or metadata.get("feature_names") != FEATURE_NAMES
                        or metadata.get("gain_names") != GAIN_NAMES
                        or metadata.get("controller_names") != CONTROLLER_NAMES):
                    raise ValueError("model metadata/schema is stale")
                import joblib
                self.model = joblib.load(MODEL_PATH)
                if hasattr(self.model, "n_jobs"):
                    self.model.n_jobs = 1
                self.model_source = "versioned RandomForestRegressor"
            elif MODEL_PATH.exists():
                self.model_source = "bounded rule-based fallback (model requires retraining)"
        except Exception as exc:
            # Safe fallback: model file/dependency problems must not stop simulation.
            self.model = None
            self.model_source = f"bounded rule-based fallback ({type(exc).__name__}: {exc})"

    def predict(self, features):
        x = np.asarray(features, dtype=float).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} finite ML gain features")
        if self.model is not None:
            try:
                prediction = np.asarray(self.model.predict(x), dtype=float)
                if prediction.shape != (1, len(GAIN_NAMES)) or not np.all(np.isfinite(prediction)):
                    raise ValueError("model returned invalid gain vector")
                y = prediction[0]
            except Exception as exc:
                self.model = None
                self.model_source = f"bounded rule-based fallback ({type(exc).__name__}: {exc})"
                y = engineered_gain_targets(x[0])
        else:
            y = engineered_gain_targets(x[0])
        # Final safety bounds are applied even to ML predictions.
        bounds = [(0.60,1.20),(0.35,1.00),(0.75,1.40),
                  (0.60,1.20),(0.75,1.35),(1.0,1.80),(0.60,1.25)]
        return {name: float(np.clip(val, lo, hi))
                for name, val, (lo, hi) in zip(GAIN_NAMES, y, bounds)}
