"""Metrics for signed forward-speed and position-reference tracking.

These simulation metrics do not establish real-world safety or compliance.
"""
import numpy as np


def _finite_array(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite samples")
    return arr


def evaluate(results, target_position=None):
    """Summarize tracking and model-screen outputs from WheelchairSimulation.

    ``target_position`` is an optional legacy distance target. Position tracking
    otherwise uses the time-varying ``reference`` signal; velocity uses signed
    forward speed and ``reference_speed`` so reverse motion counts as error.
    """
    t = _finite_array(results["time"], "time")
    x = _finite_array(results["position"], "position")
    vf = _finite_array(results.get("velocity_forward", results["velocity"]), "velocity_forward")
    ref_v = _finite_array(results.get("reference_speed", np.zeros_like(vf)), "reference_speed")
    n = min(map(len, (t, x, vf, ref_v)))
    t, x, vf, ref_v = t[:n], x[:n], vf[:n], ref_v[:n]
    ref_x = _finite_array(results.get("reference", np.full(n, target_position or 0.0)), "reference")[:n]
    out = {
        "Position reference RMSE (m)": float(np.sqrt(np.mean((x-ref_x)**2))),
        "Position reference MAE (m)": float(np.mean(np.abs(x-ref_x))),
        "Forward speed RMSE (m/s)": float(np.sqrt(np.mean((vf-ref_v)**2))),
        "Forward speed MAE (m/s)": float(np.mean(np.abs(vf-ref_v))),
        "Minimum forward speed (m/s)": float(np.min(vf)),
        "Peak |roll| relative to terrain (deg)": float(np.max(np.abs(results["roll"]))),
        "Peak |pitch| relative to terrain (deg)": float(np.max(np.abs(results["pitch"]))),
        "Peak wheel slip speed (m/s)": float(np.max(results["slip_speeds"])),
        "Minimum normal load (N)": float(np.min(results["normal_loads"])),
        "Peak friction utilization screen": float(np.max(results["friction_utilization_3d"])),
        "Simulation terminated early": bool(np.any(results["simulation_terminated"])),
    }
    if len(t) > 1:
        out["Absolute mechanical energy (J)"] = float(np.trapz(np.abs(results["power"][:n]), t))
    if target_position is not None:
        out["Final distance error to legacy target (m)"] = float(target_position-x[-1])
    for key, label in (("safety_speed_ok", "Speed screen passed"),
                        ("safety_reverse_motion_ok", "Forward-motion screen passed"),
                        ("safety_accel_ok", "Acceleration screen passed"),
                        ("safety_roll_ok", "Roll screen passed"),
                        ("safety_pitch_ok", "Pitch screen passed"),
                        ("safety_yaw_rate_ok", "Yaw-rate screen passed"),
                        ("safety_lateral_accel_ok", "Lateral-acceleration screen passed"),
                        ("safety_support_ok", "Support polygon screen passed"),
                        ("safety_traction_ok", "Traction screen passed"),
                        ("safety_governor_feasible", "Acceleration-envelope actuator feasible")):
        if key in results:
            out[label] = bool(np.all(results[key]))
    return out


def print_metrics(metrics):
    print("\n========== SIMULATION PERFORMANCE METRICS ==========")
    for key, value in metrics.items():
        print(f"{key:<44}: {value:.4f}" if isinstance(value, float) else f"{key:<44}: {value}")
    print("Screens are model checks, not certified safety limits.")
    print("=====================================================")
