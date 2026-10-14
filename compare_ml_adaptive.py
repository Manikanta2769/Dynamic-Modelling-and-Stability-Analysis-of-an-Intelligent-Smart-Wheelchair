"""Compare fixed-gain and ML-adaptive controllers on identical 3-D test conditions.

Example:
  python compare_ml_adaptive.py --slope 5 --cross 5 --scenario sudden --horizon 6 --step 0.01

This is a simulation comparison, not physical wheelchair safety validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import WheelchairSimulation

HERE = Path(__file__).resolve().parent
CONTROLLERS = ("pid", "nonlinear_pid", "smc", "fuzzy")
GRAPH_DIR = HERE / "output" / "graphs"


def summarize(result, controller, mode, scenario, slope, cross):
    # Track signed velocity along the terrain tangent so reverse motion cannot
    # look like successful forward speed tracking.
    speed = np.asarray(result["velocity_forward"], dtype=float)
    ref = np.asarray(result["reference_speed"], dtype=float)
    t = np.asarray(result["time"], dtype=float)
    slip = np.asarray(result["slip_speeds"], dtype=float)
    threshold = 0.9 * float(np.max(ref))
    ref_cross = np.flatnonzero(ref >= threshold)
    speed_cross = np.flatnonzero(speed >= threshold)
    cruise_delay = (float(t[speed_cross[0]] - t[ref_cross[0]])
                    if len(speed_cross) and len(ref_cross) else np.nan)
    recovery = np.nan
    disturbance_start = np.flatnonzero(np.asarray(result["disturbance_force"]) > 0)
    if len(disturbance_start):
        start = int(disturbance_start[0])
        settled = [i for i in range(start, len(t))
                   if np.all(np.abs(speed[i:] - ref[i:]) <= 0.05)]
        if settled:
            recovery = float(t[settled[0]] - t[start])
    abs_power = np.abs(np.asarray(result["power"], dtype=float))
    energy_j = float(np.sum(0.5 * (abs_power[:-1] + abs_power[1:]) * np.diff(t)))
    return {
        "controller": controller,
        "mode": mode,
        "scenario": scenario,
        "slope_deg": slope,
        "cross_slope_deg": cross,
        "speed_rmse_mps": float(np.sqrt(np.mean((speed - ref) ** 2))),
        "peak_abs_speed_error_mps": float(np.max(np.abs(speed - ref))),
        "minimum_forward_speed_mps": float(np.min(speed)),
        "time_to_90pct_cruise_s": float(t[speed_cross[0]]) if len(speed_cross) else np.nan,
        "reference_time_to_90pct_cruise_s": float(t[ref_cross[0]]) if len(ref_cross) else np.nan,
        "90pct_cruise_delay_vs_reference_s": cruise_delay,
        "tracking_recovery_after_disturbance_s": recovery,
        "peak_speed_mps": float(np.max(speed)),
        "peak_abs_roll_deg": float(np.max(np.abs(result["roll"]))),
        "peak_abs_pitch_deg": float(np.max(np.abs(result["pitch"]))),
        "peak_slip_speed": float(np.max(slip)),
        "peak_longitudinal_slip_mps": float(np.max(np.abs(result["longitudinal_slip_speeds"]))),
        "peak_lateral_slip_mps": float(np.max(np.abs(result["lateral_slip_speeds"]))),
        "peak_wheel_speed_radps": float(np.max(np.abs(result["wheel_speed"]))),
        "minimum_normal_load_n": float(np.min(result["normal_loads"])),
        "peak_friction_utilization": float(np.max(result["friction_utilization_3d"])),
        "peak_forward_accel_mps2": float(np.max(np.abs(result["acceleration_forward"]))),
        "peak_lateral_accel_mps2": float(np.max(np.abs(result["acceleration_lateral"]))),
        "mean_abs_power_w": float(np.mean(np.abs(result["power"]))),
        "absolute_energy_j": energy_j,
        "all_safety_screens_pass": bool(all(np.all(result[key]) for key in (
            "safety_speed_ok", "safety_reverse_motion_ok", "safety_accel_ok", "safety_roll_ok",
            "safety_lateral_accel_ok", "safety_pitch_ok",
            "safety_yaw_rate_ok", "safety_traction_ok", "safety_support_ok",
            "safety_governor_feasible"))),
        "speed_screen_pass": bool(np.all(result["safety_speed_ok"])),
        "reverse_motion_screen_pass": bool(np.all(result["safety_reverse_motion_ok"])),
        "acceleration_screen_pass": bool(np.all(result["safety_accel_ok"])),
        "lateral_acceleration_screen_pass": bool(np.all(result["safety_lateral_accel_ok"])),
        "roll_screen_pass": bool(np.all(result["safety_roll_ok"])),
        "pitch_screen_pass": bool(np.all(result["safety_pitch_ok"])),
        "yaw_rate_screen_pass": bool(np.all(result["safety_yaw_rate_ok"])),
        "traction_screen_pass": bool(np.all(result["safety_traction_ok"])),
        "support_screen_pass": bool(np.all(result["safety_support_ok"])),
        "acceleration_envelope_actuator_feasible": bool(np.all(result["safety_governor_feasible"])),
        "simulation_terminated": bool(np.any(result["simulation_terminated"])),
        "termination_reason": str(result.get("termination_reason", "none")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slope", type=float, default=5.0)
    ap.add_argument("--cross", type=float, default=5.0)
    ap.add_argument("--scenario", choices=("smooth", "rough", "sudden", "repeated"), default="sudden")
    ap.add_argument("--horizon", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    ap.add_argument("--plot-controller", choices=CONTROLLERS, default="nonlinear_pid",
                    help="controller shown in the fixed-vs-ML response figure")
    ap.add_argument("--output", default="ml_adaptive_comparison.csv",
                    help="CSV output filename (use separate files for separate scenarios)")
    args = ap.parse_args()
    if args.horizon <= 0 or args.step <= 0:
        ap.error("--horizon and --step must be positive")

    output_tag = Path(args.output).stem
    rows = []
    plot_results = {}
    for name in args.controllers:
        for adaptive in (False, True):
            mode = "ml_adaptive" if adaptive else "fixed_gain"
            print(f"Running {name} ({mode})...", flush=True)
            sim = WheelchairSimulation(
                controller_name=name, slope=args.slope, cross_slope=args.cross,
                scenario=args.scenario, ml_adaptive=adaptive,
                duration=args.horizon, time_step=args.step)
            result = sim.run()
            rows.append(summarize(result, name, mode, args.scenario,
                                  args.slope, args.cross))
            if name == args.plot_controller:
                plot_results[mode] = result
            if adaptive and "ml_gain_scales" in result:
                cols = ["time", *result["ml_gain_names"]]
                trace = pd.DataFrame(
                    np.column_stack((result["ml_gain_time"], result["ml_gain_scales"])),
                    columns=cols)
                trace.to_csv(HERE / f"ml_gain_trace_{name}_{output_tag}.csv", index=False)
                print(f"  Model source: {result['ml_model_source']}")

    out = pd.DataFrame(rows)
    # Direct paired deltas make the adaptation's tradeoffs legible without
    # treating any one tracking metric as a complete safety verdict.
    for metric in ("speed_rmse_mps", "time_to_90pct_cruise_s", "absolute_energy_j"):
        # Use bracket access: ``DataFrame.mode`` is a pandas method, not the
        # comparison table's "mode" column.
        fixed = out[out["mode"] == "fixed_gain"].set_index("controller")[metric]
        adaptive = out[out["mode"] == "ml_adaptive"].set_index("controller")[metric]
        out[f"{metric}_change_pct_vs_fixed"] = out.apply(
            lambda row: (100.0 * (adaptive[row["controller"]] - fixed[row["controller"]])
                         / abs(fixed[row["controller"]]))
            if np.isfinite(fixed[row["controller"]]) and abs(fixed[row["controller"]]) > 1e-12
            and np.isfinite(adaptive[row["controller"]]) else np.nan, axis=1)
    output = HERE / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fixed_result = plot_results.get("fixed_gain")
    adaptive_result = plot_results.get("ml_adaptive")
    if fixed_result is not None and adaptive_result is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for label, result in (("Fixed gains", fixed_result), ("ML adaptive", adaptive_result)):
            axes[0].plot(result["time"], result["velocity_forward"], label=label)
            axes[1].plot(result["time"], result["roll"], label=f"{label} roll")
            axes[1].plot(result["time"], result["pitch"], linestyle="--", label=f"{label} pitch")
        axes[0].plot(fixed_result["time"], fixed_result["reference_speed"], "k:", label="Reference")
        axes[0].set_ylabel("Forward speed (m/s)")
        axes[1].set_ylabel("Attitude relative to terrain (deg)")
        axes[1].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True); ax.legend(fontsize=8)
        fig.suptitle(f"Fixed vs ML-adaptive: {args.plot_controller}")
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / f"ml_fixed_vs_adaptive_{args.plot_controller}.png", dpi=160)
        plt.close(fig)
        if "ml_gain_scales" in adaptive_result:
            fig, ax = plt.subplots(figsize=(10, 5))
            for gain_name, values in zip(adaptive_result["ml_gain_names"],
                                         adaptive_result["ml_gain_scales"].T):
                ax.plot(adaptive_result["ml_gain_time"], values, label=gain_name)
            ax.set(xlabel="Time (s)", ylabel="Gain multiplier", title="ML gain schedule")
            ax.grid(True); ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(GRAPH_DIR / f"ml_gain_schedule_{args.plot_controller}.png", dpi=160)
            plt.close(fig)
    print("\nComparison summary:")
    print(out.to_string(index=False))
    print(f"\nSaved {output.name}; gain traces and selected response figures are in {GRAPH_DIR}.")
    print("Compare fixed_gain vs ml_adaptive per controller. Lower signed forward-speed "
          "RMSE indicates better tracking; also inspect peak error, roll/pitch, slip, "
          "power, and termination. A shorter rise time alone is not sufficient.")


if __name__ == "__main__":
    main()
