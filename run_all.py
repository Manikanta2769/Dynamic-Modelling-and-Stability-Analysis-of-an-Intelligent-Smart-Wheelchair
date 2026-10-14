import argparse
import csv
from datetime import datetime
import os
import sys

# Set backend before any project module imports matplotlib.pyplot.
import matplotlib
matplotlib.use("Agg", force=True)

# Always work relative to this folder, so output/ and the URDF are found
# no matter where the command is launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

PYBULLET_NAMES = {"pid": "Linear PID", "nonlinear_pid": "Nonlinear PID",
                  "smc": "Sliding Mode", "fuzzy": "Fuzzy"}


def run_sim(args):
    from utils import create_directories
    from simulation import WheelchairSimulation, compare_scenarios, compare_controllers
    from plots import generate_all_plots, plot_scenario_comparisons, plot_controller_graphs
    create_directories()
    print(f"\n=== NumPy 3-D model: {args.controller}, grade {args.slope:g} deg, "
          f"cross-slope {args.cross:g} deg, scenario {args.scenario} ===")
    sim = WheelchairSimulation(args.controller, slope=args.slope, scenario=args.scenario,
                               cross_slope=args.cross, ml_adaptive=args.ml_adaptive)
    print(f"ML adaptive gain scheduling: {'ON' if args.ml_adaptive else 'OFF'}")
    if getattr(args, "full_run", False):
        # Full run: the comparison below simulates every controller/scenario and
        # saves their graphs, so the extra single-controller run is skipped.
        results = None
    else:
        results = sim.run()
        sim.summary()
        generate_all_plots(results)
        sim.export_csv()
    if args.compare:
        print("Running controller x scenario comparison (a few minutes)...")
        print(f"Comparison resolution: {args.compare_step:g} s, duration: {args.compare_duration:g} s")
        if getattr(args, "full_run", False):
            # Default run: ONE scenario (--scenario, default smooth), all controllers,
            # one summary graph per controller in output/controller_graphs/<controller>/
            exps = {args.scenario: compare_controllers(
                args.slope, args.scenario, args.cross, ml_adaptive=args.ml_adaptive,
                duration=args.compare_duration, time_step=args.compare_step)}
            for controller_result in exps[args.scenario].values():
                print(f"  saving graph: {controller_result['controller_name']} ...", flush=True)
                plot_controller_graphs(controller_result)
            print("Controller graphs saved under: output/controller_graphs")
        else:
            exps = compare_scenarios(
                args.slope, args.cross, ml_adaptive=args.ml_adaptive,
                duration=args.compare_duration, time_step=args.compare_step)
            plot_scenario_comparisons(exps, args.slope, args.cross)
        rows = []
        screen_columns = {
            "speed_screen_pass": "safety_speed_ok",
            "reverse_motion_screen_pass": "safety_reverse_motion_ok",
            "acceleration_screen_pass": "safety_accel_ok",
            "lateral_acceleration_screen_pass": "safety_lateral_accel_ok",
            "roll_screen_pass": "safety_roll_ok",
            "pitch_screen_pass": "safety_pitch_ok",
            "yaw_rate_screen_pass": "safety_yaw_rate_ok",
            "support_screen_pass": "safety_support_ok",
            "traction_screen_pass": "safety_traction_ok",
            "actuator_envelope_feasible": "safety_governor_feasible",
        }
        for scenario, controller_results in exps.items():
            for controller_name, result in controller_results.items():
                t = result["time"]
                speed_error = result["velocity_forward"] - result["reference_speed"]
                row = {
                    "controller": controller_name,
                    "gain_mode": "ml_adaptive" if args.ml_adaptive else "fixed_gain",
                    "scenario": scenario,
                    "grade_deg": args.slope,
                    "cross_slope_deg": args.cross,
                    "duration_s": float(t[-1]),
                    "time_step_s": args.compare_step,
                    "forward_speed_rmse_mps": float((speed_error @ speed_error / len(speed_error)) ** 0.5),
                    "peak_abs_speed_error_mps": float(abs(speed_error).max()),
                    "minimum_forward_speed_mps": float(result["velocity_forward"].min()),
                    "peak_forward_speed_mps": float(result["velocity_forward"].max()),
                    "peak_forward_acceleration_mps2": float(abs(result["acceleration_forward"]).max()),
                    "peak_lateral_acceleration_mps2": float(abs(result["acceleration_lateral"]).max()),
                    "peak_roll_deg": float(abs(result["roll"]).max()),
                    "peak_pitch_deg": float(abs(result["pitch"]).max()),
                    "peak_total_slip_mps": float(result["slip_speeds"].max()),
                    "peak_longitudinal_slip_mps": float(abs(result["longitudinal_slip_speeds"]).max()),
                    "peak_lateral_slip_mps": float(abs(result["lateral_slip_speeds"]).max()),
                    "peak_friction_utilization": float(result["friction_utilization_3d"].max()),
                    "minimum_normal_load_n": float(result["normal_loads"].min()),
                    "absolute_mechanical_energy_j": float(
                        (0.5 * (abs(result["power"][:-1]) + abs(result["power"][1:]))
                         * (t[1:] - t[:-1])).sum()),
                    "simulation_terminated": bool(result["simulation_terminated"].any()),
                    "termination_reason": result["termination_reason"],
                }
                for output_name, result_key in screen_columns.items():
                    row[output_name] = bool(result[result_key].all())
                row["all_screens_pass"] = all(row[name] for name in screen_columns)
                rows.append(row)

        comparison_csv = os.path.join("output", "csv", args.compare_output)
        os.makedirs(os.path.dirname(comparison_csv), exist_ok=True)

        def write_comparison(path):
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        try:
            write_comparison(comparison_csv)
        except PermissionError:
            # Excel/OneDrive can lock an existing CSV. Preserve the completed
            # comparison under a timestamped name instead of discarding it.
            stem, extension = os.path.splitext(comparison_csv)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_csv = f"{stem}_{timestamp}{extension}"
            write_comparison(comparison_csv)
            print("The requested comparison CSV was locked; saved a new copy instead.")
        print(f"Numerical comparison table saved to: {comparison_csv}")
        print("Comparison uses a coarsened step for screening; rerun finalist cases at 0.0025 s before reporting.")


def run_visual(args):
    print("\n=== PyBullet co-simulation (run_co_simulation.py) ===")
    try:
        from run_co_simulation import run_wheelchair_stability_analyzer
    except ImportError as exc:
        print(f"\n[ERROR] {exc}")
        return False
    name = PYBULLET_NAMES.get(args.controller, "Nonlinear PID")
    run_wheelchair_stability_analyzer(controller_type=name, headless=args.headless,
                                      ml_adaptive=args.ml_adaptive)
    return True


def run_demo(args):
    # Compatibility alias: the old demo fabricated attitude signals and did
    # not use the project plant.
    run_sim(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="NumPy 3-D model + PyBullet co-simulation")
    mode.add_argument("--sim", action="store_true", help="NumPy 3-D model + graphs only")
    mode.add_argument("--visual", action="store_true", help="PyBullet co-simulation only")
    mode.add_argument("--demo", action="store_true", help="compatibility alias for the NumPy 3-D model")
    ap.add_argument("--headless", action="store_true", help="PyBullet without a window (saves graph only)")
    ap.add_argument("--controller", default="nonlinear_pid", choices=["pid", "nonlinear_pid", "fuzzy", "smc"])
    ap.add_argument("--slope", type=float, default=5.0, help="grade in deg for the NumPy model")
    ap.add_argument("--cross", type=float, default=0.0, help="cross-slope in deg for the NumPy model")
    ap.add_argument("--scenario", default="smooth", choices=["smooth", "rough", "sudden", "repeated"])
    ap.add_argument("--compare", action="store_true", help="with --sim/--all: all controllers x all scenarios")
    ap.add_argument("--compare-duration", type=float, default=20.0,
                    help="duration in seconds for each controller-comparison case")
    ap.add_argument("--compare-step", type=float, default=0.0025,
                    help="time step in seconds for controller-comparison cases (default: 0.0025; "
                         "0.01 is too coarse and gives numerical contact chatter in the graphs)")
    ap.add_argument("--compare-output", default="controller_scenario_comparison.csv",
                    help="CSV filename for numerical controller/scenario comparison")
    ap.add_argument("--ml-adaptive", action="store_true", help="enable ML-predicted dynamic gain scheduling in the NumPy 3-D model")
    a = ap.parse_args(argv)
    if a.compare and (a.compare_duration <= 0 or a.compare_step <= 0):
        ap.error("--compare-duration and --compare-step must be positive")

    if a.visual:
        run_visual(a)
    elif a.demo:
        run_demo(a)
    elif a.sim:
        run_sim(a)
    elif a.all:
        run_sim(a)
        run_visual(a)
    else:
        # Default (no mode flag): ML adaptive gains ON, PyBullet terrain track
        # (2 speed humps, 10 deg climb, crest, 10 deg decline).
        #  * Linear PID, Sliding Mode and Fuzzy run headless (no window) -> graph only
        #  * Nonlinear PID runs with the visual simulation window -> graph + visual
        #  * Fuzzy runs headless on the SAME terrain track, so its graph matches the others
        # Each run saves output/controller_graphs/<controller>/00_summary.png.
        a.ml_adaptive = True
        window = a.headless
        for ctrl in ("pid", "smc", "fuzzy"):
            a.controller, a.headless = ctrl, True
            run_visual(a)
        a.controller, a.headless = "nonlinear_pid", window
        run_visual(a)


if __name__ == "__main__":
    main()