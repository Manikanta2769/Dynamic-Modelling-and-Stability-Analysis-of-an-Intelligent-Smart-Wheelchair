"""Per-controller component, acceleration and force graphs.

ADD-ON SCRIPT: it only *imports* the existing simulation and never modifies
or overwrites any existing project file or graph. Everything is written to
``output/controller_graphs/<scenario>/<controller>/``.

Usage (from this folder)::

    python generate_controller_graphs.py                       # all 4 controllers, smooth, 5 deg
    python generate_controller_graphs.py --scenario rough sudden
    python generate_controller_graphs.py --slope 5 --cross 0 --ml-adaptive
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from parameters import wheel_radius, mass_total, dpi  # noqa: E402
from simulation import WheelchairSimulation  # noqa: E402
from controllers import controller_names  # noqa: E402

OUT_ROOT = os.path.join("output", "controller_graphs")
WHEELS = ["Front right (passive)", "Front left (passive)",
          "Rear right (driven)", "Rear left (driven)"]
AXES = ["x", "y", "z"]
PRETTY = {"pid": "Linear PID", "nonlinear_pid": "Nonlinear PID",
          "fuzzy": "Fuzzy", "smc": "Sliding Mode"}
COLORS = {"pid": "tab:blue", "nonlinear_pid": "tab:orange",
          "fuzzy": "tab:green", "smc": "tab:red"}


def _title(r, text):
    gm = "ML-adaptive" if r.get("ml_gain_scales") is not None else "fixed gains"
    return (f"{text}\n{PRETTY.get(r['controller_name'], r['controller_name'])} | {r['scenario']} | "
            f"grade {r['grade_deg'][0]:g}°, cross-slope {r['cross_slope_deg'][0]:g}° | {gm}")


def _save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=dpi)
    plt.close(fig)


def _components(r, key_xyz, ylabel, name, extra=()):
    """x/y/z components in three stacked panels (plus optional extra traces)."""
    t = r["time"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, r[key_xyz][:, i], lw=1.6, label=f"{name} {AXES[i]}")
        for lab, data in extra:
            if data.ndim == 2:
                ax.plot(t, data[:, i], "--", lw=1, label=lab)
        ax.set_ylabel(f"{AXES[i]} ({ylabel})"); ax.grid(True); ax.legend(fontsize=8, loc="best")
    axs[-1].set_xlabel("Time (s)")
    return fig, axs


def graphs_for_run(r, folder):
    t = r["time"]
    ctrl = r["controller_name"]

    # ---- 1. COMPONENT graphs ------------------------------------------------
    fig, axs = _components(r, "position_xyz", "m", "position")
    axs[0].set_title(_title(r, "Position components"))
    _save(fig, folder, "01_position_components.png")

    fig, axs = _components(r, "velocity_xyz", "m/s", "velocity")
    axs[0].plot(t, r["reference_speed"], "k:", label="reference speed (along grade)")
    axs[0].legend(fontsize=8)
    axs[0].set_title(_title(r, "Velocity components"))
    _save(fig, folder, "02_velocity_components.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r["velocity_forward"], lw=1.8, label="forward speed")
    ax.plot(t, r["reference_speed"], "k--", label="reference speed")
    ax.set(xlabel="Time (s)", ylabel="m/s", title=_title(r, "Speed tracking"))
    ax.grid(True); ax.legend()
    _save(fig, folder, "03_speed_tracking.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, lab in (("roll", "roll"), ("pitch", "pitch"), ("yaw", "yaw")):
        ax.plot(t, r[k], label=lab)
    ax.set(xlabel="Time (s)", ylabel="deg (relative to terrain)",
           title=_title(r, "Attitude components (roll / pitch / yaw)"))
    ax.grid(True); ax.legend()
    _save(fig, folder, "04_attitude_components.png")

    # ---- 2. ACCELERATION graphs ---------------------------------------------
    fig, axs = _components(r, "acceleration_xyz", "m/s²", "acceleration")
    axs[0].set_title(_title(r, "Acceleration components (world x, y, z)"))
    _save(fig, folder, "05_acceleration_components.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r["acceleration_forward"], label="forward (along grade)")
    ax.plot(t, r["acceleration_lateral"], label="lateral")
    ax.plot(t, r["acceleration"], "k", lw=1, alpha=.7, label="|a|")
    ax.set(xlabel="Time (s)", ylabel="m/s²",
           title=_title(r, "Forward / lateral / magnitude acceleration"))
    ax.grid(True); ax.legend()
    _save(fig, folder, "06_acceleration_forward_lateral.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r["angular_acceleration_body"])
    ax.set(xlabel="Time (s)", ylabel="rad/s²",
           title=_title(r, "Angular acceleration (body x, y, z)"))
    ax.legend(["roll axis", "pitch axis", "yaw axis"]); ax.grid(True)
    _save(fig, folder, "07_angular_acceleration.png")

    # ---- 3. FORCE graphs ----------------------------------------------------
    cf = r["contact_forces"]                       # (N, 4 wheels, 3 world components)
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axs):
        for w in range(4):
            ax.plot(t, cf[:, w, i], lw=1.2, label=WHEELS[w])
        ax.set_ylabel(f"F{AXES[i]} (N)"); ax.grid(True)
    axs[0].legend(fontsize=8, ncol=2)
    axs[0].set_title(_title(r, "Tyre/ground contact force per wheel (world x, y, z)"))
    axs[-1].set_xlabel("Time (s)")
    _save(fig, folder, "08_contact_force_per_wheel.png")

    total_cf = cf.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(3):
        ax.plot(t, total_cf[:, i], label=f"ΣF{AXES[i]}")
    ax.plot(t, np.linalg.norm(total_cf, axis=1), "k", lw=1.2, label="|ΣF|")
    ax.set(xlabel="Time (s)", ylabel="Force (N)",
           title=_title(r, "Total ground contact force"))
    ax.grid(True); ax.legend()
    _save(fig, folder, "09_total_contact_force.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r["normal_loads"], lw=1.3)
    ax.set(xlabel="Time (s)", ylabel="Normal load (N)",
           title=_title(r, "Wheel normal loads"))
    ax.legend(WHEELS, fontsize=8, ncol=2); ax.grid(True)
    _save(fig, folder, "10_wheel_normal_loads.png")

    drive_force = r["torque_wheels"].sum(axis=1) / wheel_radius
    cmd_force = r["command_torque"].sum(axis=1) / wheel_radius
    net_force = mass_total * r["acceleration_xyz"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, drive_force, label="actual drive force (Στorque / r)")
    axs[0].plot(t, cmd_force, "--", lw=1, label="commanded drive force")
    axs[0].set_ylabel("Force (N)"); axs[0].grid(True); axs[0].legend()
    axs[0].set_title(_title(r, "Drive force and applied disturbance force"))
    for i in range(3):
        axs[1].plot(t, r["disturbance_force_xyz"][:, i], label=f"disturbance F{AXES[i]}")
    axs[1].plot(t, r["disturbance_force"], "k", lw=1, label="|disturbance|")
    axs[1].set(xlabel="Time (s)", ylabel="Force (N)"); axs[1].grid(True); axs[1].legend(fontsize=8)
    _save(fig, folder, "11_drive_and_disturbance_force.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(3):
        ax.plot(t, net_force[:, i], label=f"m·a{AXES[i]}")
    ax.set(xlabel="Time (s)", ylabel="Net force (N)",
           title=_title(r, "Net force on chair (m_total · acceleration)"))
    ax.grid(True); ax.legend()
    _save(fig, folder, "12_net_force.png")

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t, r["torque_wheels"]); axs[0].set_ylabel("Wheel torque (Nm)")
    axs[0].legend(WHEELS, fontsize=8, ncol=2); axs[0].grid(True)
    axs[0].set_title(_title(r, "Wheel torque and wheel speed"))
    axs[1].plot(t, r["wheel_speed"]); axs[1].set(xlabel="Time (s)", ylabel="Wheel speed (rad/s)")
    axs[1].grid(True)
    _save(fig, folder, "13_wheel_torque_and_speed.png")

    # ---- 4. one-page summary for the report ---------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axs[0, 0].plot(t, r["velocity_forward"], label="forward speed")
    axs[0, 0].plot(t, r["reference_speed"], "k--", label="reference")
    axs[0, 0].set_title("Speed (m/s)"); axs[0, 0].legend(fontsize=8)
    for i in range(3):
        axs[0, 1].plot(t, r["velocity_xyz"][:, i], label=f"v{AXES[i]}")
        axs[1, 0].plot(t, r["acceleration_xyz"][:, i], label=f"a{AXES[i]}")
        axs[1, 1].plot(t, total_cf[:, i], label=f"ΣF{AXES[i]}")
    axs[0, 1].set_title("Velocity components (m/s)")
    axs[1, 0].set_title("Acceleration components (m/s²)")
    axs[1, 1].set_title("Total contact force components (N)")
    axs[2, 0].plot(t, drive_force, label="drive force")
    axs[2, 0].plot(t, r["disturbance_force"], label="|disturbance|")
    axs[2, 0].set_title("Drive / disturbance force (N)")
    axs[2, 1].plot(t, r["roll"], label="roll"); axs[2, 1].plot(t, r["pitch"], label="pitch")
    axs[2, 1].set_title("Roll / pitch (deg)")
    for ax in axs.flat:
        ax.grid(True); ax.legend(fontsize=8)
    axs[2, 0].set_xlabel("Time (s)"); axs[2, 1].set_xlabel("Time (s)")
    fig.suptitle(_title(r, "Summary"), fontsize=12)
    _save(fig, folder, "00_summary.png")


def comparison_graphs(runs, folder):
    """Same quantities, all controllers overlaid (for the comparison slide)."""
    any_r = next(iter(runs.values()))
    t = any_r["time"]

    def overlay(fn, ylabel, title, name, ref=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        for c, r in runs.items():
            ax.plot(r["time"], fn(r), color=COLORS.get(c), lw=1.5, label=PRETTY.get(c, c))
        if ref is not None:
            ax.plot(t, ref, "k--", lw=1, label="reference")
        ax.set(xlabel="Time (s)", ylabel=ylabel,
               title=f"{title}\n{any_r['scenario']} | grade {any_r['grade_deg'][0]:g}°, "
                     f"cross-slope {any_r['cross_slope_deg'][0]:g}°")
        ax.grid(True); ax.legend()
        _save(fig, folder, name)

    overlay(lambda r: r["velocity_forward"], "m/s", "Forward speed – all controllers",
            "cmp_speed.png", any_r["reference_speed"])
    overlay(lambda r: r["acceleration_forward"], "m/s²", "Forward acceleration – all controllers",
            "cmp_acceleration_forward.png")
    overlay(lambda r: r["acceleration_lateral"], "m/s²", "Lateral acceleration – all controllers",
            "cmp_acceleration_lateral.png")
    overlay(lambda r: r["acceleration"], "m/s²", "|Acceleration| – all controllers",
            "cmp_acceleration_magnitude.png")
    overlay(lambda r: r["torque_wheels"].sum(axis=1) / wheel_radius, "N",
            "Drive force – all controllers", "cmp_drive_force.png")
    overlay(lambda r: np.linalg.norm(r["contact_forces"].sum(axis=1), axis=1), "N",
            "Total ground contact force |ΣF| – all controllers", "cmp_contact_force.png")
    overlay(lambda r: r["pitch"], "deg", "Pitch relative to terrain – all controllers",
            "cmp_pitch.png")
    overlay(lambda r: r["roll"], "deg", "Roll relative to terrain – all controllers",
            "cmp_roll.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", nargs="+", default=["smooth"],
                    choices=["smooth", "rough", "sudden", "repeated"])
    ap.add_argument("--controllers", nargs="+", default=list(controller_names()))
    ap.add_argument("--slope", type=float, default=5.0)
    ap.add_argument("--cross", type=float, default=0.0)
    ap.add_argument("--ml-adaptive", action="store_true")
    a = ap.parse_args()

    for scenario in a.scenario:
        runs = {}
        for c in a.controllers:
            print(f"[{scenario}] simulating {c} ...", flush=True)
            sim = WheelchairSimulation(c, slope=a.slope, scenario=scenario,
                                       cross_slope=a.cross, ml_adaptive=a.ml_adaptive)
            r = sim.run()
            runs[c] = r
            graphs_for_run(r, os.path.join(OUT_ROOT, scenario, c))
        comparison_graphs(runs, os.path.join(OUT_ROOT, scenario, "all_controllers"))
    print("Done. Graphs in", os.path.join(HERE, OUT_ROOT))


if __name__ == "__main__":
    main()
