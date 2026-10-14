"""Fixed-step simulation of the full 3-D wheelchair model."""
import csv
import numpy as np
from pathlib import Path

from parameters import *
from dynamics import *
from controllers import *
from spatial_stability import assess_spatial_stability

SCENARIO_DESCRIPTIONS = {
    "smooth": "planar terrain with no scenario-specific disturbance",
    "rough": "planar terrain with a sinusoidal longitudinal force disturbance",
    "sudden": "planar terrain with a longitudinal force pulse from 5.0 to 5.5 s",
    "repeated": "planar terrain with longitudinal force pulses at 4, 7, 10, 13, and 16 s",
}


class WheelchairSimulation:
    def __init__(self, controller_name="pid", slope=0, scenario="smooth",
                 cross_slope=0.0, lateral_acceleration=0.0, ml_adaptive=False,
                 duration=None, time_step=None):
        self.controller_name = controller_name
        self.ml_adaptive = bool(ml_adaptive)
        self.controller = create_controller(controller_name, ml_adaptive=self.ml_adaptive)
        self.slope = float(slope)
        self.scenario = scenario
        self.cross_slope = float(cross_slope)
        self.lateral_acceleration = float(lateral_acceleration)
        self.duration = float(simulation_time if duration is None else duration)
        self.time_step = float(dt if time_step is None else time_step)
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be a positive finite number")
        if not np.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be a positive finite number")
        if not np.isfinite(self.slope) or abs(self.slope) > max_slope:
            raise ValueError(f"slope must be finite and within +/-{max_slope:g} deg")
        if not np.isfinite(self.cross_slope) or abs(self.cross_slope) > max_cross_slope:
            raise ValueError(f"cross_slope must be finite and within +/-{max_cross_slope:g} deg")
        if not np.isfinite(self.lateral_acceleration):
            raise ValueError("lateral_acceleration must be finite")
        if self.scenario not in SCENARIO_DESCRIPTIONS:
            raise ValueError(f"Unknown disturbance scenario {self.scenario!r}; choose from {tuple(SCENARIO_DESCRIPTIONS)}")
        self.results = {}

    def initial_state_3d(self):
        n, ex, ey = terrain(self.slope, self.cross_slope)
        # Body frame aligned with the local terrain tangent plane.
        R = np.column_stack((ex, ey, n))
        q = rotation_matrix_to_quaternion(R)
        normal_weight = mass_total*g*n[2]
        compression = normal_weight/(4.0*contact_stiffness)
        p = (cog_height - compression)*n
        return np.r_[p, np.zeros(3), q, np.zeros(3), np.zeros(4), np.zeros(4)]

    def disturbance_force(self, t):
        n, ex, ey = terrain(self.slope, self.cross_slope)
        normal = mass_total*g*n[2]
        scale = 0.05*mu_static*normal
        lateral = mass_total*self.lateral_acceleration*ey
        if self.scenario == "smooth": return lateral
        if self.scenario == "rough": return lateral + scale*np.sin(2*np.pi*1.5*t)*ex
        if self.scenario == "sudden": return lateral + (scale*ex if 5.0 <= t < 5.5 else np.zeros(3))
        if self.scenario == "repeated": return lateral + (scale*ex if any(k <= t < k+0.25 for k in (4,7,10,13,16)) else np.zeros(3))
        raise ValueError(f"Unknown disturbance scenario: {self.scenario}")

    def run(self):
        # This is the actual plant integration grid. Every state and actuator
        # update below uses the interval between adjacent samples.
        n_steps = int(np.ceil(self.duration / self.time_step))
        t = np.minimum(np.arange(n_steps + 1, dtype=float) * self.time_step,
                       self.duration)
        t[-1] = self.duration
        states = np.zeros((len(t), 21))
        states[0] = self.initial_state_3d()
        normal_axis, forward_axis, lateral_axis = terrain(self.slope, self.cross_slope)
        terrain_rotation = np.column_stack((forward_axis, lateral_axis, normal_axis))
        torque = np.zeros((len(t), 4))
        command_torque = np.zeros((len(t), 4))
        disturbance = np.zeros((len(t), 3))
        acceleration = np.zeros((len(t), 3))
        angular_accel = np.zeros((len(t), 3))
        normals = np.zeros((len(t), 4))
        slips = np.zeros((len(t), 4))
        longitudinal_slips = np.zeros((len(t), 4))
        lateral_slips = np.zeros((len(t), 4))
        contact_forces_hist = np.zeros((len(t), 4, 3))
        governor_feasible = np.ones(len(t), dtype=bool)
        terminated = np.zeros(len(t), dtype=bool)
        termination_reason = ""
        # Start with the drive holding the chair against the grade (like a
        # parking brake), otherwise it rolls backwards for the first 0.2 s.
        hold = mass_total*g*np.sin(np.deg2rad(self.slope))*wheel_radius/max(np.sum(driven_wheels), 1)
        actuator_torque_state = np.where(driven_wheels, hold, 0.0)

        for i in range(len(t)-1):
            step_dt = float(t[i + 1] - t[i])
            s = states[i]
            p, v = s[:3], s[3:6]
            q = quat_normalize(s[6:10])
            w = s[10:13]
            # Express attitude relative to the local ground plane; a steady
            # uphill pose must not be reported as an unsafe body pitch.
            relative_rotation = terrain_rotation.T @ quat_to_matrix(q)
            euler = euler_from_quaternion(rotation_matrix_to_quaternion(relative_rotation))
            # Stop integration after a gross loss of model validity. This is
            # an event flag, not a claim that 80 deg is a physical safety limit.
            if (not np.all(np.isfinite(s))) or np.max(np.abs(euler[:2])) > np.deg2rad(80.0) or np.linalg.norm(p) > 1e3:
                terminated[i:] = True
                governor_feasible[i:] = False
                termination_reason = "integration stopped: gross attitude/state excursion"
                states[i+1:] = s
                torque[i:] = actuator_torque_state
                command_torque[i:] = actuator_torque_state
                acceleration[i:] = 0.0
                angular_accel[i:] = 0.0
                normals[i:] = 0.0
                slips[i:] = 0.0
                longitudinal_slips[i:] = 0.0
                lateral_slips[i:] = 0.0
                contact_forces_hist[i:] = 0.0
                disturbance[i:] = disturbance[i-1] if i > 0 else 0.0
                break
            disturbance[i] = self.disturbance_force(t[i])
            requested = self.controller.command_3d(
                t[i], p, v, euler, w, 0.0, self.slope, self.cross_slope)
            requested = np.asarray(requested, dtype=float)
            # Motor-side aggregate torque is divided over the two assumed drive wheels.
            per_wheel_limit = (motor_torque_max*gear_ratio*motor_efficiency) / max(np.sum(driven_wheels),1)
            requested = np.clip(requested, -per_wheel_limit, per_wheel_limit)
            requested[~driven_wheels] = 0.0

            # Limit requested torque against the configured envelope. This
            # command governor cannot guarantee the lagged plant stays inside it.
            speed = np.linalg.norm(v)
            v_fwd = float(np.dot(v, forward_axis))
            if v_fwd >= max_velocity:
                # Permit braking torque while preventing further positive drive.
                requested = np.minimum(requested, 0.0)
            elif v_fwd <= -max_velocity:
                requested = np.maximum(requested, 0.0)

            # Bound both positive and negative longitudinal acceleration.
            # Gravity, rolling resistance and drag are included in the force
            # interval, so a downhill case can command braking force.
            normal_ground = mass_total*g*normal_axis[2]
            grade_force = mass_total*g*np.dot(np.array([0.0, 0.0, -1.0]), forward_axis)
            rolling_force = rolling_resistance*normal_ground*np.tanh(v_fwd/0.1)
            drag_force_fwd = 0.5*rho_air*drag_coefficient*frontal_area*v_fwd*abs(v_fwd)
            resistance_force = -grade_force + rolling_force + drag_force_fwd
            accel_force = effective_mass()*max_acceleration
            min_drive_force = resistance_force - accel_force
            max_drive_force = resistance_force + accel_force
            requested_drive_force = float(np.sum(requested[driven_wheels]) / max(wheel_radius, 1e-9))
            actuator_force_limit = per_wheel_limit * np.sum(driven_wheels) / max(wheel_radius, 1e-9)
            feasible_low = max(min_drive_force, -actuator_force_limit)
            feasible_high = min(max_drive_force, actuator_force_limit)
            if feasible_low <= feasible_high:
                limited_drive_force = float(np.clip(
                    requested_drive_force, feasible_low, feasible_high))
                governor_feasible[i] = True
            else:
                # The available drive cannot satisfy the requested accel
                # envelope on this terrain; clamp to hardware limits and flag.
                limited_drive_force = float(np.clip(
                    requested_drive_force, -actuator_force_limit, actuator_force_limit))
                governor_feasible[i] = False
            if abs(requested_drive_force) > 1e-9:
                requested[driven_wheels] *= limited_drive_force / requested_drive_force
            elif np.any(driven_wheels):
                requested[driven_wheels] = (
                    limited_drive_force * wheel_radius / np.sum(driven_wheels))

            # First-order drivetrain response: mechanical actuator abstraction.
            alpha_m = min(1.0, step_dt/max(motor_time_constant, step_dt))
            actuator_torque_state += alpha_m*(requested-actuator_torque_state)
            actuator_torque_state[~driven_wheels] = 0.0
            command_torque[i] = requested
            torque[i] = actuator_torque_state

            deriv, aux = state_derivative(
                s, torque[i], self.slope, self.cross_slope, disturbance[i],
                step_duration=step_dt)
            acceleration[i] = deriv[3:6]
            angular_accel[i] = deriv[10:13]
            normals[i] = aux["normal_loads"]
            slips[i] = aux["slip_speeds"]
            longitudinal_slips[i] = aux["longitudinal_slip_speeds"]
            lateral_slips[i] = aux["lateral_slip_speeds"]
            contact_forces_hist[i] = aux["forces"]

            # Semi-implicit translational/angular update, quaternion explicit update.
            ns = s.copy()
            ns[3:6] = s[3:6] + acceleration[i]*step_dt
            ns[0:3] = s[0:3] + ns[3:6]*step_dt
            ns[10:13] = s[10:13] + angular_accel[i]*step_dt
            ns[6:10] = quat_normalize(s[6:10] + quat_derivative(s[6:10], ns[10:13])*step_dt)
            ns[17:21] = s[17:21] + deriv[17:21]*step_dt
            ns[13:17] = s[13:17] + ns[17:21]*step_dt
            states[i+1] = ns

        disturbance[-1] = disturbance[-2]
        governor_feasible[-1] = governor_feasible[-2]
        torque[-1] = torque[-2]
        command_torque[-1] = command_torque[-2]
        acceleration[-1] = acceleration[-2]
        angular_accel[-1] = angular_accel[-2]
        normals[-1] = normals[-2]
        slips[-1] = slips[-2]
        longitudinal_slips[-1] = longitudinal_slips[-2]
        lateral_slips[-1] = lateral_slips[-2]
        contact_forces_hist[-1] = contact_forces_hist[-2]

        pos = states[:,0:3]
        vel = states[:,3:6]
        eulers = np.array([
            euler_from_quaternion(rotation_matrix_to_quaternion(
                terrain_rotation.T @ quat_to_matrix(q)))
            for q in states[:,6:10]])
        wheel_angles = states[:,13:17]
        wheel_speeds = states[:,17:21]
        speed = np.linalg.norm(vel, axis=1)
        yaw_rate = np.array([
            abs((quat_to_matrix(q) @ w)[2])
            for q, w in zip(states[:,6:10], states[:,10:13])])
        drive_torque = np.sum(torque, axis=1)
        kinetic = np.array([
            kinetic_energy(v, w, wheel_speed_row)
            for v, w, wheel_speed_row in zip(vel, states[:,10:13], wheel_speeds)])
        potential = mass_total*g*pos[:,2]
        contact_elastic = np.array([
            wheel_contact_elastic_energy(p_i, q_i, self.slope, self.cross_slope)
            for p_i, q_i in zip(pos, states[:,6:10])])
        attitude_elastic = np.array([
            attitude_elastic_energy(q_i, self.slope, self.cross_slope)
            for q_i in states[:,6:10]])
        total = kinetic + potential + contact_elastic + attitude_elastic
        mech_power = np.sum(torque*wheel_speeds, axis=1)

        if not termination_reason:
            termination_reason = "none"

        forward_position = (pos - pos[0]) @ forward_axis
        forward_velocity = vel @ forward_axis
        forward_acceleration = acceleration @ forward_axis
        lateral_acceleration = acceleration @ lateral_axis
        spatial = assess_spatial_stability(
            self.slope, self.cross_slope, forward_acceleration,
            lateral_acceleration, drive_torque)
        support_safe = spatial["support_polygon_screen_safe"]
        traction_safe = spatial["combined_traction_screen_safe"]
        screen_state = np.where(~support_safe, "support polygon screen failed",
                        np.where(~traction_safe, "traction screen failed", "screened stable"))

        self.results = {
            "controller_name": self.controller_name,
            "scenario": self.scenario,
            "time": t,
            "state": states,
            "position_xyz": pos,
            "velocity_xyz": vel,
            "position": forward_position,
            "velocity": speed,
            "velocity_forward": forward_velocity,
            "velocity_x": vel[:,0],
            "velocity_y": vel[:,1],
            "velocity_z": vel[:,2],
            "acceleration_xyz": acceleration,
            "acceleration": np.linalg.norm(acceleration, axis=1),
            "acceleration_x": acceleration[:,0],
            "acceleration_y": acceleration[:,1],
            "acceleration_forward": forward_acceleration,
            "acceleration_lateral": lateral_acceleration,
            "acceleration_z": acceleration[:,2],
            "angular_velocity_body": states[:,10:13],
            "angular_acceleration_body": angular_accel,
            "roll": np.rad2deg(eulers[:,0]),
            "pitch": np.rad2deg(eulers[:,1]),
            "yaw": np.rad2deg(eulers[:,2]),
            "wheel_angle": wheel_angles,
            "wheel_speed": wheel_speeds,
            "omega": np.mean(wheel_speeds, axis=1),
            "torque_wheels": torque,
            "torque": drive_torque,
            "command_torque": command_torque,
            "disturbance_force_xyz": disturbance,
            "disturbance_force": np.linalg.norm(disturbance, axis=1),
            "normal_loads": normals,
            "slip_speeds": slips,
            "longitudinal_slip_speeds": longitudinal_slips,
            "lateral_slip_speeds": lateral_slips,
            "contact_forces": contact_forces_hist,
            "kinetic_energy": kinetic,
            "potential_energy": potential,
            "contact_elastic_energy": contact_elastic,
            "attitude_elastic_energy": attitude_elastic,
            "total_energy": total,
            "power": mech_power,
            # These are quasi-static planar support/traction screens, not a
            # dynamic stability certificate or a single slope-only index.
            "stability_margin": spatial["support_margin_min"],
            "stability_state": screen_state,
            "reference": np.array([reference_position(tt) for tt in t]),
            "reference_speed": np.array([reference_speed(tt) for tt in t]),
            "safety_speed_ok": speed <= max_velocity,
            "safety_reverse_motion_ok": (forward_velocity >= -0.05)
                | (np.array([reference_speed(tt) for tt in t]) <= 0.05),
            "safety_accel_ok": np.abs(forward_acceleration) <= max_acceleration + 1e-6,
            "safety_lateral_accel_ok": np.abs(lateral_acceleration) <= max_lateral_acceleration + 1e-6,
            "safety_roll_ok": np.abs(eulers[:,0]) <= np.deg2rad(max_roll_deg),
            "safety_pitch_ok": np.abs(eulers[:,1]) <= np.deg2rad(max_pitch_deg),
            "safety_yaw_rate_ok": yaw_rate <= max_yaw_rate,
            "safety_traction_ok": traction_safe,
            "safety_support_ok": support_safe,
            "safety_governor_feasible": governor_feasible,
            "acceleration_envelope_actuator_feasible": governor_feasible,
            "simulation_terminated": terminated,
            "termination_reason": termination_reason,
            "scenario_description": SCENARIO_DESCRIPTIONS[self.scenario],
            "terrain_description": "planar grade/cross-slope; disturbances are applied forces, not geometric roughness",
            "lateral_acceleration_disturbance_mps2": self.lateral_acceleration,
            "grade_deg": np.full_like(t, self.slope),
            "cross_slope_deg": np.full_like(t, self.cross_slope),
            **spatial,
        }
        if self.ml_adaptive and self.controller.gain_history:
            gain_array = np.asarray(self.controller.gain_history, dtype=float)
            self.results["ml_gain_time"] = gain_array[:, 0]
            self.results["ml_gain_scales"] = gain_array[:, 1:]
            self.results["ml_gain_names"] = [
                "pid_kp_scale", "pid_ki_scale", "pid_kd_scale",
                "smc_lambda_scale", "smc_k_scale", "smc_phi_scale",
                "fuzzy_output_scale"]
            self.results["ml_model_source"] = self.controller.ml_gain_tuner.model_source
            self.results["ml_state"] = np.asarray(self.controller.state_history, dtype=float)[:, 1:]
        return self.results

    def export_csv(self, filename="simulation_3d.csv"):
        """Export sampled signals plus context metadata in a flat CSV."""
        r = self.results
        columns = {
            "time_s": r["time"], "position_forward_m": r["position"],
            "reference_position_m": r["reference"],
            "speed_magnitude_mps": r["velocity"],
            "velocity_forward_mps": r["velocity_forward"],
            "reference_speed_mps": r["reference_speed"],
            "acceleration_forward_mps2": r["acceleration_forward"],
            "acceleration_lateral_mps2": r["acceleration_lateral"],
            "roll_relative_deg": r["roll"], "pitch_relative_deg": r["pitch"],
            "yaw_relative_deg": r["yaw"], "power_mechanical_w": r["power"],
            "kinetic_energy_j": r["kinetic_energy"],
            "potential_energy_j": r["potential_energy"],
            "contact_elastic_energy_j": r["contact_elastic_energy"],
            "attitude_elastic_energy_j": r["attitude_elastic_energy"],
            "total_mechanical_energy_j": r["total_energy"],
            "support_margin_m": r["support_margin_min"],
            "friction_utilization": r["friction_utilization_3d"],
            "cop_x_m": r["cop_x"], "cop_y_m": r["cop_y"],
            "safety_speed_ok": r["safety_speed_ok"],
            "safety_accel_ok": r["safety_accel_ok"],
            "safety_lateral_accel_ok": r["safety_lateral_accel_ok"],
            "safety_reverse_motion_ok": r["safety_reverse_motion_ok"],
            "safety_roll_ok": r["safety_roll_ok"],
            "safety_pitch_ok": r["safety_pitch_ok"],
            "safety_yaw_rate_ok": r["safety_yaw_rate_ok"],
            "safety_support_ok": r["safety_support_ok"],
            "safety_traction_ok": r["safety_traction_ok"],
            "governor_feasible": r["safety_governor_feasible"],
            "simulation_terminated": r["simulation_terminated"],
        }
        for wheel in range(4):
            columns[f"wheel_angular_speed_{wheel}_radps"] = r["wheel_speed"][:, wheel]
            columns[f"wheel_torque_{wheel}_nm"] = r["torque_wheels"][:, wheel]
            columns[f"wheel_command_torque_{wheel}_nm"] = r["command_torque"][:, wheel]
            columns[f"normal_load_{wheel}_n"] = r["normal_loads"][:, wheel]
            columns[f"slip_speed_{wheel}_mps"] = r["slip_speeds"][:, wheel]
            columns[f"longitudinal_slip_{wheel}_mps"] = r["longitudinal_slip_speeds"][:, wheel]
            columns[f"lateral_slip_{wheel}_mps"] = r["lateral_slip_speeds"][:, wheel]
            for axis, axis_name in enumerate(("x", "y", "z")):
                columns[f"contact_force_{axis_name}_{wheel}_n"] = r["contact_forces"][:, wheel, axis]

        metadata = ["controller_name", "scenario", "grade_deg", "cross_slope_deg",
                    "lateral_acceleration_disturbance_mps2", "terrain_description",
                    "scenario_description", "termination_reason"]
        csv_path = Path(__file__).resolve().parent / CSV_DIR / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([*columns, *metadata])
            n_rows = len(r["time"])
            context = [self.controller_name, self.scenario, self.slope,
                       self.cross_slope, self.lateral_acceleration,
                       r["terrain_description"], r["scenario_description"],
                       r["termination_reason"]]
            for index in range(n_rows):
                writer.writerow([values[index] for values in columns.values()] + context)
        print(f"CSV saved to {csv_path}.")
        if "ml_gain_scales" in r:
            tag = f"{self.controller_name}_{self.scenario}_g{self.slope:g}_x{self.cross_slope:g}"
            trace_path = csv_path.parent / f"ml_gain_trace_{tag}.csv"
            # Only the gains this controller actually uses (the other columns are
            # always 1.0 and were misleading), plus the model inputs that drive them.
            used = {"pid": ("pid_kp_scale", "pid_ki_scale", "pid_kd_scale"),
                    "nonlinear_pid": ("pid_kp_scale", "pid_ki_scale", "pid_kd_scale"),
                    "smc": ("smc_lambda_scale", "smc_k_scale", "smc_phi_scale"),
                    "fuzzy": ("fuzzy_output_scale",)}[self.controller_name]
            idx = [r["ml_gain_names"].index(n) for n in used]
            state_names = ["speed_error_mps", "terrain_grade_deg", "roll_deg", "pitch_deg",
                           "lateral_speed_mps", "accel_mps2"]
            with trace_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s", *state_names, *used])
                for k in range(len(r["ml_gain_time"])):
                    writer.writerow([f"{r['ml_gain_time'][k]:.2f}",
                                     *[f"{v:.5f}" for v in r["ml_state"][k]],
                                     *[f"{r['ml_gain_scales'][k, i]:.5f}" for i in idx]])
            print(f"ML gain trace saved to {trace_path} ({r.get('ml_model_source', 'unknown model source')}).")

    def summary(self):
        r=self.results
        print(f"Scenario input: {r['scenario_description']}")
        print(f"Terrain input: {r['terrain_description']}")
        if self.lateral_acceleration:
            print(f"Applied lateral acceleration disturbance: {self.lateral_acceleration:g} m/s^2")
        print(f"\nController: {self.controller_name} | grade: {self.slope:g} deg | cross-slope: {self.cross_slope:g} deg | scenario: {self.scenario}")
        print(f"Final XYZ: {r['position_xyz'][-1]} m | peak speed: {np.max(r['velocity']):.3f} m/s")
        print(f"Peak |acceleration|: {np.max(r['acceleration']):.3f} m/s² | peak |torque| sum: {np.max(np.abs(r['torque'])):.2f} Nm")
        print(f"Peak roll/pitch: {np.max(np.abs(r['roll'])):.2f}/{np.max(np.abs(r['pitch'])):.2f} deg")
        print(f"Peak tire slip: {np.max(r['slip_speeds']):.3f} m/s | minimum forward speed: {np.min(r['velocity_forward']):.3f} m/s")
        print("Screens passed: " + ", ".join(
            f"{name}={bool(np.all(r[key]))}" for key, name in (
                ("safety_speed_ok", "speed"), ("safety_reverse_motion_ok", "no reverse"),
                ("safety_accel_ok", "forward accel"),
                ("safety_lateral_accel_ok", "lateral accel"),
                ("safety_roll_ok", "roll"), ("safety_pitch_ok", "pitch"),
                ("safety_yaw_rate_ok", "yaw rate"), ("safety_traction_ok", "traction"),
                ("safety_support_ok", "support"),
                ("safety_governor_feasible", "actuator envelope"))))
        if self.ml_adaptive:
            source = self.controller.ml_gain_tuner.model_source
            print(f"Adaptive gain scheduling: ON ({source})")
            if "rule-based fallback" in source:
                print("WARNING: this run is not using a trained ML model; run optimize_and_train_ml.py first.")
        else:
            print("ML adaptive gain scheduling: OFF (fixed nominal gains)")
        print("3-D dynamics are simulated; contact/drivetrain/material parameters remain provisional.")


def rotation_matrix_to_quaternion(R):
    tr=np.trace(R)
    if tr>0:
        S=np.sqrt(tr+1.0)*2; w=0.25*S; x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/S; x=.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/S; x=(R[0,1]+R[1,0])/S; y=.25*S; z=(R[1,2]+R[2,1])/S
    else:
        S=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/S; x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=.25*S
    return quat_normalize(np.array([w,x,y,z]))


def run_all_slopes(controller="pid", ml_adaptive=False):
    return {angle: WheelchairSimulation(controller_name=controller, slope=angle,
                                        ml_adaptive=ml_adaptive).run()
            for angle in terrain_angles}


def compare_controllers(slope=10, scenario="smooth", cross_slope=0.0,
                        ml_adaptive=False, duration=None, time_step=None):
    out = {}
    for ctrl in controller_names():
        print(f"  simulating {ctrl} ({scenario}) ...", flush=True)
        out[ctrl] = WheelchairSimulation(
            ctrl, slope, scenario, cross_slope, ml_adaptive=ml_adaptive,
            duration=duration, time_step=time_step).run()
    return out


def compare_scenarios(slope=10, cross_slope=0.0, ml_adaptive=False,
                      duration=None, time_step=None):
    out = {}
    scenarios = ("smooth", "rough", "sudden", "repeated")
    for k, scenario in enumerate(scenarios, 1):
        print(f"Scenario {k}/{len(scenarios)}: {scenario}", flush=True)
        out[scenario] = compare_controllers(
            slope, scenario, cross_slope, ml_adaptive=ml_adaptive,
            duration=duration, time_step=time_step)
    return out