"""3-D rigid-body, wheel-contact and drivetrain dynamics.

Coordinates: world z is upward. The wheelchair body frame is x-forward,
y-left, z-up. Terrain is a planar grade/cross-slope. The plant has 6-DOF
rigid-body motion, four wheel spin states, compliant normal contacts, Coulomb
friction, and a first-order motor/gear actuator. It is still a research model:
there is no suspension, tire deformation model, battery/electrical model,
occupant articulation, or experimentally identified tire law.
"""
import numpy as np
from parameters import *

WHEEL_NAMES = ("front_right", "front_left", "rear_right", "rear_left")
WHEEL_SIGNS = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]], dtype=float)


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else initial_quaternion.copy()


def quat_to_matrix(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quat_derivative(q, omega_body):
    w, x, y, z = q
    p, qv, r = omega_body
    return 0.5 * np.array([
        -x*p - y*qv - z*r,
         w*p + y*r - z*qv,
         w*qv - x*r + z*p,
         w*r + x*qv - y*p,
    ])


def euler_from_quaternion(q):
    w, x, y, z = quat_normalize(q)
    pitch_arg = np.clip(2*(w*y-z*x), -1.0, 1.0)
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.array([roll, pitch, yaw])


def terrain(grade_deg, cross_slope_deg):
    ag = np.deg2rad(grade_deg)
    ac = np.deg2rad(cross_slope_deg)
    normal = np.array([-np.tan(ag), -np.tan(ac), 1.0])
    normal /= np.linalg.norm(normal)
    e_x = np.array([1.0, 0.0, np.tan(ag)])
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(normal, e_x)
    e_y /= np.linalg.norm(e_y)
    return normal, e_x, e_y


def wheel_body_positions():
    # Body origin is the occupied CoG; mixed-size caster and drive wheels have
    # distinct axle heights while sharing the assumed support plane.
    z = -cog_height + wheel_radii
    return np.array([
        [ wheel_base/2, -track_width/2, z[0]],
        [ wheel_base/2,  track_width/2, z[1]],
        [-wheel_base/2, -track_width/2, z[2]],
        [-wheel_base/2,  track_width/2, z[3]],
    ], dtype=float)


def effective_mass():
    return mass_total + float(np.sum(wheel_inertias / wheel_radii**2))


def mass_matrix():
    """Generalized rigid-body mass matrix for [v, omega]."""
    return np.block([[mass_total*np.eye(3), np.zeros((3,3))],
                     [np.zeros((3,3)), body_inertia]])


def gravity_force(theta_deg):
    return mass_total * g * np.sin(np.deg2rad(theta_deg))


def normal_force(theta_deg):
    return mass_total * g * np.cos(np.deg2rad(theta_deg))


def rolling_force(theta_deg=0.0):
    return rolling_resistance * normal_force(theta_deg)


def drag_force(speed):
    """Scalar aerodynamic drag force magnitude (N), signed with speed."""
    return 0.5 * rho_air * drag_coefficient * frontal_area * speed * abs(speed)


def drag_force_vector(v_world):
    speed = np.linalg.norm(v_world)
    return -0.5 * rho_air * drag_coefficient * frontal_area * speed * v_world


def motor_torque(throttle):
    return np.clip(throttle, -1.0, 1.0) * motor_torque_max * gear_ratio * motor_efficiency


def wheel_force(torque):
    return torque / wheel_radius


def body_wheel_centers(position, quaternion):
    R = quat_to_matrix(quaternion)
    return position[None, :] + wheel_body_positions() @ R.T


def contact_forces(position, velocity, quaternion, omega_body, wheel_speeds,
                   wheel_drive_torques, grade_deg, cross_slope_deg,
                   external_force=None, step_duration=dt):
    """Return four contact forces, torques on body, normal loads and slips."""
    external_force = np.zeros(3) if external_force is None else np.asarray(external_force, dtype=float)
    if external_force.shape != (3,) or not np.all(np.isfinite(external_force)):
        raise ValueError("external_force must be a finite XYZ force vector")
    if not np.isfinite(step_duration) or step_duration <= 0.0:
        raise ValueError("step_duration must be positive and finite")
    n, t_x, t_y = terrain(grade_deg, cross_slope_deg)
    R = quat_to_matrix(quaternion)
    inverse_inertia_world = R @ np.linalg.solve(body_inertia, np.eye(3)) @ R.T
    centers = body_wheel_centers(position, quaternion)
    forces = np.zeros((4, 3))
    normals = np.zeros(4)
    slips = np.zeros(4)
    longitudinal_slips = np.zeros(4)
    lateral_slips = np.zeros(4)
    rolling_losses = np.zeros(4)
    body_torque = np.zeros(3)

    for i, center in enumerate(centers):
        signed_height = np.dot(center, n)  # terrain passes through origin
        radius = wheel_radii[i]
        penetration = max(0.0, radius - signed_height)
        speed_ratio = min(abs(wheel_speeds[i]) / max(wheel_speed_max, 1e-9), 1.0)
        torque_factor = 1.0 if speed_ratio <= 0.8 else max(0.0, 1.0 - (speed_ratio-0.8)/0.2)
        available_drive_torque = wheel_drive_torques[i] * torque_factor
        axle = R @ np.array([0.0, 1.0, 0.0])
        # Drive motor torque is internal to the wheel/chassis pair; apply the
        # equal-and-opposite stator reaction moment to the chassis.
        body_torque -= available_drive_torque * axle
        # Rigid-body velocity at center; wheel spin adds velocity at contact.
        r_contact = -radius * n
        wheel_center_velocity = velocity + np.cross(R @ omega_body, R @ wheel_body_positions()[i])
        contact_velocity = wheel_center_velocity + np.cross(wheel_speeds[i] * axle, r_contact)
        vt = contact_velocity - np.dot(contact_velocity, n) * n

        if penetration > 0.0:
            normal = max(0.0, contact_stiffness*penetration - contact_damping*np.dot(contact_velocity, n))
            vt_norm = np.linalg.norm(vt)
            friction_limit = mu_dynamic * normal
            # Resolve the actual tire slip into rolling and lateral directions.
            # Positive wheel spin about +axle cancels positive forward speed
            # at the contact point under this model's x-forward/y-left/z-up
            # convention. Tire force must oppose contact slip; it must not be
            # assigned directly as motor_torque/r, which bypasses wheel-spin
            # dynamics and creates artificial slip.
            drive_dir = np.cross(axle, n)
            drive_norm = np.linalg.norm(drive_dir)
            if drive_norm <= 1e-9:
                drive_dir = t_x - np.dot(t_x, n) * n
                drive_norm = np.linalg.norm(drive_dir)
            drive_dir = drive_dir / max(drive_norm, 1e-9)
            lat_dir = axle - np.dot(axle, n) * n
            lat_norm = np.linalg.norm(lat_dir)
            lat_dir = lat_dir / max(lat_norm, 1e-9)
            slip_long = float(np.dot(vt, drive_dir))
            slip_lateral = float(np.dot(vt, lat_dir))
            longitudinal_slips[i] = slip_long
            lateral_slips[i] = slip_lateral
            center_tangent = wheel_center_velocity - np.dot(wheel_center_velocity, n) * n
            center_speed_long = float(np.dot(center_tangent, drive_dir))
            rolling_force = (-rolling_resistance * normal
                             * np.tanh(center_speed_long / 0.02) * drive_dir)

            # Backward-Euler equivalent damping for the slip response. The
            # raw tire coefficients imply a sub-millisecond relaxation time;
            # applying them explicitly at the project time step makes wheel
            # speed and chassis acceleration alternate and diverge. This
            # effective coefficient is stable for the supplied step size and
            # uses the contact's translational, rotational, and wheel-spin
            # effective mass along each tangent direction.
            wheel_offset = R @ wheel_body_positions()[i]

            def implicit_slip_coefficient(coefficient, direction, spin_axis):
                body_moment = np.cross(wheel_offset, direction)
                inverse_effective_mass = (1.0 / mass_total
                    + float(body_moment @ inverse_inertia_world @ body_moment))
                spin_velocity_per_rate = np.cross(spin_axis, r_contact)
                spin_projection = float(np.dot(spin_velocity_per_rate, direction))
                inverse_effective_mass += spin_projection**2 / wheel_inertias[i]
                return coefficient / (1.0 + coefficient * step_duration
                                      * inverse_effective_mass)

            long_coefficient = implicit_slip_coefficient(
                longitudinal_slip_damping, drive_dir, axle)
            lateral_coefficient = implicit_slip_coefficient(
                lateral_slip_damping, lat_dir, axle)
            desired_tangent = (-long_coefficient * slip_long * drive_dir
                               -lateral_coefficient * slip_lateral * lat_dir
                               + rolling_force)
            desired_norm = np.linalg.norm(desired_tangent)
            friction = desired_tangent * min(1.0, friction_limit/max(desired_norm, 1e-12))
            # Longitudinal and lateral tire forces share one Coulomb limit.
            fr_norm = np.linalg.norm(friction)
            if fr_norm > friction_limit:
                friction = friction * (friction_limit / fr_norm)
            forces[i] = normal*n + friction
            normals[i] = normal
            slips[i] = vt_norm
            rolling_losses[i] = rolling_resistance * normal
            body_torque += np.cross(R @ wheel_body_positions()[i], forces[i])
        else:
            forces[i] = 0.0

    # Coarse chassis/anti-tip contact prevents the rigid-body hull from
    # numerically passing through the terrain after wheel lift. It is an
    # explicit geometry surrogate, not a detailed frame-contact model.
    chassis_local = np.array([0.0, 0.0, -cog_height + chassis_clearance])
    chassis_point = position + R @ chassis_local
    chassis_gap = np.dot(chassis_point, n)
    chassis_penetration = max(0.0, -chassis_gap)
    chassis_normal = max(0.0, contact_stiffness*chassis_penetration - contact_damping*np.dot(velocity, n))
    if chassis_normal > 0.0:
        chassis_force = chassis_normal*n
        total_chassis_torque = np.cross(R @ chassis_local, chassis_force)
    else:
        chassis_force = np.zeros(3)
        total_chassis_torque = np.zeros(3)

    total_force = np.sum(forces, axis=0) + chassis_force + external_force + mass_total*np.array([0.0, 0.0, -g]) + drag_force_vector(velocity)
    # Aggregate frame/occupant compliance keeps the rigid chassis close to the
    # terrain frame. It is a provisional surrogate for suspension/occupant
    # compliance, not a measured wheelchair stiffness.
    terrain_R = np.column_stack((t_x, t_y, n))
    relative_R = terrain_R.T @ R
    relative_euler = euler_from_quaternion(rotation_matrix_to_quaternion_local(relative_R))
    omega_terrain = terrain_R.T @ R @ omega_body
    attitude_torque = np.array([
        -attitude_stiffness[0]*relative_euler[0] - attitude_damping[0]*omega_terrain[0],
        -attitude_stiffness[1]*relative_euler[1] - attitude_damping[1]*omega_terrain[1],
        -attitude_stiffness[2]*relative_euler[2] - attitude_damping[2]*omega_terrain[2],
    ])
    # Contact moments are formed in world coordinates; the rigid-body angular
    # equation uses body coordinates. Convert both environmental and terrain-
    # frame restoring moments before combining them.
    total_torque = (R.T @ (body_torque + total_chassis_torque)
                    + R.T @ terrain_R @ attitude_torque)
    return (total_force, total_torque, forces, normals, slips, rolling_losses,
            longitudinal_slips, lateral_slips)


def rotation_matrix_to_quaternion_local(R):
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


def body_acceleration(velocity, omega_body, total_force, total_torque, quaternion):
    translational = total_force / mass_total
    Iw = body_inertia @ omega_body
    angular = np.linalg.solve(body_inertia, total_torque - np.cross(omega_body, Iw))
    return translational, angular


def state_derivative(state, actuator_torques, grade_deg, cross_slope_deg,
                     external_force=None, step_duration=dt):
    """21-state derivative: r, v, q, omega, four wheel angles and speeds."""
    p = state[0:3]
    v = state[3:6]
    q = quat_normalize(state[6:10])
    w = state[10:13]
    wheel_speed = state[17:21]
    F, M, forces, normals, slips, rolling, slip_long, slip_lateral = contact_forces(
        p, v, q, w, wheel_speed, actuator_torques,
        grade_deg, cross_slope_deg, external_force, step_duration)
    a, alpha = body_acceleration(v, w, F, M, q)
    qdot = quat_derivative(q, w)
    # Motor torque is opposed by contact torque. This is deliberately a simple
    # mechanical wheel model; no motor electrical dynamics are claimed.
    R = quat_to_matrix(q)
    wheel_accel = np.zeros(4)
    for i in range(4):
        axle = R @ np.array([0.0, 1.0, 0.0])
        contact_torque = np.dot(np.cross(-wheel_radii[i]*terrain(grade_deg, cross_slope_deg)[0], forces[i]), axle)
        speed_ratio = min(abs(wheel_speed[i]) / max(wheel_speed_max, 1e-9), 1.0)
        torque_factor = 1.0 if speed_ratio <= 0.8 else max(0.0, 1.0 - (speed_ratio-0.8)/0.2)
        available_motor_torque = actuator_torques[i] * torque_factor
        wheel_accel[i] = (available_motor_torque + contact_torque) / wheel_inertias[i]
    return np.r_[v, a, qdot, alpha, wheel_speed, wheel_accel], {
        "forces": forces, "normal_loads": normals, "slip_speeds": slips,
        "longitudinal_slip_speeds": slip_long,
        "lateral_slip_speeds": slip_lateral,
        "rolling_losses": rolling, "net_force": F, "net_torque": M
    }


def linear_acceleration(torque, velocity, slope_deg, disturbance=0.0):
    """Compatibility helper: grade-aligned 1-D equivalent acceleration."""
    return (wheel_force(torque) - gravity_force(slope_deg)
            - rolling_force(slope_deg)*np.sign(velocity)
            - drag_force(velocity) - disturbance) / effective_mass()


def kinetic_energy(velocity, omega, wheel_speeds=None):
    energy = (0.5*mass_total*np.dot(velocity, velocity)
              + 0.5*np.dot(omega, body_inertia @ omega))
    if wheel_speeds is not None:
        speeds = np.asarray(wheel_speeds, dtype=float)
        if speeds.shape != (wheel_count,):
            raise ValueError(f"wheel_speeds must have shape ({wheel_count},)")
        energy += 0.5*np.dot(wheel_inertias, speeds**2)
    return float(energy)


def wheel_contact_elastic_energy(position, quaternion, grade_deg, cross_slope_deg=0.0):
    """Stored energy in the provisional wheel/chassis normal-contact springs."""
    n, _, _ = terrain(grade_deg, cross_slope_deg)
    centers = body_wheel_centers(position, quaternion)
    penetration = np.maximum(0.0, wheel_radii - centers @ n)
    R = quat_to_matrix(quaternion)
    chassis_local = np.array([0.0, 0.0, -cog_height + chassis_clearance])
    chassis_point = np.asarray(position) + R @ chassis_local
    chassis_penetration = max(0.0, -float(np.dot(chassis_point, n)))
    return 0.5*contact_stiffness*(np.dot(penetration, penetration)
                                  + chassis_penetration**2)


def attitude_elastic_energy(quaternion, grade_deg, cross_slope_deg=0.0):
    """Stored energy in the provisional roll/pitch restoring springs."""
    n, t_x, t_y = terrain(grade_deg, cross_slope_deg)
    terrain_R = np.column_stack((t_x, t_y, n))
    relative_R = terrain_R.T @ quat_to_matrix(quaternion)
    relative_euler = euler_from_quaternion(
        rotation_matrix_to_quaternion_local(relative_R))
    return 0.5*float(np.dot(attitude_stiffness[:2], relative_euler[:2]**2))


def potential_energy(position, grade_deg, cross_slope_deg=0.0):
    """Gravitational potential from XYZ position(s), or forward distance scalar.

    A scalar is interpreted as distance along the grade tangent with zero
    lateral displacement. Vector inputs must end in an XYZ coordinate.
    """
    position = np.asarray(position, dtype=float)
    if position.ndim == 0:
        return mass_total*g*position*np.sin(np.deg2rad(grade_deg))
    if position.shape[-1] != 3:
        raise ValueError("position arrays must have a final dimension of 3 (XYZ)")
    return mass_total*g*position[..., 2]


def total_energy(position, velocity, omega, grade_deg, cross_slope_deg=0.0,
                 wheel_speeds=None, quaternion=None):
    """Mechanical energy, optionally including wheel spin and compliant springs."""
    energy = kinetic_energy(velocity, omega, wheel_speeds)
    energy += potential_energy(position, grade_deg, cross_slope_deg)
    if quaternion is not None:
        energy += wheel_contact_elastic_energy(
            position, quaternion, grade_deg, cross_slope_deg)
        energy += attitude_elastic_energy(quaternion, grade_deg, cross_slope_deg)
    return energy


def power(torque, omega):
    return np.sum(np.asarray(torque) * np.asarray(omega))


def state_equations(t, y, controller, slope_deg, disturbance=0.0):
    """Compatibility wrapper for legacy callers; uses a 3-D state internally."""
    if len(y) == 21:
        throttle = controller(t, y[0], y[3]) if not hasattr(controller, 'command') else controller.command(t, y[0], y[3], disturbance, slope_deg)
        tau = np.zeros(4)
        tau[driven_wheels] = np.asarray(throttle if np.ndim(throttle) else motor_torque(throttle)) / max(np.sum(driven_wheels),1)
        return state_derivative(y, tau, slope_deg, 0.0)[0]
    raise ValueError("3-D model expects a 21-state vector: r(3), v(3), q(4), omega(3), wheel angles(4), wheel speeds(4)")


def system_information():
    print("\n========== 3-D Wheelchair Model ==========")
    print(f"Mass                    : {mass_total:.2f} kg")
    print(f"Body mass (excluding wheels): {body_mass:.2f} kg")
    print(f"Wheelbase / track       : {wheel_base:.2f} / {track_width:.2f} m")
    print(f"Front/rear radii        : {front_caster_radius:.2f}/{wheel_radius:.2f} m (assumed)")
    print(f"Body inertia diag       : {np.diag(body_inertia)} kg m^2 (provisional)")
    print(f"Driven wheels            : {list(np.where(driven_wheels)[0])}")
    print(f"Wheel torque cap/pair    : {motor_torque_max*gear_ratio*motor_efficiency:.2f} Nm aggregate")
    print("Contact model            : compliant normal + Coulomb friction")
    print("Rigid-body DOF           : 3 translation + 3 rotation")
    print("Wheel states             : 4 spin angles + 4 spin speeds")
    print("Important                : inertia/contact/drivetrain values are provisional")
    print("==========================================")
