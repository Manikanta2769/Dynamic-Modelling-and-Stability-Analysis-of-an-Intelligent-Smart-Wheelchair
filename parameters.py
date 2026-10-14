"""Central parameters for the 3-D smart-wheelchair model.

Values marked ``assumption`` are project inputs and must not be presented as
measured wheelchair data.  The model deliberately keeps provenance visible.
"""
import numpy as np

g = 9.81
rho_air = 1.225

# Geometry (provisional project configuration)
# Wheel order throughout the model: front-right, front-left, rear-right, rear-left.
wheel_radius = 0.30  # rear drive-wheel radius (legacy scalar alias)
front_caster_radius = 0.11
wheel_radii = np.array([front_caster_radius, front_caster_radius,
                        wheel_radius, wheel_radius], dtype=float)
wheel_base = 0.70
track_width = 0.65
cog_height = 0.513
cog_x = 0.0
cog_y = 0.0
chassis_clearance = 0.05        # m; provisional lower-frame ground-contact envelope

# Mass properties
mass_chair = 40.0
mass_user = 60.0
mass_total = mass_chair + mass_user
wheel_masses = np.array([0.7, 0.7, 2.5, 2.5], dtype=float)  # assumed caster/rear masses
wheel_mass = float(np.mean(wheel_masses))  # legacy scalar alias only
wheel_count = 4
body_mass = mass_total - float(np.sum(wheel_masses))
# Passive caster rollers use a spherical inertia approximation; rear wheels
# use a solid-disc approximation. These are assumptions pending CAD data.
wheel_inertias = np.array([
    0.4 * wheel_masses[0] * wheel_radii[0]**2,
    0.4 * wheel_masses[1] * wheel_radii[1]**2,
    0.5 * wheel_masses[2] * wheel_radii[2]**2,
    0.5 * wheel_masses[3] * wheel_radii[3]**2,
], dtype=float)
wheel_inertia = float(wheel_inertias[2])  # legacy rear-wheel alias
# Provisional occupied-body rotational inertias. These are NOT literature data.
body_inertia = np.diag([18.0, 22.0, 25.0])
attitude_stiffness = np.array([140.0, 180.0, 0.0])   # N m/rad; provisional aggregate frame/occupant restoring model
attitude_damping = np.array([35.0, 40.0, 8.0])         # N m s/rad; provisional

# Contact/material abstractions
rolling_resistance = 0.015
mu_static = 0.70          # rubber tyre on dry pavement/concrete (typ. 0.6-0.8)
mu_dynamic = 0.60         # kinetic friction, slightly below static
longitudinal_slip_damping = 12000.0    # N/(m/s), provisional tire surrogate
lateral_slip_damping = 9000.0          # N/(m/s), provisional tire surrogate
contact_stiffness = 120000.0       # N/m, provisional rigid-contact surrogate
contact_damping = 4500.0          # N/(m/s), provisional

# Aerodynamics
drag_coefficient = 0.80
frontal_area = 0.60

# Drivetrain: provisional aggregate actuator model
motor_torque_max = 35.0           # Nm at motor shaft, aggregate assumption
motor_speed_max_rpm = 3000.0
motor_time_constant = 0.08        # s, first-order actuator lag
motor_efficiency = 0.90
gear_ratio = 20.0
wheel_speed_max = motor_speed_max_rpm * 2.0*np.pi/60.0 / gear_ratio
# Rear pair is assumed driven. Change only with configuration evidence.
driven_wheels = np.array([False, False, True, True], dtype=bool)  # FR, FL, RR, RL

# Simulation
simulation_time = 20.0
dt = 0.0025
time = np.linspace(0, simulation_time, int(simulation_time / dt) + 1)

# Initial pose/state
initial_position = np.array([0.0, 0.0, cog_height])
initial_velocity = np.zeros(3)
initial_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
initial_angular_velocity = np.zeros(3)
initial_wheel_angles = np.zeros(4)
initial_wheel_speeds = np.zeros(4)

# Legacy aliases retained for analysis/plots
initial_state = np.array([0.0, 0.0])
slope_angle_deg = 0.0
terrain_angles = np.array([0, 5, 10, 15, 20, 25, 30, 35])

# Controllers
KP = 150.0
KI = 25.0
KD = 30.0
lambda_smc = 5.0
k_smc = 40.0
boundary_layer = 0.05
fuzzy_error_range = (-1.0, 1.0)
fuzzy_output_range = (-20.0, 20.0)

# Safety/test limits
max_velocity = 2.5
max_acceleration = 2.0
max_lateral_acceleration = 1.5
max_roll_deg = 15.0
max_pitch_deg = 20.0
max_yaw_rate = 1.5
max_slope = 35.0
max_cross_slope = 20.0

# Output
figure_size = (10, 6)
dpi = 150
GRAPH_DIR = "output/graphs/"
CSV_DIR = "output/csv/"

# Provenance labels used in reports
PARAMETER_PROVENANCE = {
    "g": "standard constant",
    "rho_air": "standard nominal assumption",
    "geometry": "project assumption; verify against chair",
    "mass": "project test case; verify configuration",
    "body_inertia": "provisional assumption; measure/CAD/literature required",
    "contact": "provisional contact surrogate; identify tire/surface",
    "aerodynamics": "provisional estimate",
    "drivetrain": "provisional aggregate actuator model",
}
