# 3-D Dynamic Smart Wheelchair Model

## Scope

The project has been upgraded from a one-dimensional longitudinal plant to a
three-dimensional rigid-body/contact model. The independent mechanical states
are:

- world position `r = [x,y,z]`
- world translational velocity `v = [vx,vy,vz]`
- body orientation represented by a unit quaternion `q`
- body angular velocity `[p,q,r]`
- four wheel angles and four wheel angular velocities

Thus the simulation contains 3 translational DOF + 3 rotational body DOF + 4
wheel-spin DOF. Wheel rotation is now an independent state rather than a
position-derived no-slip quantity.

## Coordinate convention

The body frame is x-forward, y-left, z-up. The terrain is represented by a
plane with user-configured grade and cross-slope. A local terrain basis and
surface normal are generated from those two angles.

The NumPy controller regulates forward speed only. Although the rigid-body
plant has six body degrees of freedom, it has no steering or path-following
controller, and the four-corner support calculation is a quasi-static screen.
The `rough`, `sudden`, and `repeated` cases apply external force disturbances
while the surface remains planar; they are not simulations of bumps or measured
3-D terrain geometry. `lateral_acceleration` adds a prescribed lateral force,
not a lateral controller.

## Equations of motion

The rigid-body translational equation is

`m a = sum(F_contact) + F_external + F_gravity + F_aero`.

The rotational equation in body coordinates is

`I omega_dot = M_contact - omega x (I omega)`.

Quaternion attitude is integrated from body angular velocity.

Reported stored mechanical energy includes chassis translation/rotation, the
four wheel-spin energies, gravitational potential, and the modeled contact and
roll/pitch spring energies. It excludes electrical/battery energy, spring and
contact damping losses, rolling losses, and motor/gearbox thermal losses, so it
is not an energy-conservation validation or battery-consumption estimate.

Each wheel has

`I_w omega_w_dot = tau_drive + tau_contact`.

Wheel-ground contact uses a compliant normal surrogate

`N = max(0, k delta - c v_n)`

where `delta` is wheel-ground penetration. Tangential force is limited by a
Coulomb friction bound. This is a computational contact approximation, not a
validated tire model.

Longitudinal and lateral tire forces oppose their contact slip velocities using
provisional velocity-force coefficients, then are limited together by the
friction circle. Motor torque accelerates the wheel; it is not converted
directly into an imposed tire force. This retains the wheel-spin/contact
interaction. The slip response uses a step-size-aware implicit damping
coefficient to avoid numerical oscillation at the model's simulation step.
The tire coefficients still require calibration.

## Contact model limitations

The current 3-D model does **not** include:

- suspension compliance or suspension geometry
- detailed pneumatic/solid tire deformation
- experimentally identified tire force-slip curves
- wheel camber/toe effects
- caster steering dynamics
- battery voltage/current dynamics
- motor back-EMF and identified torque-speed map
- gearbox backlash or detailed efficiency map
- articulated occupant motion
- terrain roughness represented by measured 3-D geometry

Therefore the model is a 3-D dynamic research simulation, not a validated
physical digital twin of a particular wheelchair.

## Drivetrain assumption

The current project assumes two rear driven wheels and two non-driven front
contact wheels. Motor torque is represented by a bounded aggregate actuator,
with a first-order mechanical response. This must be replaced by the actual
motor count, torque-speed/current data, gearbox ratio and efficiency map when
available.

## Parameter provenance

### Standard constants

`g` and the nominal air density are standard constants/nominal environmental
inputs.

### Project assumptions requiring validation

- mass and occupied load case
- wheel radius and wheel mass
- wheelbase and track width
- CoG height and location
- body rotational inertias
- contact stiffness/damping
- rolling resistance
- tire/surface friction coefficients
- aerodynamic coefficient and frontal area
- motor torque, gear ratio and efficiency
- driven-wheel configuration

The body inertia values currently in `parameters.py` are explicitly provisional
and are not literature values.

## Verification plan

1. Check zero-grade equilibrium and wheel-contact force balance.
2. Check convergence by reducing `dt` and comparing trajectory/attitude.
3. Verify quaternion norm remains one.
4. Check friction forces never exceed the configured Coulomb limit.
5. Check wheel torque and speed limits.
6. Compare normal loads against the quasi-static spatial screen.
7. Replace provisional parameters with measured/manufacturer/literature data.
8. Calibrate on one dataset and validate on separate datasets.

No standards-compliance or real-world validity claim follows from the current
simulation alone.

## Additional 3-D modeling choices

The 3-D implementation uses a quaternion to avoid Euler-angle integration
singularities. Euler angles are calculated only for reporting and safety
screens.

A compliant wheel-ground normal-force surrogate is used instead of a hard
constraint solver. A small lower-frame/chassis contact envelope is also used
to prevent numerical penetration after wheel lift. Both are computational
surrogates and require identification if the model is to represent a specific
wheelchair.

A provisional roll/pitch restoring torque is included to represent aggregate
frame/occupant compliance:

`M_att = -K_att (eta - eta_terrain) - C_att omega`.

This term is explicitly **not** a measured suspension stiffness. Remove or
replace it when actual chassis, suspension, seat, and occupant compliance data
are available.

The four wheel normal loads and friction utilization are retained as direct
simulation outputs. The older `stability.py` grade-only screen remains a
separate simplified reference calculation and should not be confused with the
integrated 3-D dynamics.
