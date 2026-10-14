# 3-D model upgrade

## What changed

- Replaced the 1-D `[x, v]` plant with a 21-state 3-D plant.
- Added world `x,y,z` position and velocity.
- Added quaternion body attitude and body angular velocity.
- Added roll, pitch and yaw dynamics through a full rigid-body inertia matrix.
- Added four independent wheel angles and wheel angular velocities.
- Added four wheel-ground contacts with normal force and Coulomb friction.
- Added wheel torque-to-contact-force interaction and wheel torque-speed limiting.
- Added two rear driven wheels as an explicit provisional configuration.
- Added first-order drivetrain actuator lag.
- Added provisional aggregate frame/occupant roll-pitch compliance.
- Added lower-frame contact envelope for numerical ground penetration protection.
- Added 3-D trajectory, attitude, wheel-load, wheel-torque and velocity plots.
- Updated metrics for roll, pitch, yaw, wheel slip, normal load and friction use.
- Updated `MODEL.md` with the 3-D equations, assumptions and limitations.

## Deliberate limitations

This is a 3-D dynamic research model, not a validated digital twin. It still
needs actual motor/drivetrain data, wheel/tire properties, body inertias, CoG
measurements, and calibration/validation data. The contact and compliance
terms are provisional.


## ML optimizer refinement (2026-10-11)
- Updated gain-search objective to use normalized tracking, attitude, slip, and power terms.
- Added large penalties for failed simulator safety screens and early termination.
- Documented that labels remain shared compromises across four controllers and synthetic feature augmentation is not independent evidence.

## Review corrections (2026-10)

- Replaced the obsolete optimization report and corrected the trainer to tune
  each controller's active gains independently, include the nominal-gain
  baseline, sample multiple signed terrain/disturbance contexts, and split
  validation by whole context. Existing model artifacts with the old schema
  are rejected until retraining.
- Corrected tracking metrics and the paired comparison aggregator to use the
  signed forward-speed reference and current comparison CSV schema.
- Corrected the PyBullet URDF mass/layout assumptions, made caster pivots
  passive swivel joints, smoothed/decimated camera tracking, and guarded normal
  GUI closure. Replaced the duplicate broken visual runner with a wrapper.
- Made model output call its support/traction results quasi-static screens and
  added drivetrain motor reaction moments to chassis dynamics.
- An earlier smoke optimization (2 candidates/context, 2 s horizon, 0.02 s
  step) checked the training pipeline only; its model and report were invalidated
  by the changes below and are not current artifacts.
- These changes have not been numerically verified in this pass. Physical
  parameter identification and independent real-chair validation remain open.
- Corrected normal-load initialization for the combined terrain normal,
  enabled the existing lateral-acceleration disturbance input, and made the
  acceleration governor bidirectional with an explicit actuator-feasibility
  flag. Optimizer rollouts now default to 8 s so sudden/repeated disturbances
  actually occur; model schema v4 invalidates the earlier trained artifact.
- Corrected legacy static slip/tip screens for signed grades and cross-slope,
  clarified that the PyBullet HUD uses display screens rather than safety
  certification, and separated the units in the stability plot.
- Replaced the imposed motor-torque/r tire force with a bounded contact-slip
  force law, and added longitudinal/lateral slip, wheel speed, command torque,
  contact-force, and per-screen diagnostic outputs. Added a reverse-motion
  screen and invalidated prior trained gains under model schema v5. These
  corrections need a local smooth-flat baseline run before ML retraining.
- Made tire slip-force damping step-size-aware using the effective mass at
  each contact. This addresses the alternating acceleration spikes observed
  in the smooth-flat baseline; numerical improvement still needs a rerun.
- Added a numerical controller-by-scenario CSV export to `main.py --compare`,
  including signed tracking, wheel slip, loads, energy, termination, and each
  individual model screen.
