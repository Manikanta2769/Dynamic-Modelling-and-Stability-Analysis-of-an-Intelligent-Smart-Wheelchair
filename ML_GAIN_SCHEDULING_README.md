# ML-assisted controller gain scheduling

## What the ML does

The Random Forest estimates bounded multipliers for the selected controller's
gains from the current simulated speed error, grade, cross-slope, terrain-relative
roll/pitch, lateral speed, and filtered acceleration. The controller queries it
online and uses those multipliers in its force/torque command. Existing plant
actuator limits still clamp the command.

This is gain scheduling, not a learned policy that directly drives the chair.
It does not guarantee a faster response. “Faster” is evaluated with signed
forward-speed tracking, time to 90% of cruise speed, recovery after a disturbance,
energy use, slip, attitude, safety screens, and simulation termination.

PyBullet is optional for the separate visualization. It has been removed from
`requirements.txt` so a PyBullet build failure cannot block the ML/dynamics
workflow. For Windows, the included `environment.yml` installs PyBullet from
conda-forge rather than building it from the PyPI source package.

## Setup and run

From this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train/retrain the current model from the simulator's bounded random-search
objective (this runs many simulated rollouts and can take several minutes):

```powershell
.\.venv\Scripts\python.exe optimize_and_train_ml.py --trials 4 --horizon 8 --step 0.01
```

This writes the model plus `ml_gain_model.json` metadata, optimization tables,
and a report. The metadata is required: artifacts with an old or incompatible
feature/controller schema are ignored, so an old model is not silently treated
as the current controller-specific training result.

Run the 3-D model with online ML gains:

```powershell
.\.venv\Scripts\python.exe main.py --sim --controller nonlinear_pid --slope 5 --cross 5 --scenario sudden --ml-adaptive
```

Run the same controller with fixed nominal gains by omitting `--ml-adaptive`.
PyBullet is a separate legacy visualization and is not controlled by this ML
gain scheduler. Use `--visual` or `--all` to run that path explicitly.

## Paired comparison

This runs fixed and adaptive versions with the same controller, terrain,
disturbance, duration, and integration spacing:

```powershell
.\.venv\Scripts\python.exe compare_ml_adaptive.py --slope 5 --cross 5 --scenario sudden --horizon 6 --step 0.01
```

It writes `ml_adaptive_comparison.csv`, one gain trace per adaptive controller,
and a compact speed/attitude plot plus gain schedule plot for the selected
controller (`--plot-controller nonlinear_pid`). Lower speed RMSE is better
tracking; negative percentage change means improvement. Check the other
columns too: a shorter response is not an improvement if energy, slip, roll,
pitch, or screen failures become worse. A 0.01 s study is a coarsened simulation;
compare finalist cases again at the default 0.0025 s model step.

## What conclusions the data supports

The trainer separately optimizes only the active gains for each controller in
nine designed terrain/disturbance contexts. The 8 s training horizon includes
the sudden pulse and the first two repeated pulses; a horizon of 16.25 s is
needed to include all five. It samples state features from the resulting best
simulated rollouts. Those samples are correlated states carrying the same
context-level gain label; they are not independent observations or evidence
that the labels are locally optimal. Leave-one-context-out MAE over nine
contexts is a limited extrapolation check. Random search is not a global
optimum. Retraining is required before the new controller-specific feature
schema can load an older model artifact.

The tire force/slip law changed after the previous model was trained. The
current model schema is version 5, so the old artifact is ignored and adaptive
runs use the bounded rule-based fallback until retraining. First verify the
fixed-gain baseline on smooth flat terrain; retrain only after accepting the
corrected plant behavior. The checked-in optimization tables, training data,
gain traces, and comparison CSVs are historical outputs and should not be
presented as results for the current plant.

The plant and safety thresholds are project assumptions, not a measured chair,
certification test, or clinical safety model. The ML result can support a
course-project demonstration that online gain scheduling was integrated and
compared reproducibly. It cannot establish that a real wheelchair is safer or
faster without calibrated hardware parameters and independent validation data.
