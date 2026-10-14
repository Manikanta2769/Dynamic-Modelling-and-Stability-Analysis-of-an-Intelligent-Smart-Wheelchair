# Dynamic-Modelling-and-Stability-Analysis-of-an-Intelligent-Smart-Wheelchair

A simulation project that models a powered wheelchair and compares four speed controllers (Linear PID, Nonlinear PID, Fuzzy and Sliding Mode) with machine-learning adaptive gain scheduling. The study uses a NumPy 3-D dynamic model and a PyBullet co-simulation on a terrain track, and analyses speed tracking, attitude (roll/pitch/yaw), acceleration and contact forces.

> **Disclaimer:** This is a simulation study. Controller gains are provisional tuning values, and the ML model is trained on simulator data. Nothing here is validated on hardware or a certified safety limit.

---

## Features

- **Four controllers:** Linear PID, Nonlinear PID, Fuzzy, Sliding Mode (SMC)
- **ML adaptive gains:** a trained RandomForest model adjusts the gains online from the chair's state (speed error, grade, roll, pitch, lateral speed, acceleration)
- **Two simulation levels:**
  - NumPy 3-D dynamic model with 4 scenarios: smooth, rough, sudden, repeated
  - PyBullet co-simulation on a terrain track: 2 speed humps, 10° climb, crest, 10° decline
- **Same graph layout for every controller:** position, speed, velocity, acceleration, contact force, drive force, roll/pitch
- **CSV logs** of the ML gain changes together with the states that caused them

---

## Project structure

| File | Purpose |
|---|---|
| `run_all.py` | Main entry point |
| `controllers.py` | All controllers (NumPy 3-D and PyBullet versions) |
| `simulation.py` | NumPy 3-D simulation and CSV export |
| `dynamics.py`, `parameters.py` | Plant model and parameters |
| `stability.py`, `spatial_stability.py` | Stability analysis |
| `run_co_simulation.py` | PyBullet terrain co-simulation |
| `plots.py` | Graph generation |
| `ml_gain_tuner.py` | ML gain scheduler (loads `ml_gain_model.joblib`) |
| `optimize_and_train_ml.py` | Gain search and model training |
| `pybullet_wheelchair.urdf` | Wheelchair model for PyBullet |
| `requirements.txt` | Core Python packages (see below) |
| `environment.yml` | Full conda environment, including PyBullet |

---

## Installation

Python 3.11 is recommended.

### About `requirements.txt`

`requirements.txt` lists the core packages:

```
numpy
matplotlib
pandas
scikit-learn
joblib
```

These are enough to run the NumPy 3-D simulation (`--sim` mode). **`pybullet` is not in this file.** It is needed for the terrain co-simulation (`--visual` and the default `python run_all.py`), so install it separately, or use `environment.yml`.

### Option 1: pip

```bash
pip install -r requirements.txt
pip install pybullet
```

On Windows, if the PyBullet build fails (it may need Microsoft C++ Build Tools), use conda instead.

### Option 2: conda (installs everything, including PyBullet)

```bash
conda env create -f environment.yml
conda activate smart-wheelchair
```

---

## Usage

**Full run (ML on):**

```bash
python run_all.py
```

This runs Linear PID, Sliding Mode and Fuzzy headless (graphs only), then Nonlinear PID with the PyBullet window.

**Other modes:**

```bash
python run_all.py --visual --controller smc              # PyBullet only, one controller
python run_all.py --visual --controller fuzzy --headless
python run_all.py --sim --compare --ml-adaptive          # NumPy model, all controllers x scenarios
python run_all.py --sim --compare --ml-adaptive --compare-duration 5   # quick check
```

| Flag | Meaning |
|---|---|
| `--controller` | `pid`, `nonlinear_pid`, `fuzzy`, `smc` |
| `--scenario` | `smooth`, `rough`, `sudden`, `repeated` |
| `--slope`, `--cross` | Grade and cross-slope in degrees (NumPy model) |
| `--ml-adaptive` | Enable ML gain scheduling in the NumPy model |
| `--headless` | PyBullet without a window |
| `--compare-duration`, `--compare-step` | Duration and time step of comparison runs |

---

## Output

| Folder | Contents |
|---|---|
| `output/controller_graphs/<controller>/` | `00_summary.png`, the summary graph for each controller |
| `output/graphs/` | PyBullet `stability_hud_analysis_<controller>.png` |
| `output/csv/` | `pybullet_ml_gains_<controller>.csv`, `ml_gain_trace_<controller>_<scenario>_....csv`, `simulation_3d.csv`, `controller_scenario_comparison.csv` |

### Summary graph panels (PyBullet terrain run)

Position (px, py, pz), speed, velocity, acceleration, total ground contact force m(a − g), drive force, and roll/pitch with the path slope. All panels are derived from the recorded position, so they agree with each other. Pitch is plotted nose-up positive.

### ML gain CSV

Each CSV lists only the gains used by that controller, together with the model inputs at each update:

| Controller | Gain columns |
|---|---|
| Linear / Nonlinear PID | `pid_kp_scale`, `pid_ki_scale`, `pid_kd_scale` |
| Sliding Mode | `smc_lambda_scale`, `smc_k_scale`, `smc_phi_scale` |
| Fuzzy | `fuzzy_output_scale` |

On a steady constant slope the inputs barely change, so the gains stay almost constant. On the PyBullet terrain (climb, crest, decline) the gains change with the grade.

---

## Notes

- ML-on runs are slow (several minutes for all cases). Graph saving prints progress.
- A scikit-learn `InconsistentVersionWarning` only means your version differs from the one used to save the model. It is harmless if the run reports ML adaptive gain scheduling is ON.
- Delete `output/controller_graphs` before a fresh run to remove stale files.
