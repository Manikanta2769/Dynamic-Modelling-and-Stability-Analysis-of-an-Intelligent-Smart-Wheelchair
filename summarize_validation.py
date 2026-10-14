"""Aggregate paired fixed-gain/ML comparison CSVs from compare_ml_adaptive.py.

Example: python summarize_validation.py ml_smooth.csv ml_rough.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ("speed_rmse_mps", "peak_abs_roll_deg", "peak_abs_pitch_deg",
           "peak_slip_speed", "mean_abs_power_w", "absolute_energy_j")
CONDITION_KEYS = ("scenario", "slope_deg", "cross_slope_deg")
REQUIRED = {*CONDITION_KEYS, "controller", "mode", "simulation_terminated", *METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="comparison CSVs")
    args = parser.parse_args()
    frames = []
    for path in args.inputs:
        if not path.is_file():
            parser.error(f"input file not found: {path}")
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED - set(frame.columns))
        if missing:
            parser.error(f"{path} is not a compare_ml_adaptive.py output; missing: {', '.join(missing)}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    pair_keys = [*CONDITION_KEYS, "controller", "mode"]
    duplicates = data.duplicated(pair_keys)
    if duplicates.any():
        parser.error("duplicate scenario/controller/mode rows; use one CSV per unique test condition")
    print("\nMEAN METRICS BY CONTROLLER AND MODE (lower tracking error/power is not a safety verdict)")
    print(data.groupby(["controller", "mode"])[list(METRICS)].mean().round(5).to_string())
    pivot = data.pivot(index=[*CONDITION_KEYS, "controller"], columns="mode", values="speed_rmse_mps")
    if {"fixed_gain", "ml_adaptive"}.issubset(pivot.columns):
        denom = pivot["fixed_gain"].abs().replace(0, np.nan)
        improvement = 100 * (pivot["fixed_gain"] - pivot["ml_adaptive"]) / denom
        print("\nSIGNED FORWARD-SPEED RMSE IMPROVEMENT BY SCENARIO (%)")
        print(improvement.unstack("scenario").round(2).to_string())
        print("\nMean RMSE improvement by controller (%)")
        print(improvement.groupby("controller").mean().round(2).to_string())
    terminated = data[data["simulation_terminated"].astype(bool)]
    print("\nEARLY TERMINATIONS")
    cols = [c for c in ("scenario", "controller", "mode", "termination_reason") if c in data]
    print(terminated[cols].to_string(index=False) if not terminated.empty else "None")
    print("\nThese are simulator comparisons on assumed parameters, not independent physical validation.")


if __name__ == "__main__":
    main()
