"""Optimize controller gain scales in the 3-D simulator, then train the ML scheduler.

The search objective is a simulation proxy, not a clinical/real-world safety metric.
Run: python optimize_and_train_ml.py
Longer search: python optimize_and_train_ml.py --trials 12 --horizon 8 --step 0.01
Quick pipeline check: python optimize_and_train_ml.py --trials 2 --horizon 2 --step 0.02
Dependencies: numpy pandas scikit-learn joblib
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import joblib

HERE = Path(__file__).resolve().parent
import ml_gain_tuner as tuner
from simulation import WheelchairSimulation

GAIN_BOUNDS = np.array([
    [0.60, 1.20], [0.35, 1.00], [0.75, 1.40],
    [0.60, 1.20], [0.75, 1.35], [1.00, 1.80], [0.60, 1.25],
], dtype=float)
CONTEXTS = [
    {"name":"flat_smooth", "slope":0.0, "cross":0.0, "scenario":"smooth"},
    {"name":"flat_rough", "slope":0.0, "cross":0.0, "scenario":"rough"},
    {"name":"uphill_rough", "slope":8.0, "cross":0.0, "scenario":"rough"},
    {"name":"downhill_rough", "slope":-8.0, "cross":0.0, "scenario":"rough"},
    {"name":"cross_slope_left", "slope":5.0, "cross":5.0, "scenario":"rough"},
    {"name":"cross_slope_right", "slope":5.0, "cross":-5.0, "scenario":"rough"},
    {"name":"uphill_sudden", "slope":8.0, "cross":0.0, "scenario":"sudden"},
    {"name":"downhill_sudden", "slope":-8.0, "cross":0.0, "scenario":"sudden"},
    {"name":"combined_repeated", "slope":5.0, "cross":5.0, "scenario":"repeated"},
]
CONTROLLERS = ["pid", "nonlinear_pid", "smc", "fuzzy"]
CONTROLLER_GAIN_INDICES = {
    "pid": (0, 1, 2), "nonlinear_pid": (0, 1, 2),
    "smc": (3, 4, 5), "fuzzy": (6,),
}


def gains_dict(values):
    return {name: float(v) for name, v in zip(tuner.GAIN_NAMES, values)}


def score_candidate(values, context, controller_name, duration=8.0, time_step=0.01):
    """Score candidate gain scales with tracking, energy, slip, and safety screens.

    The weights are engineering proxies for simulation comparison, not validated
    clinical or real-world wheelchair safety limits. Lower is better.
    """
    import parameters as P
    base = {
        "KP": P.KP, "KI": P.KI, "KD": P.KD,
        "lambda_smc": P.lambda_smc, "k_smc": P.k_smc,
        "boundary_layer": P.boundary_layer,
        "fuzzy_output_range": tuple(P.fuzzy_output_range),
    }
    g = gains_dict(values)
    try:
        for name in ("KP", "KI", "KD", "lambda_smc", "k_smc", "boundary_layer"):
            setattr(P, name, base[name])
        P.fuzzy_output_range = base["fuzzy_output_range"]

        if controller_name in ("pid", "nonlinear_pid"):
            P.KP = base["KP"] * g["pid_kp_scale"]
            P.KI = base["KI"] * g["pid_ki_scale"]
            P.KD = base["KD"] * g["pid_kd_scale"]
        elif controller_name == "smc":
            P.lambda_smc = base["lambda_smc"] * g["smc_lambda_scale"]
            P.k_smc = base["k_smc"] * g["smc_k_scale"]
            P.boundary_layer = base["boundary_layer"] * g["smc_phi_scale"]
        elif controller_name == "fuzzy":
            lo, hi = base["fuzzy_output_range"]
            P.fuzzy_output_range = (
                lo * g["fuzzy_output_scale"], hi * g["fuzzy_output_scale"])

        result = WheelchairSimulation(
            controller_name, slope=context["slope"], cross_slope=context["cross"],
            scenario=context["scenario"], duration=duration,
            time_step=time_step).run()
        velocity = np.asarray(result["velocity_forward"], dtype=float)
        ref = np.asarray(result["reference_speed"], dtype=float)
        t = np.asarray(result["time"], dtype=float)
        rmse = float(np.sqrt(np.mean((velocity-ref)**2)))
        cruise_threshold = 0.9 * float(np.max(ref))
        ref_cross = np.flatnonzero(ref >= cruise_threshold)
        speed_cross = np.flatnonzero(velocity >= cruise_threshold)
        if len(ref_cross) and len(speed_cross):
            response_delay = max(0.0, float(t[speed_cross[0]] - t[ref_cross[0]]))
        else:
            response_delay = duration
        peak_roll = float(np.max(np.abs(result["roll"])))
        peak_pitch = float(np.max(np.abs(result["pitch"])))
        slip = float(np.max(np.asarray(result["slip_speeds"], dtype=float)))
        power = float(np.mean(np.abs(result["power"])))
        terminated = bool(np.any(result["simulation_terminated"]))

        # Weights are project engineering choices, not certified safety limits.
        score = (rmse + 0.20 * response_delay
                 + 0.25 * peak_roll / max(float(P.max_roll_deg), 1.0)
                 + 0.25 * peak_pitch / max(float(P.max_pitch_deg), 1.0)
                 + 0.15 * slip / 0.05
                 + 0.10 * power / 250.0)
        safety_keys = ("safety_roll_ok", "safety_pitch_ok", "safety_traction_ok",
                       "safety_support_ok", "safety_speed_ok", "safety_reverse_motion_ok", "safety_accel_ok",
                       "safety_lateral_accel_ok", "safety_yaw_rate_ok",
                       "safety_governor_feasible")
        safety_failures = sum(1 for key in safety_keys
                              if not bool(np.all(result[key])))
        score += 10.0 * safety_failures
        if terminated:
            score += 100.0
        if not np.isfinite(score):
            score = 1e6
    finally:
        for name in ("KP", "KI", "KD", "lambda_smc", "k_smc", "boundary_layer"):
            setattr(P, name, base[name])
        P.fuzzy_output_range = base["fuzzy_output_range"]
    return float(score), result


def state_features_from_result(context, controller_name, result):
    """Build training inputs from states visited by the optimized simulation."""
    from dynamics import terrain
    _, _, lateral_axis = terrain(context["slope"], context["cross"])
    lateral_speed = np.asarray(result["velocity_xyz"]) @ lateral_axis
    return np.asarray([
        tuner.feature_vector(
            ref - speed, context["slope"], context["cross"], roll, pitch,
            lat_speed, accel, controller_name)
        for ref, speed, roll, pitch, lat_speed, accel in zip(
            result["reference_speed"], result["velocity_forward"],
            result["roll"], result["pitch"], lateral_speed,
            result["acceleration_forward"])
    ], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=int, default=4, help='random-search candidates per terrain context')
    ap.add_argument('--horizon', type=float, default=8.0, help='optimization rollout horizon (seconds; >=7.25 s includes repeated disturbances)')
    ap.add_argument('--step', type=float, default=0.01, help='optimization rollout time spacing (seconds)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.trials < 2 or args.horizon <= 0 or args.step <= 0:
        ap.error('--trials must be >=2; horizon and step must be positive')
    if args.horizon < 5.5:
        print('WARNING: the horizon ends before the sudden disturbance pulse completes; some designed contexts are not exercised.', flush=True)
    elif args.horizon < 7.25:
        print('WARNING: only the first repeated disturbance pulse is included at this horizon.', flush=True)

    rng = np.random.default_rng(args.seed)
    search_rows = []
    best_rows = []
    X_parts, Y_parts, group_parts = [], [], []
    for context_index, context in enumerate(CONTEXTS):
        for controller_name in CONTROLLERS:
            best_score, best_values, best_result = np.inf, None, None
            active = np.asarray(CONTROLLER_GAIN_INDICES[controller_name], dtype=int)
            print(f"\nOptimizing {controller_name} for {context['name']} "
                  f"({args.trials} candidates)", flush=True)
            for trial in range(args.trials):
                candidate = np.ones(len(tuner.GAIN_NAMES), dtype=float)
                if trial > 0:
                    candidate[active] = rng.uniform(
                        GAIN_BOUNDS[active, 0], GAIN_BOUNDS[active, 1])
                score, result = score_candidate(
                    candidate, context, controller_name, args.horizon, args.step)
                row = {"context": context["name"], "controller": controller_name,
                       "trial": trial, "objective_score": score}
                row.update(gains_dict(candidate))
                search_rows.append(row)
                if score < best_score:
                    best_score, best_values, best_result = score, candidate.copy(), result
                    print(f"  best score={best_score:.5f} at trial {trial + 1}", flush=True)
            best_row = {"context": context["name"], "controller": controller_name,
                        "slope_deg": context["slope"],
                        "cross_slope_deg": context["cross"],
                        "scenario": context["scenario"],
                        "best_objective_score": best_score}
            best_row.update(gains_dict(best_values))
            best_rows.append(best_row)

            # Train on state samples visited by the best simulated rollout,
            # rather than fabricated feature jitter around a context label.
            Xc = state_features_from_result(context, controller_name, best_result)
            yc = np.tile(best_values, (len(Xc), 1))
            X_parts.append(Xc)
            Y_parts.append(yc)
            group_parts.append(np.full(len(Xc), context_index, dtype=int))

    search_df = pd.DataFrame(search_rows)
    best_df = pd.DataFrame(best_rows)
    search_df.to_csv(HERE / 'gain_optimization_search.csv', index=False)
    best_df.to_csv(HERE / 'optimized_gain_labels.csv', index=False)

    X, Y = np.vstack(X_parts), np.vstack(Y_parts)
    groups = np.concatenate(group_parts)
    model = RandomForestRegressor(n_estimators=180, max_depth=12, min_samples_leaf=3,
                                  random_state=args.seed, n_jobs=-1)
    # Hold out all rollouts from one terrain context at a time. Samples within
    # a rollout are correlated, so this is still only a limited simulation check.
    pred = np.zeros_like(Y)
    for train_idx, test_idx in GroupKFold(n_splits=len(CONTEXTS)).split(X, Y, groups):
        fold_model = RandomForestRegressor(n_estimators=180, max_depth=12,
                                           min_samples_leaf=3, random_state=args.seed,
                                           n_jobs=-1)
        fold_model.fit(X[train_idx], Y[train_idx])
        pred[test_idx] = fold_model.predict(X[test_idx])
    mae = mean_absolute_error(Y, pred, multioutput='raw_values')
    model.fit(X, Y)
    joblib.dump(model, HERE / 'ml_gain_model.joblib')
    metadata = {
        "model_version": tuner.MODEL_VERSION,
        "estimator": "RandomForestRegressor",
        "feature_names": tuner.FEATURE_NAMES,
        "gain_names": tuner.GAIN_NAMES,
        "controller_names": tuner.CONTROLLER_NAMES,
        "training_contexts": [context["name"] for context in CONTEXTS],
        "training_labels": "per-controller random-search best gains; state samples from optimized simulated rollouts",
        "simulation_time_step_s": args.step,
        "simulation_duration_s": args.horizon,
        "random_seed": args.seed,
    }
    (HERE / 'ml_gain_model.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    dataset = pd.DataFrame(X, columns=tuner.FEATURE_NAMES)
    for j, name in enumerate(tuner.GAIN_NAMES): dataset[name] = Y[:, j]
    dataset['label_source'] = 'optimized_simulation_rollout_state_samples; correlated_within_rollout'
    dataset.to_csv(HERE / 'ml_gain_training_dataset.csv', index=False)
    report = [
        '# Gain optimization report', '',
        'This report was generated by `optimize_and_train_ml.py`.', '',
        '## Method',
        f'- Random-search candidates per context: {args.trials}',
        f'- Optimization rollout: {args.horizon:g} s with {args.step:g} s spacing (coarsened for search speed)',
        '- Each candidate is scored for one controller; active gains are optimized separately per controller and context.',
        '- Context objective includes signed forward-speed RMSE and delay to reach 90% cruise speed, peak roll/pitch, peak slip, and mean absolute power.',
        '- Failed roll/pitch/traction/support/speed/acceleration screens receive a large penalty; early termination receives a larger penalty.',
        '- Safety-screen thresholds are provisional project settings, not certified physical safety limits.',
        '- Best candidate per controller/context becomes the training label. Inputs are states visited in that candidate rollout; repeated samples within a rollout are correlated.',
        '- The best gain vector is constant over each context rollout; state samples do not establish locally optimal state-dependent gains.',
        '- These are simulator-generated labels, not measurements or independent test observations.', '',
        '## Best candidates', '', best_df.to_string(index=False), '',
        '## ML validation check',
        '- Leave-one-context-out MAE per output: ' + ', '.join(f'{n}={v:.5f}' for n,v in zip(tuner.GAIN_NAMES, mae)),
        f'- This is a limited leave-one-context-out simulation check over {len(CONTEXTS)} designed contexts, not evidence of generalization to real or unseen terrain.',
        '- The randomized search is not a proof of global optimality. Confirm finalists using the full-resolution simulator and held-out terrain scenarios.',
        '- Model parameters and contact properties are provisional simulation assumptions, not measured wheelchair properties.',
    ]
    (HERE / 'GAIN_OPTIMIZATION_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\nSaved gain_optimization_search.csv, optimized_gain_labels.csv, ml_gain_training_dataset.csv, ml_gain_model.joblib, ml_gain_model.json, and GAIN_OPTIMIZATION_REPORT.md')
    print(f'Leave-one-context-out MAE ({len(CONTEXTS)} simulated contexts; not real-world validation):')
    for name, value in zip(tuner.GAIN_NAMES, mae): print(f'  {name}: {value:.5f}')

if __name__ == '__main__':
    main()
