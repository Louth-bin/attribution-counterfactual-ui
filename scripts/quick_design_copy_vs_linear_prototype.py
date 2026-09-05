"""Screen diabetes training/test designs where CF edit-copying should shine.

This is deliberately a quick simulation, not a production experiment rebuild.
The selection heuristics use model/explanation geometry, not participant
outcomes:

1. Restrict to the two non-age feature clusters used in the current design:
   glucose|bmi and blood_pressure|insulin.
2. Pick training cases whose displayed CF edits are moderate and consistent
   within target+pair strata, so a person can learn a reusable edit template.
3. Pick test cases where the learned CF edit template lands just across the
   true MLP boundary, while a linear mental boundary and a target-prototype
   rule either under-shoot or over-shoot.

The final reported distances are exact unrestricted L1 distances to the MLP
decision boundary for the selected configuration.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calculate_v09_boundary_distance import WachterBoundarySearch, model_logit


OUTDIR = ROOT / "outputs" / "v20-copy-vs-linear-prototype-design"
OUTDIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = ROOT / "analysis" / "diabetes_hierarchical_oriented_q90_anchor_search_candidate_shortlist.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"

FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
PAIRS = ["glucose | bmi", "blood_pressure | insulin"]
PAIR_IDXS = {
    "glucose | bmi": [0, 3],
    "blood_pressure | insulin": [1, 2],
}
STRATA = list(itertools.product(PAIRS, ["Diabetes", "No Diabetes"]))

# Moderate/broad memory: high enough to weight similar examples a bit, not so
# high that it becomes single-exemplar memorization.
COPY_LAMBDA = 2.0
TRAIN_PER_STRATUM = 2
TEST_PER_STRATUM = 2
TRAIN_COUNTS = {
    ("glucose | bmi", "Diabetes"): 2,
    ("glucose | bmi", "No Diabetes"): 2,
    ("blood_pressure | insulin", "Diabetes"): 2,
    ("blood_pressure | insulin", "No Diabetes"): 1,
}


def sigmoid(logit: float) -> float:
    logit = float(np.clip(logit, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-logit))


def target_probability(profile: np.ndarray, target_label: str, model: dict, rep_case: dict) -> float:
    p_no_diabetes = sigmoid(model_logit(model, profile, rep_case))
    return p_no_diabetes if target_label == "No Diabetes" else 1.0 - p_no_diabetes


def normalized_profile(raw_dict: dict[str, float], rep_case: dict) -> np.ndarray:
    ranges = np.asarray(rep_case["raw_feature_ranges"], dtype=float)
    values = np.asarray([float(raw_dict[name]) for name in FEATURES], dtype=float)
    return np.clip((values - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]), 0.0, 1.0)


def raw_profile(profile: np.ndarray, rep_case: dict) -> dict[str, float]:
    ranges = np.asarray(rep_case["raw_feature_ranges"], dtype=float)
    values = ranges[:, 0] + np.asarray(profile, dtype=float) * (ranges[:, 1] - ranges[:, 0])
    return {name: float(value) for name, value in zip(FEATURES, values)}


def load_candidates() -> tuple[pd.DataFrame, dict, dict]:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    model = bundle["browser_model"]
    rep_case = bundle["training_pool"][0]
    df = pd.read_csv(CANDIDATES)
    df = df[df["main_pair"].isin(PAIRS)].copy()
    df["profile"] = df["original_profile"].map(json.loads).map(lambda d: normalized_profile(d, rep_case))
    df["cf_profile_norm"] = df["counterfactual_profile"].map(json.loads).map(lambda d: normalized_profile(d, rep_case))
    df["delta"] = df["normalized_change_vector"].map(json.loads).map(lambda x: np.asarray(x, dtype=float))
    df["cf_l1"] = df["delta"].map(lambda d: float(np.abs(d).sum()))
    df["target_prob_cf_checked"] = [
        target_probability(p, target, model, rep_case)
        for p, target in zip(df["cf_profile_norm"], df["target_label"])
    ]
    return df, model, rep_case


def ridge_linear_weights(training: pd.DataFrame) -> tuple[np.ndarray, float]:
    x = np.vstack(training["profile"].to_numpy())
    y = np.where(training["prediction_label"].eq("No Diabetes"), 1.0, -1.0)
    x_aug = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(x_aug.shape[1]) * 0.15
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    return weights[:-1], float(weights[-1])


def minimal_linear_delta(profile: np.ndarray, target_label: str, pair: str, w: np.ndarray, b: float) -> np.ndarray:
    target_sign = 1.0 if target_label == "No Diabetes" else -1.0
    margin = target_sign * (float(profile @ w + b))
    direction_scores = target_sign * w
    selected = PAIR_IDXS[pair]
    delta = np.zeros(5)
    if margin >= 0.0:
        return delta
    needed = -margin + 0.03  # cross the participant's learned boundary by a small margin
    candidates = sorted(selected, key=lambda idx: abs(direction_scores[idx]), reverse=True)
    for idx in candidates:
        direction = 1.0 if direction_scores[idx] >= 0 else -1.0
        capacity = (1.0 - profile[idx]) if direction > 0 else profile[idx]
        gain_per_unit = abs(direction_scores[idx])
        if gain_per_unit <= 1e-9 or capacity <= 0:
            continue
        step = min(capacity, needed / gain_per_unit)
        delta[idx] = direction * step
        needed -= step * gain_per_unit
        if needed <= 1e-9:
            break
    return np.clip(profile + delta, 0.0, 1.0) - profile


def copy_delta(profile: np.ndarray, target_label: str, pair: str, training: pd.DataFrame) -> np.ndarray:
    same = training[(training["main_pair"].eq(pair)) & (training["target_label"].eq(target_label))]
    distances = np.asarray([np.mean(np.abs(profile - p)) for p in same["profile"]], dtype=float)
    weights = np.exp(-COPY_LAMBDA * distances)
    weights = weights / weights.sum()
    return np.sum(np.vstack(same["delta"].to_numpy()) * weights[:, None], axis=0)


def prototype_delta(profile: np.ndarray, target_label: str, pair: str, training: pd.DataFrame) -> np.ndarray:
    same_target = training[training["prediction_label"].eq(target_label)]
    proto = np.mean(np.vstack(same_target["profile"].to_numpy()), axis=0)
    delta = np.zeros(5)
    idxs = PAIR_IDXS[pair]
    delta[idxs] = proto[idxs] - profile[idxs]
    return np.clip(profile + delta, 0.0, 1.0) - profile


def eval_design(training: pd.DataFrame, testing: pd.DataFrame, model: dict, rep_case: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    w, b = ridge_linear_weights(training)
    rows = []
    for _, row in testing.iterrows():
        profile = row["profile"]
        pair = row["main_pair"]
        target = row["target_label"]
        true_cf_delta = row["delta"]
        for strategy, delta in [
            ("copy moderate λ=2", copy_delta(profile, target, pair, training)),
            ("linear mental model", minimal_linear_delta(profile, target, pair, w, b)),
            ("target prototype", prototype_delta(profile, target, pair, training)),
            ("oracle shown CF", true_cf_delta),
        ]:
            edited = np.clip(profile + delta, 0.0, 1.0)
            p = target_probability(edited, target, model, rep_case)
            rows.append({
                "source_split": row["source_split"],
                "source_instance_id": int(row["instance_id"]),
                "prediction_label": row["prediction_label"],
                "target_label": target,
                "main_pair": pair,
                "strategy": strategy,
                "edit_L1": float(np.abs(delta).sum()),
                "target_probability": p,
                "success": float(p >= 0.5),
                "boundary_proxy_abs_p_minus_0.5": abs(p - 0.5),
                "copy_delta_error_vs_true_cf_L1": float(np.abs(delta - true_cf_delta).sum()),
            })
    trials = pd.DataFrame(rows)
    summary = (
        trials.groupby("strategy", as_index=False)
        .agg(
            mean_edit_L1=("edit_L1", "mean"),
            median_edit_L1=("edit_L1", "median"),
            success_rate=("success", "mean"),
            mean_target_probability=("target_probability", "mean"),
            mean_boundary_proxy=("boundary_proxy_abs_p_minus_0.5", "mean"),
            mean_delta_error_vs_true_cf=("copy_delta_error_vs_true_cf_L1", "mean"),
        )
    )
    return trials, summary


def design_objective(summary: pd.DataFrame) -> float:
    by = summary.set_index("strategy")
    copy = by.loc["copy moderate λ=2"]
    lin = by.loc["linear mental model"]
    proto = by.loc["target prototype"]
    # Positive is good: copy close to boundary and successful; linear/prototype
    # need larger edits and/or end farther from boundary. Prototype should be
    # largest edit; linear should be between copy and prototype.
    return float(
        3.0 * (lin["mean_boundary_proxy"] - copy["mean_boundary_proxy"])
        + 3.0 * (proto["mean_boundary_proxy"] - copy["mean_boundary_proxy"])
        + 1.5 * (lin["mean_edit_L1"] - copy["mean_edit_L1"])
        + 1.5 * (proto["mean_edit_L1"] - lin["mean_edit_L1"])
        + 2.0 * copy["success_rate"]
        - 1.0 * abs(proto["success_rate"] - 1.0)  # overshoot is OK; failure is noisier
    )


def consistent_training_pool(df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    pools = {}
    for pair, target in STRATA:
        pool = df[
            df["main_pair"].eq(pair)
            & df["target_label"].eq(target)
            & df["cf_l1"].between(0.04, 0.42)
            & df["target_prob_cf_checked"].between(0.50, 0.75)
        ].copy()
        if len(pool) < TRAIN_COUNTS[(pair, target)]:
            pool = df[
                df["main_pair"].eq(pair)
                & df["target_label"].eq(target)
                & df["cf_l1"].between(0.02, 0.55)
                & df["target_prob_cf_checked"].between(0.50, 0.90)
            ].copy()
        pools[(pair, target)] = pool
    return pools


def candidate_training_sets(
    pools: dict[tuple[str, str], pd.DataFrame],
    df: pd.DataFrame,
    model: dict,
    rep_case: dict,
) -> Iterable[pd.DataFrame]:
    # Enumerate small within-stratum training sets and keep the ones where a
    # broad copy-template works on held-out same-stratum candidates. This still
    # uses only model/explanation geometry, not participant behavior.
    top_by_stratum = {}
    for stratum, pool in pools.items():
        pair, target = stratum
        train_count = TRAIN_COUNTS[stratum]
        same_all = df[df["main_pair"].eq(pair) & df["target_label"].eq(target)]
        rows = list(pool.index)
        scored_combos = []
        for combo_indices in itertools.combinations(rows, min(train_count, len(rows))):
            training = df.loc[list(combo_indices)].copy()
            heldout = same_all[~same_all.index.isin(combo_indices)].copy()
            if len(heldout) < TEST_PER_STRATUM:
                continue
            copy_scores = []
            for _, row in heldout.iterrows():
                profile = row["profile"]
                delta = copy_delta(profile, target, pair, training)
                p = target_probability(np.clip(profile + delta, 0.0, 1.0), target, model, rep_case)
                true_delta = row["delta"]
                copy_l1 = float(np.abs(delta).sum())
                # High when copy crosses slightly and remains close to the
                # boundary, with a small penalty for memorizing huge moves.
                copy_scores.append(
                    (
                        4.0 * float(p >= 0.5)
                        - 8.0 * abs(p - 0.525)
                        - 1.0 * abs(copy_l1 - float(row["cf_l1"]))
                        - 0.5 * float(np.abs(delta - true_delta).sum()),
                        p,
                    )
                )
            best_tests = sorted(copy_scores, reverse=True)[:TEST_PER_STRATUM]
            deltas = np.vstack(training["delta"].to_numpy())
            consistency = float(np.mean(np.sum(np.abs(deltas - deltas.mean(axis=0)), axis=1)))
            score = float(np.mean([s for s, _p in best_tests]) - 0.75 * consistency)
            scored_combos.append((score, list(combo_indices)))
        if not scored_combos:
            raise RuntimeError(f"No viable training combos for {stratum}")
        top_by_stratum[stratum] = [
            df.loc[idxs].copy()
            for _score, idxs in sorted(scored_combos, reverse=True)[:3]
        ]

    for combo in itertools.product(*(top_by_stratum[stratum] for stratum in STRATA)):
        yield pd.concat(combo, ignore_index=True)


def choose_testing(training: pd.DataFrame, df: pd.DataFrame, model: dict, rep_case: dict) -> pd.DataFrame:
    used = set(zip(training["source_split"], training["instance_id"]))
    w, b = ridge_linear_weights(training)
    chosen = []
    for pair, target in STRATA:
        pool = df[
            df["main_pair"].eq(pair)
            & df["target_label"].eq(target)
            & ~df.set_index(["source_split", "instance_id"]).index.isin(used)
            & df["cf_l1"].between(0.06, 0.32)
            & df["target_prob_cf_checked"].between(0.50, 0.80)
        ].copy()
        if len(pool) < TEST_PER_STRATUM:
            pool = df[
                df["main_pair"].eq(pair)
                & df["target_label"].eq(target)
                & ~df.set_index(["source_split", "instance_id"]).index.isin(used)
                & df["cf_l1"].between(0.02, 0.85)
                & df["target_prob_cf_checked"].between(0.50, 0.95)
            ].copy()
        scored = []
        for _, row in pool.iterrows():
            profile = row["profile"]
            true_delta = row["delta"]
            c_delta = copy_delta(profile, target, pair, training)
            l_delta = minimal_linear_delta(profile, target, pair, w, b)
            p_delta = prototype_delta(profile, target, pair, training)
            copy_p = target_probability(np.clip(profile + c_delta, 0, 1), target, model, rep_case)
            lin_p = target_probability(np.clip(profile + l_delta, 0, 1), target, model, rep_case)
            proto_p = target_probability(np.clip(profile + p_delta, 0, 1), target, model, rep_case)
            score = (
                +8.0 * float(copy_p >= 0.5)
                -12.0 * abs(copy_p - 0.525)
                -8.0 * max(0.5 - copy_p, 0.0)
                +2.0 * abs(lin_p - 0.5)
                +2.0 * abs(proto_p - 0.5)
                +1.5 * (np.abs(l_delta).sum() - np.abs(c_delta).sum())
                +1.5 * (np.abs(p_delta).sum() - np.abs(l_delta).sum())
                -1.0 * np.abs(c_delta - true_delta).sum()
            )
            scored.append((score, row.name))
        if len(scored) < TEST_PER_STRATUM:
            raise RuntimeError(f"Not enough testing candidates for {pair}, {target}: {len(scored)}")
        keep = [idx for _score, idx in sorted(scored, reverse=True)[:TEST_PER_STRATUM]]
        chosen.append(df.loc[keep])
    return pd.concat(chosen, ignore_index=True)


def choose_relaxed_contrast_testing(
    training: pd.DataFrame,
    df: pd.DataFrame,
    model: dict,
    rep_case: dict,
    *,
    n: int = 8,
) -> pd.DataFrame:
    used = set(zip(training["source_split"], training["instance_id"]))
    w, b = ridge_linear_weights(training)
    scored = []
    candidates = df[
        ~df.set_index(["source_split", "instance_id"]).index.isin(used)
        & df["target_prob_cf_checked"].between(0.50, 0.95)
    ].copy()
    for _, row in candidates.iterrows():
        profile = row["profile"]
        pair = row["main_pair"]
        target = row["target_label"]
        c_delta = copy_delta(profile, target, pair, training)
        l_delta = minimal_linear_delta(profile, target, pair, w, b)
        p_delta = prototype_delta(profile, target, pair, training)
        copy_p = target_probability(np.clip(profile + c_delta, 0, 1), target, model, rep_case)
        lin_p = target_probability(np.clip(profile + l_delta, 0, 1), target, model, rep_case)
        proto_p = target_probability(np.clip(profile + p_delta, 0, 1), target, model, rep_case)
        copy_l1 = float(np.abs(c_delta).sum())
        lin_l1 = float(np.abs(l_delta).sum())
        proto_l1 = float(np.abs(p_delta).sum())
        score = (
            10.0 * float(copy_p >= 0.5)
            - 14.0 * abs(copy_p - 0.525)
            + 2.0 * abs(lin_p - 0.5)
            + 2.0 * abs(proto_p - 0.5)
            + 1.0 * max(lin_l1 - copy_l1, 0.0)
            + 1.0 * max(proto_l1 - copy_l1, 0.0)
        )
        scored.append((score, row.name))
    keep = [idx for _score, idx in sorted(scored, reverse=True)[:n]]
    return df.loc[keep].copy().reset_index(drop=True)


def exact_boundary_distances(trials: pd.DataFrame, testing: pd.DataFrame, training: pd.DataFrame, model: dict, rep_case: dict) -> pd.DataFrame:
    w, b = ridge_linear_weights(training)
    case_lookup = {
        (row["source_split"], int(row["instance_id"])): row
        for _, row in testing.iterrows()
    }
    solver = WachterBoundarySearch(model, rep_case, allow_reference_outside_bounds=True)
    exact = []
    for _, t in trials.iterrows():
        row = case_lookup[(t["source_split"], int(t["source_instance_id"]))]
        profile = row["profile"]
        pair = row["main_pair"]
        target = row["target_label"]
        if t["strategy"] == "copy moderate λ=2":
            delta = copy_delta(profile, target, pair, training)
        elif t["strategy"] == "linear mental model":
            delta = minimal_linear_delta(profile, target, pair, w, b)
        elif t["strategy"] == "target prototype":
            delta = prototype_delta(profile, target, pair, training)
        else:
            delta = row["delta"]
        edited = np.clip(profile + delta, 0.0, 1.0)
        exact.append(solver.solve(edited).distance)
    out = trials.copy()
    out["exact_boundary_distance_after_edit"] = exact
    return out


def compact_case_table(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), start=1):
        rows.append({
            "phase": phase,
            "rank": rank,
            "source_split": r["source_split"],
            "source_instance_id": int(r["instance_id"]),
            "prediction_label": r["prediction_label"],
            "target_label": r["target_label"],
            "pair": r["main_pair"],
            "original_profile": r["original_profile"],
            "counterfactual_changes": r["counterfactual_changes"],
            "cf_L1": float(r["cf_l1"]),
            "cf_target_probability": float(r["target_prob_cf_checked"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df, model, rep_case = load_candidates()
    pools = consistent_training_pool(df)
    print("candidate pools by pair/target:")
    for key, pool in pools.items():
        print(f"  {key}: {len(pool)}")

    best = None
    searched = 0
    for training in candidate_training_sets(pools, df, model, rep_case):
        searched += 1
        try:
            testing = choose_testing(training, df, model, rep_case)
        except RuntimeError:
            continue
        trials, summary = eval_design(training, testing, model, rep_case)
        score = design_objective(summary)
        if best is None or score > best[0]:
            best = (score, training, testing, trials, summary)

    if best is None:
        raise RuntimeError("No feasible design found.")

    score, training, testing, trials, summary = best
    exact_trials = exact_boundary_distances(trials, testing, training, model, rep_case)
    exact_summary = (
        exact_trials.groupby("strategy", as_index=False)
        .agg(
            n=("source_instance_id", "count"),
            mean_edit_L1=("edit_L1", "mean"),
            median_edit_L1=("edit_L1", "median"),
            success_rate=("success", "mean"),
            mean_target_probability=("target_probability", "mean"),
            mean_exact_boundary_after=("exact_boundary_distance_after_edit", "mean"),
            median_exact_boundary_after=("exact_boundary_distance_after_edit", "median"),
        )
        .sort_values("mean_exact_boundary_after")
    )

    train_table = compact_case_table(training, "training")
    test_table = compact_case_table(testing, "testing")

    train_table.to_csv(OUTDIR / "selected_training_cases.csv", index=False)
    test_table.to_csv(OUTDIR / "selected_testing_cases.csv", index=False)
    exact_trials.to_csv(OUTDIR / "strategy_trials.csv", index=False)
    exact_summary.to_csv(OUTDIR / "strategy_summary.csv", index=False)

    relaxed_testing = choose_relaxed_contrast_testing(training, df, model, rep_case, n=8)
    relaxed_trials, _relaxed_proxy_summary = eval_design(training, relaxed_testing, model, rep_case)
    relaxed_exact_trials = exact_boundary_distances(relaxed_trials, relaxed_testing, training, model, rep_case)
    relaxed_exact_summary = (
        relaxed_exact_trials.groupby("strategy", as_index=False)
        .agg(
            n=("source_instance_id", "count"),
            mean_edit_L1=("edit_L1", "mean"),
            median_edit_L1=("edit_L1", "median"),
            success_rate=("success", "mean"),
            mean_target_probability=("target_probability", "mean"),
            mean_exact_boundary_after=("exact_boundary_distance_after_edit", "mean"),
            median_exact_boundary_after=("exact_boundary_distance_after_edit", "median"),
        )
        .sort_values("mean_exact_boundary_after")
    )
    compact_case_table(relaxed_testing, "testing").to_csv(
        OUTDIR / "selected_testing_cases_relaxed_top8.csv", index=False
    )
    relaxed_exact_trials.to_csv(OUTDIR / "strategy_trials_relaxed_top8.csv", index=False)
    relaxed_exact_summary.to_csv(OUTDIR / "strategy_summary_relaxed_top8.csv", index=False)
    (OUTDIR / "selection_summary.json").write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "candidate_file": str(CANDIDATES.relative_to(ROOT)),
                "searched_training_designs": searched,
                "copy_lambda": COPY_LAMBDA,
                "objective_score": score,
                "heuristics": [
                    "two non-age feature clusters only",
                    "training counterfactuals moderate size and close to the boundary",
                    "training edits internally consistent within pair+target",
                    "testing cases where broad copy-template lands close to boundary",
                    "testing cases where linear/prototype strategies require larger edits or end farther from boundary",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nsearched training designs: {searched}")
    print(f"best objective score: {score:.3f}")
    print("\nEXACT STRATEGY SUMMARY")
    print(exact_summary.round(4).to_string(index=False))
    print("\nSELECTED TRAINING CASES")
    print(train_table[["rank", "source_split", "source_instance_id", "prediction_label", "target_label", "pair", "cf_L1", "cf_target_probability", "counterfactual_changes"]].round(4).to_string(index=False))
    print("\nSELECTED TESTING CASES")
    print(test_table[["rank", "source_split", "source_instance_id", "prediction_label", "target_label", "pair", "cf_L1", "cf_target_probability", "counterfactual_changes"]].round(4).to_string(index=False))
    print("\nRELAXED TOP8 CONTRAST SUMMARY")
    print(relaxed_exact_summary.round(4).to_string(index=False))
    print("\nRELAXED TOP8 TESTING CASES")
    relaxed_table = compact_case_table(relaxed_testing, "testing")
    print(relaxed_table[["rank", "source_split", "source_instance_id", "prediction_label", "target_label", "pair", "cf_L1", "cf_target_probability", "counterfactual_changes"]].round(4).to_string(index=False))
    print(f"\nwrote: {OUTDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
