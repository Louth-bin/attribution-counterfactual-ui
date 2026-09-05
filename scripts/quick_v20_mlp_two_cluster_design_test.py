from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("V19_BOUNDARY_WORKERS", "1")

from scripts.evaluate_v19_latest_model_simulations import (  # noqa: E402
    exact_boundaries,
    prediction_label,
)
from src.cognitive_models.v0_1.model import (  # noqa: E402
    build_memory,
    contribution_distribution,
    exemplar_distribution,
    label_sign,
    normalize_case,
)


BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTDIR = ROOT / "outputs" / "v20-mlp-two-cluster-design-test"
OUTDIR.mkdir(parents=True, exist_ok=True)

A = "Glucose|BMI"
B = "Blood Pressure|Insulin"


def pair(case: dict) -> str:
    return "|".join(case.get("feature_pair_names", []))


def params_for(family: str, xai: str, regime: str) -> dict:
    # Small set of plausible regimes rather than exhaustive sweep.
    if regime == "minimal_local":
        p = {"eta": 1.0 if xai != "none" else 0.0, "alpha": 0.0, "rho": 0.0, "lambda": 6.0, "beta": 0.0}
    elif regime == "moderate_local":
        p = {"eta": 1.0 if xai != "none" else 0.0, "alpha": 0.0, "rho": 0.1, "lambda": 6.0, "beta": 0.0}
    elif regime == "moderate_global":
        p = {"eta": 1.0 if xai != "none" else 0.0, "alpha": 1.0, "rho": 0.1, "lambda": 2.0, "beta": 0.5}
    else:
        raise ValueError(regime)
    p["age actionable"] = 0
    p["immutable index"] = 4
    if family == "feature contribution":
        p["lambda"] = math.nan
        p["beta"] = math.nan
    else:
        p["alpha"] = math.nan
        if xai != "counterfactual":
            p["beta"] = 0.0
    return p


def simulate_design(name: str, train_cases: list[dict], test_cases: list[dict], dataset: dict) -> pd.DataFrame:
    browser_model = dataset["browser_model"]
    profiles_to_solve = {}
    rows = []
    for family in ["weighted examples", "feature contribution"]:
        for regime in ["minimal_local", "moderate_local", "moderate_global"]:
            for xai in ["none", "attribution", "counterfactual"]:
                params = params_for(family, xai, regime)
                memory = build_memory(train_cases, xai, params["eta"])
                for case in test_cases:
                    original = normalize_case(case)
                    target = -label_sign(case["prediction"]["label"])
                    k = 2
                    if family == "weighted examples":
                        dist = exemplar_distribution(original, target, memory, xai, params, k)
                    else:
                        dist = contribution_distribution(original, target, memory, params, k)
                    if not dist:
                        continue
                    subset, subset_prob, delta = max(dist, key=lambda item: item[1])
                    edited = np.clip(original + delta, 0.0, 1.0)
                    key_o = f"{name}|{family}|{regime}|{xai}|{case['instance_id']}|orig"
                    key_e = f"{name}|{family}|{regime}|{xai}|{case['instance_id']}|edit"
                    profiles_to_solve[key_o] = original
                    profiles_to_solve[key_e] = edited
                    rows.append(
                        {
                            "design": name,
                            "model family": family,
                            "regime": regime,
                            "xai": xai,
                            "instance id": int(case["instance_id"]),
                            "original label": case["prediction"]["label"],
                            "target label": "No Diabetes" if case["prediction"]["label"] == "Diabetes" else "Diabetes",
                            "feature_pair": pair(case),
                            "MAP subset": "|".join(str(i) for i in subset),
                            "MAP subset probability": subset_prob,
                            "success": float(prediction_label(edited, case, browser_model) != case["prediction"]["label"]),
                            "edit L1": float(np.abs(delta).sum()),
                            "_orig_key": key_o,
                            "_edit_key": key_e,
                        }
                    )
    boundaries = exact_boundaries(browser_model, dataset["test_pool"][0], profiles_to_solve)
    for row in rows:
        row["original boundary distance"] = boundaries[row.pop("_orig_key")]
        row["final boundary distance"] = boundaries[row.pop("_edit_key")]
        row["boundary improvement"] = row["original boundary distance"] - row["final boundary distance"]
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    summary = (
        trials.groupby(["design", "model family", "regime", "xai"], as_index=False)
        .agg(
            n_instances=("instance id", "nunique"),
            mean_original_boundary=("original boundary distance", "mean"),
            mean_final_boundary=("final boundary distance", "mean"),
            mean_boundary_improvement=("boundary improvement", "mean"),
            success_rate=("success", "mean"),
            mean_edit_L1=("edit L1", "mean"),
            mean_map_prob=("MAP subset probability", "mean"),
        )
    )
    records = []
    for keys, g in summary.groupby(["design", "model family", "regime"]):
        by = g.set_index("xai")
        if not {"none", "attribution", "counterfactual"}.issubset(by.index):
            continue
        rec = dict(zip(["design", "model family", "regime"], keys))
        for xai in ["none", "attribution", "counterfactual"]:
            for c in ["mean_final_boundary", "mean_boundary_improvement", "success_rate", "mean_edit_L1"]:
                rec[f"{xai}_{c}"] = float(by.loc[xai, c])
        rec["cf_minus_none_improvement"] = rec["counterfactual_mean_boundary_improvement"] - rec["none_mean_boundary_improvement"]
        rec["cf_minus_attr_improvement"] = rec["counterfactual_mean_boundary_improvement"] - rec["attribution_mean_boundary_improvement"]
        rec["attr_minus_none_improvement"] = rec["attribution_mean_boundary_improvement"] - rec["none_mean_boundary_improvement"]
        rec["cf_final_boundary_advantage_vs_none"] = rec["none_mean_final_boundary"] - rec["counterfactual_mean_final_boundary"]
        rec["cf_final_boundary_advantage_vs_attr"] = rec["attribution_mean_final_boundary"] - rec["counterfactual_mean_final_boundary"]
        rec["ordering_cf_attr_none"] = (
            rec["counterfactual_mean_final_boundary"]
            < rec["attribution_mean_final_boundary"]
            < rec["none_mean_final_boundary"]
        )
        rec["cf_specific_score"] = (
            rec["cf_final_boundary_advantage_vs_none"]
            + rec["cf_final_boundary_advantage_vs_attr"]
            + 0.1 * (rec["counterfactual_success_rate"] - rec["none_success_rate"])
            + 0.1 * (rec["counterfactual_success_rate"] - rec["attribution_success_rate"])
        )
        records.append(rec)
    return pd.DataFrame(records).sort_values("cf_specific_score", ascending=False)


def main() -> None:
    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    train = dataset["training_pool"]
    test = dataset["test_pool"]
    train_a = [c for c in train if pair(c) == A]
    train_b = [c for c in train if pair(c) == B]
    test_a = [c for c in test if pair(c) == A]
    test_b = [c for c in test if pair(c) == B]
    test_ab = [c for c in test if pair(c) in {A, B}]
    test_mixed_bad = [c for c in test if pair(c) not in {A, B}]

    designs = [
        ("current_train_current_test", train, test),
        ("current_train_same_pair_tests", train, test_ab),
        ("A_train_A_tests", train_a, test_a),
        ("B_train_B_tests", train_b, test_b),
        ("A_train_B_tests_transfer_bad", train_a, test_b),
        ("B_train_A_tests_transfer_bad", train_b, test_a),
        ("current_train_off_cluster_tests", train, test_mixed_bad),
    ]
    all_trials = []
    for name, tr, te in designs:
        if len(tr) == 0 or len(te) == 0:
            continue
        print(f"simulate {name}: train={len(tr)} test={len(te)}", flush=True)
        all_trials.append(simulate_design(name, tr, te, dataset))
    trials = pd.concat(all_trials, ignore_index=True)
    trials.to_csv(OUTDIR / "two_cluster_design_trials.csv", index=False)
    comp = summarize(trials)
    comp.to_csv(OUTDIR / "two_cluster_design_summary.csv", index=False)

    # Also make a model-screened current-test shortlist: top tests by average CF
    # final-boundary advantage under current training.
    cur = trials[trials["design"].eq("current_train_current_test")]
    per_instance = []
    for (iid, family, regime), g in cur.groupby(["instance id", "model family", "regime"]):
        by = g.set_index("xai")
        if not {"none", "attribution", "counterfactual"}.issubset(by.index):
            continue
        case = next(c for c in test if int(c["instance_id"]) == int(iid))
        cf_adv_none = by.loc["none", "final boundary distance"] - by.loc["counterfactual", "final boundary distance"]
        cf_adv_attr = by.loc["attribution", "final boundary distance"] - by.loc["counterfactual", "final boundary distance"]
        per_instance.append(
            {
                "instance id": int(iid),
                "target label": "No Diabetes" if case["prediction"]["label"] == "Diabetes" else "Diabetes",
                "feature_pair": pair(case),
                "model family": family,
                "regime": regime,
                "cf_adv_vs_none_final_boundary": cf_adv_none,
                "cf_adv_vs_attr_final_boundary": cf_adv_attr,
                "cf_specific_score": cf_adv_none + cf_adv_attr,
            }
        )
    inst = (
        pd.DataFrame(per_instance)
        .groupby(["instance id", "target label", "feature_pair"], as_index=False)
        .agg(
            cf_adv_vs_none_final_boundary=("cf_adv_vs_none_final_boundary", "mean"),
            cf_adv_vs_attr_final_boundary=("cf_adv_vs_attr_final_boundary", "mean"),
            cf_specific_score=("cf_specific_score", "mean"),
        )
        .sort_values("cf_specific_score", ascending=False)
    )
    inst.to_csv(OUTDIR / "model_screened_current_test_instances.csv", index=False)

    shortlist_ids = set(inst.head(8)["instance id"])
    shortlist = [c for c in test if int(c["instance_id"]) in shortlist_ids]
    shortlist_trials = simulate_design("model_screened_top8_current_train", train, shortlist, dataset)
    shortlist_trials.to_csv(OUTDIR / "model_screened_top8_trials.csv", index=False)
    shortlist_summary = summarize(shortlist_trials)
    shortlist_summary.to_csv(OUTDIR / "model_screened_top8_summary.csv", index=False)

    print("\nNON-CIRCULAR DESIGN SUMMARY: TOP ROWS")
    print(comp.head(20).round(3).to_string(index=False))
    print("\nMODEL-SCREENED TOP CURRENT INSTANCES")
    print(inst.head(12).round(3).to_string(index=False))
    print("\nMODEL-SCREENED TOP8 SUMMARY")
    print(shortlist_summary.round(3).to_string(index=False))
    print(f"\nWrote {OUTDIR}")


if __name__ == "__main__":
    main()
