from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "outputs" / "v20-v01-theoretical-parameter-sweeps" / "theoretical_sweep_trials.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTDIR = ROOT / "outputs" / "v20-cognitive-design-signal"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRIALS)
dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
cases = {int(c["instance_id"]): c for c in dataset["test_pool"]}

# Focus on the interpretable regime: K=2, age not actionable. Keep all family/parameter
# values otherwise and ask whether CF wins robustly across those settings.
df = df[df["age actionable"].eq(0)].copy()

records = []
for (family, instance_id), g in df.groupby(["model family", "instance id"]):
    case = cases[int(instance_id)]
    pivot = (
        g.groupby("xai")
        .agg(
            success=("success", "mean"),
            boundary_improvement=("boundary improvement", "mean"),
            edit_l1=("edit L1", "mean"),
            plausibility=("plausibility", "mean"),
            target_confidence_gain=("target confidence gain", "mean"),
            map_prob=("MAP subset probability", "mean"),
        )
    )
    if not {"none", "attribution", "counterfactual"}.issubset(pivot.index):
        continue
    rec = {
        "model family": family,
        "instance id": int(instance_id),
        "original label": case["prediction"]["label"],
        "target label": "No Diabetes" if case["prediction"]["label"] == "Diabetes" else "Diabetes",
        "feature_pair": "|".join(case.get("feature_pair_names", [])),
        "nearest_training_source": case.get("nearest_training_source"),
        "distance_to_nearest_training": case.get("profile_distance_to_nearest_training"),
    }
    for xai in ["none", "attribution", "counterfactual"]:
        for metric in pivot.columns:
            rec[f"{xai}_{metric}"] = float(pivot.loc[xai, metric])
    rec["cf_minus_none_boundary_improvement"] = rec["counterfactual_boundary_improvement"] - rec["none_boundary_improvement"]
    rec["cf_minus_attr_boundary_improvement"] = rec["counterfactual_boundary_improvement"] - rec["attribution_boundary_improvement"]
    rec["attr_minus_none_boundary_improvement"] = rec["attribution_boundary_improvement"] - rec["none_boundary_improvement"]
    rec["cf_minus_none_success"] = rec["counterfactual_success"] - rec["none_success"]
    rec["cf_minus_attr_success"] = rec["counterfactual_success"] - rec["attribution_success"]
    rec["cf_specific_score"] = (
        rec["cf_minus_none_boundary_improvement"]
        + rec["cf_minus_attr_boundary_improvement"]
        + 0.1 * rec["cf_minus_none_success"]
        + 0.1 * rec["cf_minus_attr_success"]
    )
    records.append(rec)

out = pd.DataFrame(records).sort_values("cf_specific_score", ascending=False)
out.to_csv(OUTDIR / "current_instances_v01_cf_advantage_by_family.csv", index=False)

agg = (
    out.groupby(["instance id", "original label", "target label", "feature_pair"], as_index=False)
    .agg(
        cf_specific_score=("cf_specific_score", "mean"),
        cf_minus_none_boundary_improvement=("cf_minus_none_boundary_improvement", "mean"),
        cf_minus_attr_boundary_improvement=("cf_minus_attr_boundary_improvement", "mean"),
        cf_minus_none_success=("cf_minus_none_success", "mean"),
        cf_minus_attr_success=("cf_minus_attr_success", "mean"),
        distance_to_nearest_training=("distance_to_nearest_training", "first"),
    )
    .sort_values("cf_specific_score", ascending=False)
)
agg.to_csv(OUTDIR / "current_instances_v01_cf_advantage_aggregated.csv", index=False)

print("CURRENT INSTANCES WHERE v0.1 MOST FAVORS COUNTERFACTUAL")
print(agg.head(10).round(4).to_string(index=False))
print("\nCURRENT INSTANCES WHERE v0.1 LEAST FAVORS COUNTERFACTUAL")
print(agg.tail(10).round(4).to_string(index=False))

print("\nBY TARGET LABEL")
print(
    agg.groupby("target label")[
        [
            "cf_specific_score",
            "cf_minus_none_boundary_improvement",
            "cf_minus_attr_boundary_improvement",
            "cf_minus_none_success",
            "cf_minus_attr_success",
            "distance_to_nearest_training",
        ]
    ]
    .mean()
    .round(4)
    .to_string()
)
print(f"\nWrote {OUTDIR}")
