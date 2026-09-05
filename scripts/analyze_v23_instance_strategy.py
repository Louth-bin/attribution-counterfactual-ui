"""Instance-level XAI comparison and cognitive-strategy outcome summary for v2.3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr, ttest_ind


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "qualtrics" / "qualtrics_results_v2.3.csv"
FITS = ROOT / "qualtrics" / "v2.3_cognitive_strategy_participants.csv"
OUT_INSTANCE = ROOT / "qualtrics" / "v2.3_instance_attribution_vs_counterfactual.csv"
OUT_STRATEGY = ROOT / "qualtrics" / "v2.3_strategy_outcome_summary.csv"
OUT_JSON = ROOT / "qualtrics" / "v2.3_instance_strategy_findings.json"


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def structural_cases() -> pd.DataFrame:
    bundle = json.loads((ROOT / "static" / "experiment-data.json").read_text(encoding="utf-8"))
    cases = bundle["datasets"]["diabetes"]["test_pool"]
    rows = []
    for case in cases:
        instance = str(case["instance_id"])
        if not instance.startswith("130"):
            continue
        names = case["raw_feature_names"]
        ranges = case["raw_feature_ranges"]
        original = np.asarray(case["raw_feature_values"], float)
        edited = np.asarray(case["counterfactual"]["raw_feature_values"], float)
        normalized_delta = np.abs(edited - original) / np.asarray(
            [high - low for low, high in ranges], float
        )
        shown = set(case["attribution"].get("shown_feature_indices", []))
        changed = {i for i, value in enumerate(normalized_delta) if value > 1e-12}
        overlap = len(shown & changed) / len(shown | changed) if shown | changed else 1.0
        attr_values = np.sort(np.abs(np.asarray(case["attribution"]["raw_values"], float)))[::-1]
        rows.append(
            {
                "instance id": int(instance),
                "designed CF normalized L1": normalized_delta.sum(),
                "designed CF max normalized change": normalized_delta.max(),
                "designed CF changed features": len(changed),
                "designed CF includes age": int(names.index("age") in changed),
                "attribution-CF feature overlap (Jaccard)": overlap,
                "attribution top-two separation": attr_values[0] - attr_values[1],
                "designed CF feature pair": "|".join(names[i] for i in sorted(changed)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_csv(DATA)
    test = data.loc[data["phase"].eq("testing")].copy()
    participant_direction = test.groupby(["participant", "xai"])[
        "participant level correct direction change"
    ].first()
    excluded = participant_direction[
        (participant_direction.index.get_level_values("xai") != "none")
        & (participant_direction < 0.60)
    ].index.get_level_values("participant").tolist()
    retained = test.loc[~test["participant"].isin(excluded)].copy()

    comparison_rows = []
    for instance, group in retained.loc[retained["xai"].isin(["attribution", "counterfactual"])].groupby("instance id"):
        attr = group.loc[group["xai"].eq("attribution")]
        cf = group.loc[group["xai"].eq("counterfactual")]
        a_success = int(attr["successful CF"].sum())
        c_success = int(cf["successful CF"].sum())
        table = [[a_success, len(attr) - a_success], [c_success, len(cf) - c_success]]
        fisher_p = fisher_exact(table)[1]
        confidence_p = ttest_ind(
            attr["delta confidence of target label"],
            cf["delta confidence of target label"],
            equal_var=False,
        ).pvalue
        row = {
            "instance id": int(instance),
            "original label": group["original label"].iloc[0],
            "target label": group["target label"].iloc[0],
            "abs original boundary distance": group["abs dist original to boundary"].iloc[0],
            "nearest training distance": group["dist to nearest training instance (L1 normalized)"].iloc[0],
            "attribution n": len(attr),
            "counterfactual n": len(cf),
            "attribution success": attr["successful CF"].mean(),
            "counterfactual success": cf["successful CF"].mean(),
            "success difference (attribution - counterfactual)": attr["successful CF"].mean() - cf["successful CF"].mean(),
            "success Fisher p": fisher_p,
            "attribution confidence change": attr["delta confidence of target label"].mean(),
            "counterfactual confidence change": cf["delta confidence of target label"].mean(),
            "confidence difference (attribution - counterfactual)": attr["delta confidence of target label"].mean() - cf["delta confidence of target label"].mean(),
            "confidence Welch p": confidence_p,
            "attribution sparsity": attr["sparsity"].mean(),
            "counterfactual sparsity": cf["sparsity"].mean(),
            "sparsity difference (attribution - counterfactual)": attr["sparsity"].mean() - cf["sparsity"].mean(),
            "attribution edit distance": attr["dist user CF from orig"].mean(),
            "counterfactual edit distance": cf["dist user CF from orig"].mean(),
            "edit distance difference (attribution - counterfactual)": attr["dist user CF from orig"].mean() - cf["dist user CF from orig"].mean(),
            "attribution plausibility": attr["plausibility"].mean(),
            "counterfactual plausibility": cf["plausibility"].mean(),
            "plausibility difference (attribution - counterfactual)": attr["plausibility"].mean() - cf["plausibility"].mean(),
        }
        comparison_rows.append(row)
    instances = pd.DataFrame(comparison_rows).merge(structural_cases(), on="instance id", how="left")
    instances["success Fisher FDR"] = bh_adjust(instances["success Fisher p"])
    instances["confidence Welch FDR"] = bh_adjust(instances["confidence Welch p"])
    instances["same advantage direction"] = np.sign(instances["success difference (attribution - counterfactual)"]) == np.sign(instances["confidence difference (attribution - counterfactual)"])
    instances.to_csv(OUT_INSTANCE, index=False)

    fits = pd.read_csv(FITS)
    participant_outcomes = retained.groupby(["participant", "xai"], as_index=False).agg(
        success=("successful CF", "mean"),
        confidence_change=("delta confidence of target label", "mean"),
        sparsity=("sparsity", "mean"),
        edit_distance=("dist user CF from orig", "mean"),
        plausibility=("plausibility", "mean"),
        direction_accuracy=("user CF increase target label confidence?", "mean"),
    )
    participant_outcomes["confidence per normalized edit"] = (
        participant_outcomes["confidence_change"] / participant_outcomes["edit_distance"]
    )
    joined = fits.merge(participant_outcomes, on=["participant", "xai"], how="inner")
    strategy_col = "best_counterfactual_strategy"
    summary = joined.groupby(strategy_col).agg(
        participants=("participant", "size"),
        mean_success=("success", "mean"),
        median_success=("success", "median"),
        mean_confidence_change=("confidence_change", "mean"),
        mean_sparsity=("sparsity", "mean"),
        mean_edit_distance=("edit_distance", "mean"),
        mean_confidence_per_edit=("confidence per normalized edit", "mean"),
        mean_plausibility=("plausibility", "mean"),
        mean_direction_accuracy=("direction_accuracy", "mean"),
        mean_fit_loss=("best_strategy_cv_mean_normalized_l1", "mean"),
        mean_fit_gap=("counterfactual_strategy_gap_to_second_best", "mean"),
    ).reset_index().sort_values(["mean_success", "mean_confidence_change"], ascending=False)
    summary.to_csv(OUT_STRATEGY, index=False)

    structural_columns = [
        "abs original boundary distance",
        "nearest training distance",
        "designed CF normalized L1",
        "designed CF max normalized change",
        "designed CF includes age",
        "attribution-CF feature overlap (Jaccard)",
        "attribution top-two separation",
    ]
    relationships = {}
    for outcome in [
        "success difference (attribution - counterfactual)",
        "confidence difference (attribution - counterfactual)",
    ]:
        relationships[outcome] = {}
        for predictor in structural_columns:
            rho, p = spearmanr(instances[predictor], instances[outcome])
            relationships[outcome][predictor] = {"spearman rho": rho, "p": p}

    payload = {
        "excluded_participants": excluded,
        "retained_participants": int(retained["participant"].nunique()),
        "attribution_better_success_instances": instances.loc[
            instances["success difference (attribution - counterfactual)"] > 0,
            "instance id",
        ].astype(int).tolist(),
        "counterfactual_better_success_instances": instances.loc[
            instances["success difference (attribution - counterfactual)"] < 0,
            "instance id",
        ].astype(int).tolist(),
        "ties_success_instances": instances.loc[
            instances["success difference (attribution - counterfactual)"].eq(0),
            "instance id",
        ].astype(int).tolist(),
        "same_sign_success_and_confidence_count": int(instances["same advantage direction"].sum()),
        "instance_structural_correlations": relationships,
        "strategy_fit_participants": int(len(fits)),
        "retained_strategy_participants": int(len(joined)),
        "unique_one_se_strategy_assignments": int(
            fits["supported_counterfactual_strategies_one_se"].fillna("").str.contains(" / ").eq(False).sum()
        ),
        "strategy_counts_retained": joined[strategy_col].value_counts().to_dict(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\nTop attribution success advantages:\n", instances.nlargest(6, "success difference (attribution - counterfactual)")[["instance id", "success difference (attribution - counterfactual)", "confidence difference (attribution - counterfactual)", "success Fisher p", "success Fisher FDR"]].to_string(index=False))
    print("\nTop counterfactual success advantages:\n", instances.nsmallest(6, "success difference (attribution - counterfactual)")[["instance id", "success difference (attribution - counterfactual)", "confidence difference (attribution - counterfactual)", "success Fisher p", "success Fisher FDR"]].to_string(index=False))
    print("\nStrategy summary:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
