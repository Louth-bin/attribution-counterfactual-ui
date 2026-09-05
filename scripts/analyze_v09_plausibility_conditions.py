"""Diagnose condition differences in v0.9 nearest-training plausibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.spatial import cKDTree

import calculate_v09_plausibility as plausibility_calc


OUTPUT_PATH = Path(plausibility_calc.TEMP_DIR) / "v09_plausibility_condition_analysis.json"

FEATURE_LABELS = {
    "housing": [
        "Living Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Construction Grade",
    ],
    "diabetes": ["Blood Glucose", "Blood Pressure", "Insulin", "BMI", "Age"],
    "safelimit": [
        "Alcohol Units",
        "Weight",
        "Drinking Duration",
        "Gender",
        "Stomach Fullness",
    ],
}


def clean_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def original_normalized_profile(attribute_text: str) -> np.ndarray:
    """Return the five normalized values before the participant's changes."""
    lines = [line for line in attribute_text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError(f"Expected five attribute lines, found {len(lines)}")
    values: list[float] = []
    for line in lines:
        matches = plausibility_calc.NORMALIZED_VALUE_RE.findall(line)
        if not matches:
            raise ValueError(f"No normalized value found in {line!r}")
        values.append(float(matches[0]))
    return np.asarray(values, dtype=float)


def fit_condition_model(data: pd.DataFrame, formula: str) -> dict[str, object]:
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["participant"]}
    )
    terms: dict[str, object] = {}
    for condition in ("attribution", "counterfactual"):
        term = f'C(xai, Treatment(reference="none"))[T.{condition}]'
        terms[condition] = {
            "coefficient_vs_none": float(model.params.get(term, np.nan)),
            "standard_error": float(model.bse.get(term, np.nan)),
            "p_value": float(model.pvalues.get(term, np.nan)),
        }
    return {"formula": formula, "r_squared": float(model.rsquared), "terms": terms}


def records(frame: pd.DataFrame, digits: int = 6) -> list[dict[str, object]]:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(digits)
    return result.replace({np.nan: None}).to_dict(orient="records")


def main() -> None:
    base = pd.read_csv(plausibility_calc.INPUT_CSV)
    scores = pd.read_csv(plausibility_calc.OUTPUT_CSV)
    base.insert(0, "row_number", np.arange(1, len(base) + 1))
    merged = base.merge(
        scores[["row_number", "participant", "domain", "phase", "plausibility"]],
        on="row_number",
        how="left",
        suffixes=("", "_score"),
        validate="one_to_one",
    )
    for column in ("participant", "domain", "phase"):
        if not (merged[column].astype(str) == merged[f"{column}_score"].astype(str)).all():
            raise ValueError(f"Row mapping mismatch for {column}")

    testing = merged.loc[merged["phase"] == "testing"].copy()
    numeric_columns = [
        "plausibility",
        "proximity",
        "sparsity",
        "valid counterfactual (0/1)",
        "actionability (0/1)",
        *[f"x_{index}_change" for index in range(1, 6)],
        *[f"x_{index}_changed" for index in range(1, 6)],
    ]
    for column in numeric_columns:
        testing[column] = testing[column].map(clean_float)

    reference_matrices = plausibility_calc.load_reference_matrices()
    experiment = json.loads(
        plausibility_calc.EXPERIMENT_DATA.read_text(encoding="utf-8")
    )
    shown_indices_by_domain = {
        domain: {
            str(case["instance_id"]): tuple(case["attribution"]["shown_feature_indices"])
            for case in experiment["datasets"][domain]["test_pool"]
        }
        for domain in FEATURE_LABELS
    }
    final_profiles = np.zeros((len(testing), 5), dtype=float)
    original_profiles = np.zeros((len(testing), 5), dtype=float)
    nearest_gaps = np.zeros((len(testing), 5), dtype=float)
    original_nearest_gaps = np.zeros((len(testing), 5), dtype=float)
    nearest_indices = np.zeros(len(testing), dtype=int)
    original_plausibility = np.zeros(len(testing), dtype=float)
    plausibility_l2 = np.zeros(len(testing), dtype=float)
    plausibility_l1_k5 = np.zeros(len(testing), dtype=float)
    for output_index, (_, row) in enumerate(testing.iterrows()):
        profile = plausibility_calc.final_normalized_profile(
            row["attribute values before and after"]
        )
        original_profile = original_normalized_profile(
            row["attribute values before and after"]
        )
        differences = np.abs(reference_matrices[row["domain"]] - profile)
        distance = differences.sum(axis=1)
        nearest_index = int(np.argmin(distance))
        if not np.isclose(distance[nearest_index], row["plausibility"], atol=1e-9):
            raise ValueError(f"Nearest-distance mismatch at row {row['row_number']}")
        final_profiles[output_index] = profile
        original_profiles[output_index] = original_profile
        nearest_gaps[output_index] = differences[nearest_index]
        nearest_indices[output_index] = nearest_index
        original_differences = np.abs(
            reference_matrices[row["domain"]] - original_profile
        )
        original_distance = original_differences.sum(axis=1)
        original_nearest_index = int(np.argmin(original_distance))
        original_plausibility[output_index] = original_distance[original_nearest_index]
        original_nearest_gaps[output_index] = original_differences[
            original_nearest_index
        ]
        plausibility_l2[output_index] = np.sqrt((differences**2).sum(axis=1)).min()
        plausibility_l1_k5[output_index] = np.partition(distance, 4)[:5].mean()

    for index in range(1, 6):
        testing[f"final_x{index}"] = final_profiles[:, index - 1]
        testing[f"nn_gap_x{index}"] = nearest_gaps[:, index - 1]
        testing[f"original_x{index}"] = original_profiles[:, index - 1]
        testing[f"original_nn_gap_x{index}"] = original_nearest_gaps[:, index - 1]
        testing[f"nn_gap_delta_x{index}"] = (
            testing[f"nn_gap_x{index}"] - testing[f"original_nn_gap_x{index}"]
        )
        testing[f"abs_change_x{index}"] = testing[f"x_{index}_change"].abs()
        testing[f"shown_x{index}"] = testing.apply(
            lambda row: int(
                index - 1
                in shown_indices_by_domain[row["domain"]].get(
                    str(row["instance id"]), ()
                )
            ),
            axis=1,
        )
    testing["original_plausibility"] = original_plausibility
    testing["plausibility_delta_from_original"] = (
        testing["plausibility"] - testing["original_plausibility"]
    )
    testing["plausibility_l2"] = plausibility_l2
    testing["plausibility_l1_k5"] = plausibility_l1_k5
    testing["nearest_training_index"] = nearest_indices
    testing["change_pattern"] = testing.apply(
        lambda row: "+".join(
            FEATURE_LABELS[row["domain"]][index - 1]
            for index in range(1, 6)
            if row[f"x_{index}_changed"] == 1
        )
        or "No change",
        axis=1,
    )

    report: dict[str, object] = {
        "row_count": len(merged),
        "testing_count": len(testing),
        "domains": {},
    }

    for domain in ("housing", "diabetes", "safelimit"):
        subset = testing.loc[testing["domain"] == domain].copy()
        if subset.empty:
            continue

        condition_summary = (
            subset.groupby("xai", observed=True)
            .agg(
                rows=("plausibility", "size"),
                participants=("participant", "nunique"),
                plausibility_mean=("plausibility", "mean"),
                plausibility_median=("plausibility", "median"),
                plausibility_sd=("plausibility", "std"),
                original_plausibility_mean=("original_plausibility", "mean"),
                plausibility_delta_mean=("plausibility_delta_from_original", "mean"),
                plausibility_l2_mean=("plausibility_l2", "mean"),
                plausibility_l1_k5_mean=("plausibility_l1_k5", "mean"),
                proximity_mean=("proximity", "mean"),
                sparsity_mean=("sparsity", "mean"),
                validity_rate=("valid counterfactual (0/1)", "mean"),
                actionability_rate=("actionability (0/1)", "mean"),
            )
            .reset_index()
        )

        participant_summary = (
            subset.groupby(["xai", "participant"], observed=True)
            .agg(
                cases=("plausibility", "size"),
                plausibility_mean=("plausibility", "mean"),
                plausibility_median=("plausibility", "median"),
                proximity_mean=("proximity", "mean"),
                sparsity_mean=("sparsity", "mean"),
                validity_rate=("valid counterfactual (0/1)", "mean"),
                strategy=("reasoning strategy", "first"),
            )
            .reset_index()
        )
        participant_condition_summary = (
            participant_summary.groupby("xai", observed=True)
            .agg(
                participants=("participant", "size"),
                participant_mean=("plausibility_mean", "mean"),
                participant_median=("plausibility_mean", "median"),
                participant_sd=("plausibility_mean", "std"),
                participant_min=("plausibility_mean", "min"),
                participant_max=("plausibility_mean", "max"),
            )
            .reset_index()
        )

        participant_tests: dict[str, object] = {}
        none_participant_means = participant_summary.loc[
            participant_summary["xai"] == "none", "plausibility_mean"
        ]
        attribution_participant_means = participant_summary.loc[
            participant_summary["xai"] == "attribution", "plausibility_mean"
        ]
        if len(none_participant_means) > 1 and len(attribution_participant_means) > 1:
            welch = stats.ttest_ind(
                attribution_participant_means,
                none_participant_means,
                equal_var=False,
            )
            mann_whitney = stats.mannwhitneyu(
                attribution_participant_means,
                none_participant_means,
                alternative="two-sided",
            )
            pooled_sd = np.sqrt(
                (
                    attribution_participant_means.var(ddof=1)
                    + none_participant_means.var(ddof=1)
                )
                / 2
            )
            participant_tests = {
                "attribution_minus_none_mean": float(
                    attribution_participant_means.mean()
                    - none_participant_means.mean()
                ),
                "welch_t": float(welch.statistic),
                "welch_p": float(welch.pvalue),
                "mann_whitney_u": float(mann_whitney.statistic),
                "mann_whitney_p": float(mann_whitney.pvalue),
                "cohens_d": float(
                    (
                        attribution_participant_means.mean()
                        - none_participant_means.mean()
                    )
                    / pooled_sd
                )
                if pooled_sd > 0
                else None,
            }

        attribution_ranked = participant_summary.loc[
            participant_summary["xai"] == "attribution"
        ].sort_values("plausibility_mean", ascending=False)
        sensitivity = []
        none_row_mean = float(
            subset.loc[subset["xai"] == "none", "plausibility"].mean()
        )
        for removed_count in range(0, min(4, len(attribution_ranked))):
            removed = attribution_ranked.head(removed_count)["participant"].tolist()
            retained_rows = subset.loc[
                (subset["xai"] == "attribution")
                & (~subset["participant"].isin(removed))
            ]
            sensitivity.append(
                {
                    "worst_attribution_participants_removed": removed_count,
                    "removed_participants": removed,
                    "attribution_row_mean": float(retained_rows["plausibility"].mean()),
                    "difference_from_none_row_mean": float(
                        retained_rows["plausibility"].mean() - none_row_mean
                    ),
                }
            )

        cases_by_condition = {
            condition: sorted(
                subset.loc[subset["xai"] == condition, "instance id"]
                .astype(str)
                .unique()
                .tolist()
            )
            for condition in sorted(subset["xai"].unique())
        }
        case_pivot = subset.pivot_table(
            index="instance id", columns="xai", values="plausibility", aggfunc="mean"
        )
        complete_case_pivot = case_pivot.dropna(
            subset=[name for name in ("none", "attribution", "counterfactual") if name in case_pivot]
        )
        case_balanced = {
            "complete_instances": int(len(complete_case_pivot)),
            "condition_means": {
                str(name): float(value)
                for name, value in complete_case_pivot.mean().items()
            },
        }
        if {"attribution", "none"}.issubset(complete_case_pivot.columns):
            differences = complete_case_pivot["attribution"] - complete_case_pivot["none"]
            case_balanced["attribution_minus_none_mean"] = float(differences.mean())
            case_balanced["instances_attribution_worse"] = int((differences > 0).sum())
        if {"attribution", "counterfactual"}.issubset(complete_case_pivot.columns):
            differences = (
                complete_case_pivot["attribution"]
                - complete_case_pivot["counterfactual"]
            )
            case_balanced["attribution_minus_counterfactual_mean"] = float(
                differences.mean()
            )
            case_balanced["instances_attribution_worse_than_counterfactual"] = int(
                (differences > 0).sum()
            )

        feature_rows = []
        for condition in sorted(subset["xai"].unique()):
            condition_data = subset.loc[subset["xai"] == condition]
            for feature_index, feature_name in enumerate(FEATURE_LABELS[domain], start=1):
                changed = condition_data[f"x_{feature_index}_changed"] == 1
                feature_rows.append(
                    {
                        "xai": condition,
                        "feature": feature_name,
                        "changed_rate": float(changed.mean()),
                        "mean_abs_change": float(
                            condition_data[f"abs_change_x{feature_index}"].mean()
                        ),
                        "mean_abs_change_when_changed": float(
                            condition_data.loc[
                                changed, f"abs_change_x{feature_index}"
                            ].mean()
                        )
                        if changed.any()
                        else None,
                        "mean_nearest_neighbor_gap": float(
                            condition_data[f"nn_gap_x{feature_index}"].mean()
                        ),
                        "mean_original_nearest_neighbor_gap": float(
                            condition_data[
                                f"original_nn_gap_x{feature_index}"
                            ].mean()
                        ),
                        "mean_nearest_neighbor_gap_delta": float(
                            condition_data[f"nn_gap_delta_x{feature_index}"].mean()
                        ),
                        "mean_final_normalized_value": float(
                            condition_data[f"final_x{feature_index}"].mean()
                        ),
                    }
                )

        display_response_rows = []
        for condition in sorted(subset["xai"].unique()):
            condition_data = subset.loc[subset["xai"] == condition]
            for feature_index, feature_name in enumerate(
                FEATURE_LABELS[domain], start=1
            ):
                shown = condition_data[f"shown_x{feature_index}"] == 1
                for shown_value, label in ((True, "shown"), (False, "not_shown")):
                    selected = condition_data.loc[shown == shown_value]
                    display_response_rows.append(
                        {
                            "xai": condition,
                            "feature": feature_name,
                            "display_status": label,
                            "rows": int(len(selected)),
                            "changed_rate": float(
                                selected[f"x_{feature_index}_changed"].mean()
                            )
                            if len(selected)
                            else None,
                            "mean_abs_change": float(
                                selected[f"abs_change_x{feature_index}"].mean()
                            )
                            if len(selected)
                            else None,
                            "plausibility_mean": float(
                                selected["plausibility"].mean()
                            )
                            if len(selected)
                            else None,
                        }
                    )

        subset["shown_features_changed"] = sum(
            subset[f"shown_x{index}"] * subset[f"x_{index}_changed"]
            for index in range(1, 6)
        )
        subset["unshown_features_changed"] = sum(
            (1 - subset[f"shown_x{index}"]) * subset[f"x_{index}_changed"]
            for index in range(1, 6)
        )
        display_count_summary = (
            subset.groupby(
                ["xai", "shown_features_changed", "unshown_features_changed"],
                observed=True,
            )
            .agg(
                rows=("plausibility", "size"),
                participants=("participant", "nunique"),
                plausibility_mean=("plausibility", "mean"),
                plausibility_delta_mean=("plausibility_delta_from_original", "mean"),
                validity_rate=("valid counterfactual (0/1)", "mean"),
            )
            .reset_index()
        )

        target_summary = (
            subset.groupby(["xai", "target label"], observed=True)
            .agg(
                rows=("plausibility", "size"),
                participants=("participant", "nunique"),
                plausibility_mean=("plausibility", "mean"),
                plausibility_median=("plausibility", "median"),
                proximity_mean=("proximity", "mean"),
                sparsity_mean=("sparsity", "mean"),
                validity_rate=("valid counterfactual (0/1)", "mean"),
            )
            .reset_index()
        )

        target_feature_rows = []
        for (condition, target_label), condition_data in subset.groupby(
            ["xai", "target label"], observed=True
        ):
            for feature_index, feature_name in enumerate(FEATURE_LABELS[domain], start=1):
                changed = condition_data[f"x_{feature_index}_changed"] == 1
                target_feature_rows.append(
                    {
                        "xai": condition,
                        "target_label": target_label,
                        "feature": feature_name,
                        "changed_rate": float(changed.mean()),
                        "mean_abs_change": float(
                            condition_data[f"abs_change_x{feature_index}"].mean()
                        ),
                        "mean_nearest_neighbor_gap": float(
                            condition_data[f"nn_gap_x{feature_index}"].mean()
                        ),
                    }
                )

        pattern_summary = (
            subset.groupby(["xai", "change_pattern"], observed=True)
            .agg(
                rows=("plausibility", "size"),
                participants=("participant", "nunique"),
                plausibility_mean=("plausibility", "mean"),
                proximity_mean=("proximity", "mean"),
                validity_rate=("valid counterfactual (0/1)", "mean"),
            )
            .reset_index()
            .sort_values(["xai", "rows"], ascending=[True, False])
        )
        top_patterns = (
            pattern_summary.groupby("xai", group_keys=False, observed=True)
            .head(8)
            .reset_index(drop=True)
        )

        strategy_summary = (
            participant_summary.groupby(["xai", "strategy"], dropna=False, observed=True)
            .agg(
                participants=("participant", "size"),
                plausibility_mean=("plausibility_mean", "mean"),
                proximity_mean=("proximity_mean", "mean"),
                sparsity_mean=("sparsity_mean", "mean"),
                validity_rate=("validity_rate", "mean"),
            )
            .reset_index()
            .sort_values(["xai", "participants"], ascending=[True, False])
        )

        worst_attribution_participants = (
            participant_summary.loc[participant_summary["xai"] == "attribution"]
            .sort_values("plausibility_mean", ascending=False)
            .head(10)
        )

        attribution_participant_features = []
        attribution_rows = subset.loc[subset["xai"] == "attribution"]
        for participant, participant_data in attribution_rows.groupby(
            "participant", observed=True
        ):
            top_pattern = (
                participant_data["change_pattern"].value_counts().index[0]
                if len(participant_data)
                else None
            )
            participant_record: dict[str, object] = {
                "participant": participant,
                "cases": int(len(participant_data)),
                "plausibility_mean": float(participant_data["plausibility"].mean()),
                "top_change_pattern": top_pattern,
            }
            for feature_index, feature_name in enumerate(
                FEATURE_LABELS[domain], start=1
            ):
                participant_record[f"{feature_name}_changed_rate"] = float(
                    participant_data[f"x_{feature_index}_changed"].mean()
                )
                participant_record[f"{feature_name}_mean_abs_change"] = float(
                    participant_data[f"abs_change_x{feature_index}"].mean()
                )
                participant_record[f"{feature_name}_mean_nn_gap"] = float(
                    participant_data[f"nn_gap_x{feature_index}"].mean()
                )
            attribution_participant_features.append(participant_record)
        attribution_participant_features.sort(
            key=lambda item: item["plausibility_mean"], reverse=True
        )

        formulas = [
            'plausibility ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))',
            'plausibility ~ C(xai, Treatment(reference="none")) + C(Q("instance id")) + proximity + sparsity',
            'plausibility ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))'
            + " + "
            + " + ".join(f"abs_change_x{index}" for index in range(1, 6)),
        ]
        models = [fit_condition_model(subset, formula) for formula in formulas]

        robustness_models = [
            fit_condition_model(
                subset,
                f'{outcome} ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))',
            )
            for outcome in (
                "plausibility_delta_from_original",
                "plausibility_l2",
                "plausibility_l1_k5",
            )
        ]

        reference_matrix = reference_matrices[domain]
        reference_tree = cKDTree(reference_matrix)
        leave_one_out_distances = reference_tree.query(
            reference_matrix, k=2, p=1
        )[0][:, 1]
        correlation_matrix = np.corrcoef(reference_matrix, rowvar=False)
        reference_correlations = []
        for left_index, left_name in enumerate(FEATURE_LABELS[domain]):
            for right_index in range(left_index + 1, 5):
                reference_correlations.append(
                    {
                        "left": left_name,
                        "right": FEATURE_LABELS[domain][right_index],
                        "pearson_r": float(
                            correlation_matrix[left_index, right_index]
                        ),
                    }
                )
        reference_correlations.sort(
            key=lambda item: abs(item["pearson_r"]), reverse=True
        )

        report["domains"][domain] = {
            "condition_summary": records(condition_summary),
            "participant_condition_summary": records(participant_condition_summary),
            "participant_tests": participant_tests,
            "outlier_sensitivity": sensitivity,
            "cases_by_condition": cases_by_condition,
            "case_balanced": case_balanced,
            "models": models,
            "robustness_models": robustness_models,
            "reference_structure": {
                "training_rows": int(len(reference_matrix)),
                "leave_one_out_l1_mean": float(leave_one_out_distances.mean()),
                "leave_one_out_l1_median": float(np.median(leave_one_out_distances)),
                "feature_correlations": reference_correlations,
            },
            "feature_summary": feature_rows,
            "display_response_summary": display_response_rows,
            "display_count_summary": records(display_count_summary),
            "target_summary": records(target_summary),
            "target_feature_summary": target_feature_rows,
            "top_change_patterns": records(top_patterns),
            "strategy_summary": records(strategy_summary),
            "worst_attribution_participants": records(
                worst_attribution_participants[
                    [
                        "participant",
                        "cases",
                        "plausibility_mean",
                        "plausibility_median",
                        "proximity_mean",
                        "sparsity_mean",
                        "validity_rate",
                        "strategy",
                    ]
                ]
            ),
            "attribution_participant_features": attribution_participant_features,
        }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
