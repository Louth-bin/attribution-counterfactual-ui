"""Diagnose domain-specific XAI differences in v0.9 boundary distance."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import calculate_v09_boundary_distance as boundary_calc


OUTPUT_PATH = Path(tempfile.gettempdir()) / "v09_boundary_condition_analysis.json"
BOUNDARY_VALUES = Path(tempfile.gettempdir()) / "boundary_distance_v09_values.csv"


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(8)
    return result.replace({np.nan: None}).to_dict(orient="records")


def model_summary(data: pd.DataFrame, outcome: str) -> dict[str, object]:
    formula = (
        f'{outcome} ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))'
    )
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["participant"]}
    )
    result: dict[str, object] = {
        "formula": formula,
        "rows": int(model.nobs),
        "participants": int(data["participant"].nunique()),
        "r_squared": float(model.rsquared),
        "terms": {},
    }
    for condition in ("attribution", "counterfactual"):
        term = f'C(xai, Treatment(reference="none"))[T.{condition}]'
        result["terms"][condition] = {
            "coefficient_vs_none": float(model.params.get(term, np.nan)),
            "standard_error": float(model.bse.get(term, np.nan)),
            "p_value": float(model.pvalues.get(term, np.nan)),
        }
    contrast = np.zeros(len(model.params))
    names = list(model.params.index)
    attr_term = 'C(xai, Treatment(reference="none"))[T.attribution]'
    cf_term = 'C(xai, Treatment(reference="none"))[T.counterfactual]'
    contrast[names.index(cf_term)] = 1.0
    contrast[names.index(attr_term)] = -1.0
    test = model.t_test(contrast)
    result["counterfactual_vs_attribution"] = {
        "coefficient": float(test.effect.item()),
        "standard_error": float(test.sd.item()),
        "p_value": float(test.pvalue),
    }
    return result


def main() -> None:
    base = pd.read_csv(boundary_calc.DEFAULT_INPUT)
    boundary = pd.read_csv(BOUNDARY_VALUES)
    base.insert(0, "row_number", np.arange(1, len(base) + 1))
    data = base.merge(
        boundary,
        on=["row_number", "participant", "domain", "phase"],
        how="left",
        validate="one_to_one",
    )
    data = data.loc[data["phase"] == "testing"].copy()
    rename = {
        boundary_calc.ORIGINAL_COLUMN: "boundary_original",
        boundary_calc.EDITED_COLUMN: "boundary_new",
        boundary_calc.DELTA_COLUMN: "boundary_delta",
        "valid counterfactual (0/1)": "valid",
        "distance to closest minimal counterfactual": "distance_to_minimal",
        "confidence for target label original": "confidence_original",
        "confidence for target label counterfactual": "confidence_new",
        "delta confidence of target label": "confidence_delta",
    }
    data = data.rename(columns=rename)
    numeric_columns = [
        "boundary_original",
        "boundary_new",
        "boundary_delta",
        "valid",
        "distance_to_minimal",
        "confidence_original",
        "confidence_new",
        "confidence_delta",
        "proximity",
        "sparsity",
        *[f"x_{index}_change" for index in range(1, 6)],
        *[f"x_{index}_changed" for index in range(1, 6)],
    ]
    for column in numeric_columns:
        data[column] = clean_number(data[column])
    data["absolute_target_margin"] = (data["confidence_new"] - 0.5).abs()

    experiment = json.loads(
        boundary_calc.DEFAULT_EXPERIMENT_DATA.read_text(encoding="utf-8")
    )
    displayed_solvers = {
        domain: boundary_calc.WachterBoundarySearch(
            dataset["browser_model"], dataset["test_pool"][0]
        )
        for domain, dataset in experiment["datasets"].items()
    }
    training_explanation_summaries: dict[str, dict[str, object]] = {}
    for domain, dataset in experiment["datasets"].items():
        training_records = []
        feature_counts: Counter[str] = Counter()
        for case in dataset["training_pool"]:
            counterfactual = case["counterfactual"]
            target_probability = float(counterfactual["target_probability"])
            displayed_profile = np.asarray(
                [
                    boundary_calc.normalized_value(value, feature_type, feature_range)
                    for value, feature_type, feature_range in zip(
                        counterfactual["feature_values"],
                        case["feature_types"],
                        case["raw_feature_ranges"],
                    )
                ]
            )
            displayed_boundary_distance = displayed_solvers[domain].solve(
                displayed_profile
            ).distance
            feature_counts.update(counterfactual["selected_feature_names"])
            training_records.append({
                "displayed_cf_target_probability": target_probability,
                "displayed_cf_probability_margin": abs(target_probability - 0.5),
                "displayed_cf_proximity": float(
                    counterfactual["optimization"]["objective_value"]
                ),
                "displayed_cf_boundary_distance": float(
                    displayed_boundary_distance
                ),
                "displayed_cf_features": "+".join(
                    counterfactual["selected_feature_names"]
                ),
            })
        training_frame = pd.DataFrame(training_records)
        training_explanation_summaries[domain] = {
            "cases": int(len(training_frame)),
            "feature_counts": dict(feature_counts),
            "target_probability_mean": float(
                training_frame["displayed_cf_target_probability"].mean()
            ),
            "probability_margin_mean": float(
                training_frame["displayed_cf_probability_margin"].mean()
            ),
            "probability_margin_median": float(
                training_frame["displayed_cf_probability_margin"].median()
            ),
            "displayed_proximity_mean": float(
                training_frame["displayed_cf_proximity"].mean()
            ),
            "displayed_proximity_median": float(
                training_frame["displayed_cf_proximity"].median()
            ),
            "displayed_boundary_distance_mean": float(
                training_frame["displayed_cf_boundary_distance"].mean()
            ),
            "displayed_boundary_distance_median": float(
                training_frame["displayed_cf_boundary_distance"].median()
            ),
            "feature_pairs": training_frame["displayed_cf_features"].tolist(),
        }

    report: dict[str, object] = {"rows": int(len(data)), "domains": {}}
    for domain in ("housing", "diabetes", "safelimit"):
        subset = data.loc[data["domain"] == domain].copy()
        if subset.empty:
            continue
        condition = (
            subset.groupby("xai", observed=True)
            .agg(
                rows=("boundary_new", "size"),
                participants=("participant", "nunique"),
                boundary_original_mean=("boundary_original", "mean"),
                boundary_new_mean=("boundary_new", "mean"),
                boundary_new_median=("boundary_new", "median"),
                boundary_new_sd=("boundary_new", "std"),
                boundary_delta_mean=("boundary_delta", "mean"),
                boundary_delta_median=("boundary_delta", "median"),
                validity_rate=("valid", "mean"),
                proximity_mean=("proximity", "mean"),
                attributes_changed_mean=("sparsity", "mean"),
                confidence_margin_mean=("absolute_target_margin", "mean"),
                distance_to_minimal_mean=("distance_to_minimal", "mean"),
            )
            .reset_index()
        )

        valid_condition = (
            subset.groupby(["xai", "valid"], observed=True)
            .agg(
                rows=("boundary_new", "size"),
                participants=("participant", "nunique"),
                boundary_new_mean=("boundary_new", "mean"),
                boundary_new_median=("boundary_new", "median"),
                boundary_delta_mean=("boundary_delta", "mean"),
                proximity_mean=("proximity", "mean"),
                attributes_changed_mean=("sparsity", "mean"),
                confidence_margin_mean=("absolute_target_margin", "mean"),
            )
            .reset_index()
        )
        target_condition = (
            subset.groupby(["xai", "target label"], observed=True)
            .agg(
                rows=("boundary_new", "size"),
                boundary_new_mean=("boundary_new", "mean"),
                boundary_delta_mean=("boundary_delta", "mean"),
                validity_rate=("valid", "mean"),
                proximity_mean=("proximity", "mean"),
            )
            .reset_index()
        )

        case_pivot = subset.pivot_table(
            index="instance id", columns="xai", values="boundary_new", aggfunc="mean"
        ).dropna(subset=["attribution", "counterfactual", "none"])
        case_balanced = {
            "complete_cases": int(len(case_pivot)),
            "means": {name: float(value) for name, value in case_pivot.mean().items()},
            "counterfactual_minus_none": float(
                (case_pivot["counterfactual"] - case_pivot["none"]).mean()
            ),
            "counterfactual_closer_than_none_cases": int(
                (case_pivot["counterfactual"] < case_pivot["none"]).sum()
            ),
            "counterfactual_minus_attribution": float(
                (case_pivot["counterfactual"] - case_pivot["attribution"]).mean()
            ),
            "counterfactual_closer_than_attribution_cases": int(
                (case_pivot["counterfactual"] < case_pivot["attribution"]).sum()
            ),
        }

        participant = (
            subset.groupby(["xai", "participant"], observed=True)
            .agg(
                rows=("boundary_new", "size"),
                boundary_new_mean=("boundary_new", "mean"),
                boundary_delta_mean=("boundary_delta", "mean"),
                validity_rate=("valid", "mean"),
                proximity_mean=("proximity", "mean"),
                attributes_changed_mean=("sparsity", "mean"),
            )
            .reset_index()
        )
        participant_condition = (
            participant.groupby("xai", observed=True)
            .agg(
                participants=("participant", "size"),
                participant_mean=("boundary_new_mean", "mean"),
                participant_median=("boundary_new_mean", "median"),
                participant_sd=("boundary_new_mean", "std"),
                participant_min=("boundary_new_mean", "min"),
                participant_max=("boundary_new_mean", "max"),
            )
            .reset_index()
        )

        cf_participant = participant.loc[participant["xai"] == "counterfactual"]
        none_participant = participant.loc[participant["xai"] == "none"]
        participant_test: dict[str, float] = {}
        if len(cf_participant) > 1 and len(none_participant) > 1:
            welch = stats.ttest_ind(
                cf_participant["boundary_new_mean"],
                none_participant["boundary_new_mean"],
                equal_var=False,
            )
            mann = stats.mannwhitneyu(
                cf_participant["boundary_new_mean"],
                none_participant["boundary_new_mean"],
                alternative="two-sided",
            )
            participant_test = {
                "welch_t": float(welch.statistic),
                "welch_p": float(welch.pvalue),
                "mann_whitney_u": float(mann.statistic),
                "mann_whitney_p": float(mann.pvalue),
            }

        feature_rows = []
        for condition_name, condition_rows in subset.groupby("xai", observed=True):
            for index in range(1, 6):
                changed = condition_rows[f"x_{index}_changed"] == 1
                feature_rows.append(
                    {
                        "xai": condition_name,
                        "feature_index": index,
                        "changed_rate": float(changed.mean()),
                        "absolute_change_mean": float(
                            condition_rows[f"x_{index}_change"].abs().mean()
                        ),
                        "boundary_new_when_changed": float(
                            condition_rows.loc[changed, "boundary_new"].mean()
                        )
                        if changed.any()
                        else None,
                    }
                )

        cf_rows = subset.loc[subset["xai"] == "counterfactual"].copy()
        cf_correlations = {}
        for predictor in (
            "proximity",
            "sparsity",
            "distance_to_minimal",
        ):
            correlation = stats.spearmanr(
                cf_rows["boundary_new"], cf_rows[predictor], nan_policy="omit"
            )
            cf_correlations[predictor] = {
                "spearman_r": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            }

        models = {
            "all_boundary_new": model_summary(subset, "boundary_new"),
            "all_boundary_delta": model_summary(subset, "boundary_delta"),
        }
        valid_rows = subset.loc[subset["valid"] == 1]
        if (
            valid_rows["participant"].nunique() > 2
            and set(valid_rows["xai"].unique()) == {"attribution", "counterfactual", "none"}
        ):
            models["valid_only_boundary_new"] = model_summary(
                valid_rows, "boundary_new"
            )

        report["domains"][domain] = {
            "condition_summary": records(condition),
            "validity_summary": records(valid_condition),
            "target_summary": records(target_condition),
            "case_balanced": case_balanced,
            "participant_summary": records(participant_condition),
            "participant_test_counterfactual_vs_none": participant_test,
            "models": models,
            "feature_summary": feature_rows,
            "counterfactual_condition_correlations": cf_correlations,
            "displayed_training_counterfactual_summary": (
                training_explanation_summaries[domain]
            ),
            "counterfactual_participants": records(
                cf_participant.sort_values("boundary_new_mean")
            ),
        }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
