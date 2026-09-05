"""Analyze whether v2.0 participants use stable, target-specific endpoints.

The unit of analysis is a participant's changed feature on a testing trial. A
leave-one-out endpoint model predicts the final normalized value from the
participant's other edits to the same feature and target label. A competing
fixed-change model predicts the final value by adding the participant's median
change amount to the current original value. This distinguishes genuine
endpoint/threshold consistency from merely repeating a delta.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "outputs"
    / "v20-normalized-original-final-values"
    / "qualtrics_results_v2.0.roundtrip.csv"
)
OUT = ROOT / "outputs" / "v20-threshold-analysis"

FEATURES = (
    (1, "Glucose"),
    (2, "Blood Pressure"),
    (3, "Insulin"),
    (4, "BMI"),
    (5, "Age"),
)
CONDITIONS = ("none", "attribution", "counterfactual")
TARGETS = ("Diabetes", "No Diabetes")
RNG = np.random.default_rng(20260902)


def original_col(index: int, feature: str) -> str:
    return f"x_{index} {feature} original normalized value"


def final_col(index: int, feature: str) -> str:
    return f"x_{index} {feature} final changed normalized value"


def bh(values: pd.Series) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = values.notna().to_numpy()
    if valid.any():
        result[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return result


def rank_omnibus_permutation(groups: list[np.ndarray], permutations: int = 5000) -> float:
    groups = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if len(groups) < 2:
        return math.nan
    values = np.concatenate(groups)
    labels = np.concatenate(
        [np.full(len(group), index, dtype=int) for index, group in enumerate(groups)]
    )
    observed = stats.kruskal(*groups).statistic
    exceed = 0
    for _ in range(permutations):
        shuffled = RNG.permutation(labels)
        simulated = [values[shuffled == index] for index in range(len(groups))]
        statistic = stats.kruskal(*simulated).statistic
        exceed += statistic >= observed - 1e-12
    return (exceed + 1) / (permutations + 1)


def prepare_long(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    testing = data.loc[data["phase"].eq("testing")].copy()
    for _, row in testing.iterrows():
        for index, feature in FEATURES:
            final = row[final_col(index, feature)]
            if pd.isna(final):
                continue
            original = float(row[original_col(index, feature)])
            record = {
                "participant": row["participant"],
                "xai": row["xai"],
                "trial number": int(row["trial number"]),
                "instance id": int(row["instance id"]),
                "target label": row["target label"],
                "feature index": index,
                "feature": feature,
                "original": original,
                "final": float(final),
                "delta": float(final) - original,
            }
            for optional in (
                "successful CF",
                "successful counterfactual (continuous)",
                "cognitive model v0.1 family",
                "global relevance reliance (α)",
            ):
                if optional in data.columns:
                    record[optional] = row[optional]
            rows.append(record)
    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No testing endpoints were found")
    return long


def add_loo_predictions(long: pd.DataFrame) -> pd.DataFrame:
    result = long.copy()
    for column in (
        "threshold prediction",
        "fixed-delta prediction",
        "target-agnostic threshold prediction",
    ):
        result[column] = np.nan

    for _, indexes in result.groupby(
        ["participant", "target label", "feature"], sort=False
    ).groups.items():
        indexes = list(indexes)
        if len(indexes) < 2:
            continue
        for index in indexes:
            others = [other for other in indexes if other != index]
            result.at[index, "threshold prediction"] = result.loc[others, "final"].median()
            result.at[index, "fixed-delta prediction"] = (
                result.at[index, "original"] + result.loc[others, "delta"].median()
            )

    for _, indexes in result.groupby(["participant", "feature"], sort=False).groups.items():
        indexes = list(indexes)
        if len(indexes) < 2:
            continue
        for index in indexes:
            others = [other for other in indexes if other != index]
            result.at[index, "target-agnostic threshold prediction"] = result.loc[
                others, "final"
            ].median()

    result["threshold absolute error"] = (
        result["final"] - result["threshold prediction"]
    ).abs()
    result["fixed-delta absolute error"] = (
        result["final"] - result["fixed-delta prediction"]
    ).abs()
    result["target-agnostic absolute error"] = (
        result["final"] - result["target-agnostic threshold prediction"]
    ).abs()
    return result


def within_cell_slope(group: pd.DataFrame) -> float:
    usable: list[pd.DataFrame] = []
    for _, cell in group.groupby(["target label", "feature"]):
        if len(cell) >= 2:
            centered = cell[["original", "final"]] - cell[["original", "final"]].mean()
            usable.append(centered)
    if not usable:
        return math.nan
    centered = pd.concat(usable, ignore_index=True)
    denominator = float(np.square(centered["original"]).sum())
    if denominator <= 1e-12:
        return math.nan
    return float((centered["original"] * centered["final"]).sum() / denominator)


def exact_repeat_rate(group: pd.DataFrame) -> float:
    flags: list[bool] = []
    for _, cell in group.groupby(["target label", "feature"]):
        values = cell["final"].to_numpy(float)
        if len(values) < 2:
            continue
        for index, value in enumerate(values):
            others = np.delete(values, index)
            flags.append(bool(np.isclose(others, value, rtol=0.0, atol=1e-10).any()))
    return float(np.mean(flags)) if flags else math.nan


def summarize_participants(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_rows: list[dict] = []
    target_rows: list[dict] = []

    for participant, group in scored.groupby("participant"):
        evaluable = group.dropna(subset=["threshold absolute error"])
        row = {
            "participant": participant,
            "xai": group["xai"].iloc[0],
            "n edits": len(group),
            "n evaluable edits": len(evaluable),
            "evaluable coverage": len(evaluable) / len(group),
            "n repeated feature-target cells": int(
                (group.groupby(["feature", "target label"]).size() >= 2).sum()
            ),
            "threshold CV MAE": evaluable["threshold absolute error"].mean(),
            "fixed-delta CV MAE": evaluable["fixed-delta absolute error"].mean(),
            "target-agnostic CV MAE": evaluable[
                "target-agnostic absolute error"
            ].mean(),
            "within 0.025 of LOO threshold": (
                evaluable["threshold absolute error"] <= 0.025 + 1e-12
            ).mean(),
            "within 0.05 of LOO threshold": (
                evaluable["threshold absolute error"] <= 0.05 + 1e-12
            ).mean(),
            "exact endpoint repeat rate": exact_repeat_rate(group),
            "within-cell endpoint-on-original slope": within_cell_slope(group),
        }
        row["threshold advantage over fixed delta"] = (
            row["fixed-delta CV MAE"] - row["threshold CV MAE"]
        )
        row["threshold relative gain over fixed delta"] = (
            row["threshold advantage over fixed delta"] / row["fixed-delta CV MAE"]
            if row["fixed-delta CV MAE"] > 1e-12
            else math.nan
        )
        row["target-specific advantage"] = (
            row["target-agnostic CV MAE"] - row["threshold CV MAE"]
        )

        if "cognitive model v0.1 family" in group:
            row["cognitive model v0.1 family"] = group[
                "cognitive model v0.1 family"
            ].iloc[0]
        if "global relevance reliance (α)" in group:
            row["global relevance reliance (α)"] = group[
                "global relevance reliance (α)"
            ].iloc[0]
        participant_rows.append(row)

        for target, target_group in group.groupby("target label"):
            target_eval = target_group.dropna(subset=["threshold absolute error"])
            target_row = {
                "participant": participant,
                "xai": group["xai"].iloc[0],
                "target label": target,
                "n edits": len(target_group),
                "n evaluable edits": len(target_eval),
                "evaluable coverage": (
                    len(target_eval) / len(target_group) if len(target_group) else math.nan
                ),
                "threshold CV MAE": target_eval["threshold absolute error"].mean(),
                "fixed-delta CV MAE": target_eval[
                    "fixed-delta absolute error"
                ].mean(),
                "within 0.05 of LOO threshold": (
                    target_eval["threshold absolute error"] <= 0.05 + 1e-12
                ).mean(),
            }
            target_row["threshold advantage over fixed delta"] = (
                target_row["fixed-delta CV MAE"] - target_row["threshold CV MAE"]
            )
            target_row["threshold relative gain over fixed delta"] = (
                target_row["threshold advantage over fixed delta"]
                / target_row["fixed-delta CV MAE"]
                if target_row["fixed-delta CV MAE"] > 1e-12
                else math.nan
            )
            target_rows.append(target_row)

    participants = pd.DataFrame(participant_rows)
    by_target = pd.DataFrame(target_rows)

    participants["near-exact threshold"] = (
        participants["n evaluable edits"].ge(15)
        & participants["evaluable coverage"].ge(0.60)
        & participants["threshold CV MAE"].le(0.05)
        & participants["threshold relative gain over fixed delta"].ge(0.20)
        & participants["within 0.05 of LOO threshold"].ge(0.75)
    ).astype(int)
    participants["strong threshold-like"] = (
        participants["n evaluable edits"].ge(15)
        & participants["evaluable coverage"].ge(0.60)
        & participants["threshold CV MAE"].le(0.075)
        & participants["threshold relative gain over fixed delta"].ge(0.20)
        & participants["within-cell endpoint-on-original slope"].abs().le(0.25)
    ).astype(int)
    by_target["very consistent threshold for target"] = (
        by_target["n evaluable edits"].ge(6)
        & by_target["evaluable coverage"].ge(0.50)
        & by_target["threshold CV MAE"].le(0.05)
        & by_target["threshold relative gain over fixed delta"].ge(0.20)
        & by_target["within 0.05 of LOO threshold"].ge(0.75)
    ).astype(int)
    return participants, by_target


def cell_thresholds(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in long.groupby(
        ["participant", "xai", "feature", "target label"], sort=False
    ):
        participant, xai, feature, target = keys
        values = group["final"].to_numpy(float)
        if len(values) < 2:
            continue
        rows.append(
            {
                "participant": participant,
                "xai": xai,
                "feature": feature,
                "target label": target,
                "n edits": len(values),
                "estimated threshold": float(np.median(values)),
                "endpoint MAD": float(stats.median_abs_deviation(values)),
                "endpoint IQR": float(np.subtract(*np.percentile(values, [75, 25]))),
                "endpoint range": float(np.ptp(values)),
                "original range": float(np.ptp(group["original"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows)


def condition_summary(participants: pd.DataFrame) -> pd.DataFrame:
    return (
        participants.groupby("xai")
        .agg(
            participants=("participant", "nunique"),
            near_exact_n=("near-exact threshold", "sum"),
            near_exact_rate=("near-exact threshold", "mean"),
            strong_threshold_like_n=("strong threshold-like", "sum"),
            strong_threshold_like_rate=("strong threshold-like", "mean"),
            median_threshold_cv_mae=("threshold CV MAE", "median"),
            mean_threshold_cv_mae=("threshold CV MAE", "mean"),
            median_fixed_delta_cv_mae=("fixed-delta CV MAE", "median"),
            median_relative_gain=("threshold relative gain over fixed delta", "median"),
            median_within_005=("within 0.05 of LOO threshold", "median"),
            median_threshold_slope=("within-cell endpoint-on-original slope", "median"),
            median_target_specific_advantage=("target-specific advantage", "median"),
        )
        .reset_index()
    )


def condition_tests(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for flag in ("near-exact threshold", "strong threshold-like"):
        table = pd.crosstab(participants["xai"], participants[flag]).reindex(
            index=CONDITIONS, columns=[0, 1], fill_value=0
        )
        if (table.sum(axis=0) == 0).any():
            chi2, p_chi = 0.0, 1.0
        else:
            chi2, p_chi, _, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "metric": flag,
                "comparison": "omnibus 3x2 chi-square",
                "effect": chi2,
                "p": p_chi,
            }
        )
        for left, right in combinations(CONDITIONS, 2):
            pair = table.loc[[left, right], [0, 1]].to_numpy()
            odds, p = stats.fisher_exact(pair)
            rows.append(
                {
                    "metric": flag,
                    "comparison": f"{left} vs {right}",
                    "effect": odds,
                    "p": p,
                }
            )

    continuous = (
        "threshold CV MAE",
        "threshold relative gain over fixed delta",
        "within 0.05 of LOO threshold",
        "within-cell endpoint-on-original slope",
        "target-specific advantage",
    )
    for metric in continuous:
        groups = [
            participants.loc[participants["xai"].eq(condition), metric]
            .dropna()
            .to_numpy(float)
            for condition in CONDITIONS
        ]
        h, p = stats.kruskal(*groups)
        rows.append(
            {
                "metric": metric,
                "comparison": "omnibus Kruskal-Wallis",
                "effect": h,
                "p": p,
            }
        )
        for left, right in combinations(CONDITIONS, 2):
            x = participants.loc[participants["xai"].eq(left), metric].dropna()
            y = participants.loc[participants["xai"].eq(right), metric].dropna()
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{left} vs {right}",
                    "effect": float(x.median() - y.median()),
                    "p": p,
                }
            )
    tests = pd.DataFrame(rows)
    tests["q within metric"] = tests.groupby("metric")["p"].transform(
        lambda values: bh(values)
    )
    return tests


def model_comparison_tests(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = [("all", participants)] + [
        (condition, participants.loc[participants["xai"].eq(condition)])
        for condition in CONDITIONS
    ]
    for scope, group in scopes:
        comparisons = (
            (
                "target-specific endpoint vs fixed delta",
                "threshold CV MAE",
                "fixed-delta CV MAE",
            ),
            (
                "target-specific endpoint vs target-agnostic endpoint",
                "threshold CV MAE",
                "target-agnostic CV MAE",
            ),
        )
        for comparison, preferred, competitor in comparisons:
            valid = group.dropna(subset=[preferred, competitor])
            difference = valid[competitor] - valid[preferred]
            statistic, p = stats.wilcoxon(
                valid[preferred], valid[competitor], alternative="less"
            )
            rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "participants": len(valid),
                    "preferred model median MAE": valid[preferred].median(),
                    "competitor median MAE": valid[competitor].median(),
                    "median MAE improvement": difference.median(),
                    "participants preferred model wins": int((difference > 0).sum()),
                    "Wilcoxon statistic": statistic,
                    "one-sided p": p,
                }
            )
    result = pd.DataFrame(rows)
    result["q across model comparisons"] = bh(result["one-sided p"])
    return result


def target_consistency_analysis(
    by_target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        by_target.groupby(["target label", "xai"])
        .agg(
            participants=("participant", "nunique"),
            median_threshold_cv_mae=("threshold CV MAE", "median"),
            mean_threshold_cv_mae=("threshold CV MAE", "mean"),
            median_fixed_delta_cv_mae=("fixed-delta CV MAE", "median"),
            median_relative_gain=("threshold relative gain over fixed delta", "median"),
            median_within_005=("within 0.05 of LOO threshold", "median"),
        )
        .reset_index()
    )
    wide = by_target.pivot(
        index=["participant", "xai"], columns="target label", values="threshold CV MAE"
    ).dropna(subset=list(TARGETS))
    wide["Diabetes minus No Diabetes threshold MAE"] = (
        wide["Diabetes"] - wide["No Diabetes"]
    )
    wide = wide.reset_index()
    statistic, p = stats.wilcoxon(
        wide["Diabetes"], wide["No Diabetes"], alternative="two-sided"
    )
    rows = [
        {
            "scope": "all",
            "comparison": "Diabetes vs No Diabetes threshold CV MAE",
            "participants": len(wide),
            "median Diabetes MAE": wide["Diabetes"].median(),
            "median No Diabetes MAE": wide["No Diabetes"].median(),
            "median paired difference": wide[
                "Diabetes minus No Diabetes threshold MAE"
            ].median(),
            "test": "paired Wilcoxon",
            "statistic": statistic,
            "p": p,
        }
    ]
    condition_groups = [
        wide.loc[
            wide["xai"].eq(condition), "Diabetes minus No Diabetes threshold MAE"
        ].to_numpy(float)
        for condition in CONDITIONS
    ]
    statistic, p = stats.kruskal(*condition_groups)
    rows.append(
        {
            "scope": "condition interaction",
            "comparison": "condition difference in target-label MAE contrast",
            "participants": len(wide),
            "median Diabetes MAE": math.nan,
            "median No Diabetes MAE": math.nan,
            "median paired difference": math.nan,
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p": p,
        }
    )
    for condition in CONDITIONS:
        group = wide.loc[wide["xai"].eq(condition)]
        statistic, p = stats.wilcoxon(
            group["Diabetes"], group["No Diabetes"], alternative="two-sided"
        )
        rows.append(
            {
                "scope": condition,
                "comparison": "Diabetes vs No Diabetes threshold CV MAE",
                "participants": len(group),
                "median Diabetes MAE": group["Diabetes"].median(),
                "median No Diabetes MAE": group["No Diabetes"].median(),
                "median paired difference": group[
                    "Diabetes minus No Diabetes threshold MAE"
                ].median(),
                "test": "paired Wilcoxon",
                "statistic": statistic,
                "p": p,
            }
        )
    tests = pd.DataFrame(rows)
    tests["q"] = bh(tests["p"])
    return summary, tests


def sensitivity_table(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mae_cutoff in (0.025, 0.05, 0.075, 0.10):
        for relative_gain in (0.0, 0.10, 0.20):
            flag = (
                participants["n evaluable edits"].ge(15)
                & participants["evaluable coverage"].ge(0.60)
                & participants["threshold CV MAE"].le(mae_cutoff)
                & participants["threshold relative gain over fixed delta"].ge(relative_gain)
                & participants["within-cell endpoint-on-original slope"].abs().le(0.25)
            )
            table = pd.crosstab(participants["xai"], flag).reindex(
                index=CONDITIONS, columns=[False, True], fill_value=0
            )
            if (table.sum(axis=0) == 0).any():
                p = 1.0
            else:
                _, p, _, _ = stats.chi2_contingency(table)
            row = {
                "threshold CV MAE cutoff": mae_cutoff,
                "minimum relative gain over fixed delta": relative_gain,
                "omnibus chi-square p": p,
            }
            for condition in CONDITIONS:
                count = int(table.loc[condition, True])
                total = int(table.loc[condition].sum())
                row[f"{condition} n"] = count
                row[f"{condition} total"] = total
                row[f"{condition} rate"] = count / total
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_level_analysis(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    tests: list[dict] = []
    for (feature, target), group in cells.groupby(["feature", "target label"]):
        for condition in CONDITIONS:
            values = group.loc[group["xai"].eq(condition), "estimated threshold"]
            summaries.append(
                {
                    "feature": feature,
                    "target label": target,
                    "xai": condition,
                    "participants": values.size,
                    "median estimated threshold": values.median(),
                    "mean estimated threshold": values.mean(),
                    "SD across participants": values.std(ddof=1),
                    "median within-participant MAD": group.loc[
                        group["xai"].eq(condition), "endpoint MAD"
                    ].median(),
                }
            )
        arrays = [
            group.loc[group["xai"].eq(condition), "estimated threshold"].to_numpy(float)
            for condition in CONDITIONS
        ]
        if all(len(array) >= 2 for array in arrays):
            h, p = stats.kruskal(*arrays)
            permutation_p = rank_omnibus_permutation(arrays)
        else:
            h = p = permutation_p = math.nan
        tests.append(
            {
                "feature": feature,
                "target label": target,
                "comparison": "omnibus",
                "effect": h,
                "p": p,
                "permutation p": permutation_p,
            }
        )
        for left, right in combinations(CONDITIONS, 2):
            x = group.loc[group["xai"].eq(left), "estimated threshold"]
            y = group.loc[group["xai"].eq(right), "estimated threshold"]
            if len(x) >= 2 and len(y) >= 2:
                _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                effect = float(x.median() - y.median())
            else:
                p = effect = math.nan
            tests.append(
                {
                    "feature": feature,
                    "target label": target,
                    "comparison": f"{left} vs {right}",
                    "effect": effect,
                    "p": p,
                    "permutation p": math.nan,
                }
            )
    tests_frame = pd.DataFrame(tests)
    tests_frame["q across all threshold tests"] = bh(tests_frame["p"])
    tests_frame["q among omnibus tests"] = np.nan
    omnibus = tests_frame["comparison"].eq("omnibus")
    tests_frame.loc[omnibus, "q among omnibus tests"] = bh(
        tests_frame.loc[omnibus, "p"]
    )
    tests_frame["q among pairwise tests"] = np.nan
    pairwise = ~omnibus
    tests_frame.loc[pairwise, "q among pairwise tests"] = bh(
        tests_frame.loc[pairwise, "p"]
    )
    return pd.DataFrame(summaries), tests_frame


def target_separation(cells: pd.DataFrame) -> pd.DataFrame:
    pivot = cells.pivot_table(
        index=["participant", "xai", "feature"],
        columns="target label",
        values="estimated threshold",
        aggfunc="first",
    ).reset_index()
    if not set(TARGETS).issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=list(TARGETS)).copy()
    pivot["Diabetes minus No Diabetes threshold"] = (
        pivot["Diabetes"] - pivot["No Diabetes"]
    )
    pivot["expected target ordering"] = (
        pivot["Diabetes minus No Diabetes threshold"] > 0
    ).astype(int)
    return pivot


def target_separation_summary(separation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = (
        separation.groupby(["participant", "xai"])
        .agg(
            feature_target_pairs=("feature", "count"),
            median_target_separation=(
                "Diabetes minus No Diabetes threshold",
                "median",
            ),
            mean_target_separation=("Diabetes minus No Diabetes threshold", "mean"),
            expected_ordering_rate=("expected target ordering", "mean"),
        )
        .reset_index()
    )
    summary = (
        participant.groupby("xai")
        .agg(
            participants=("participant", "nunique"),
            median_target_separation=("median_target_separation", "median"),
            mean_target_separation=("median_target_separation", "mean"),
            median_expected_ordering_rate=("expected_ordering_rate", "median"),
        )
        .reset_index()
    )
    rows = []
    for metric in ("median_target_separation", "expected_ordering_rate"):
        arrays = [
            participant.loc[participant["xai"].eq(condition), metric].to_numpy(float)
            for condition in CONDITIONS
        ]
        statistic, p = stats.kruskal(*arrays)
        rows.append({"metric": metric, "statistic": statistic, "p": p})
    tests = pd.DataFrame(rows)
    tests["q"] = bh(tests["p"])
    return summary, tests


def training_alignment(data: pd.DataFrame, long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = data.loc[
        data["phase"].eq("training") & data["xai"].eq("counterfactual")
    ].copy()
    training["inferred target label"] = training["original label"].map(
        {"Diabetes": "No Diabetes", "No Diabetes": "Diabetes"}
    )
    references: list[dict] = []
    for _, row in training.iterrows():
        for index, feature in FEATURES:
            endpoint = row[final_col(index, feature)]
            if pd.notna(endpoint):
                references.append(
                    {
                        "instance id": int(row["instance id"]),
                        "feature": feature,
                        "target label": row["inferred target label"],
                        "training endpoint": float(endpoint),
                    }
                )
    reference = pd.DataFrame(references).drop_duplicates()
    lookup = {
        key: group["training endpoint"].to_numpy(float)
        for key, group in reference.groupby(["feature", "target label"])
    }

    aligned = long.copy()
    nearest = []
    distance_to_median = []
    for _, row in aligned.iterrows():
        endpoints = lookup.get((row["feature"], row["target label"]), np.array([]))
        if len(endpoints):
            nearest.append(float(np.min(np.abs(endpoints - row["final"]))))
            distance_to_median.append(float(abs(np.median(endpoints) - row["final"])))
        else:
            nearest.append(math.nan)
            distance_to_median.append(math.nan)
    aligned["distance to nearest displayed training endpoint"] = nearest
    aligned["distance to median displayed training endpoint"] = distance_to_median
    aligned["within 0.025 of displayed training endpoint"] = (
        aligned["distance to nearest displayed training endpoint"] <= 0.025 + 1e-12
    ).astype(int)

    participant = (
        aligned.groupby(["participant", "xai"])
        .agg(
            n_aligned_edits=("distance to nearest displayed training endpoint", "count"),
            mean_nearest_training_endpoint_distance=(
                "distance to nearest displayed training endpoint",
                "mean",
            ),
            median_nearest_training_endpoint_distance=(
                "distance to nearest displayed training endpoint",
                "median",
            ),
            rate_within_0025_training_endpoint=(
                "within 0.025 of displayed training endpoint",
                "mean",
            ),
        )
        .reset_index()
    )
    rows = []
    for metric in (
        "mean_nearest_training_endpoint_distance",
        "rate_within_0025_training_endpoint",
    ):
        arrays = [
            participant.loc[participant["xai"].eq(condition), metric].to_numpy(float)
            for condition in CONDITIONS
        ]
        h, p = stats.kruskal(*arrays)
        rows.append(
            {"metric": metric, "comparison": "omnibus", "effect": h, "p": p}
        )
        for left, right in combinations(CONDITIONS, 2):
            x = participant.loc[participant["xai"].eq(left), metric]
            y = participant.loc[participant["xai"].eq(right), metric]
            _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{left} vs {right}",
                    "effect": float(x.median() - y.median()),
                    "p": p,
                }
            )
    tests = pd.DataFrame(rows)
    tests["q within metric"] = tests.groupby("metric")["p"].transform(
        lambda values: bh(values)
    )
    return reference, participant, tests


def plots(participants: pd.DataFrame, cells: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"none": "#6B7280", "attribution": "#2563EB", "counterfactual": "#DC2626"}

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for condition in CONDITIONS:
        subset = participants.loc[participants["xai"].eq(condition)]
        axis.scatter(
            subset["fixed-delta CV MAE"],
            subset["threshold CV MAE"],
            s=np.where(subset["strong threshold-like"].eq(1), 95, 48),
            facecolors=np.where(subset["strong threshold-like"].eq(1), palette[condition], "none"),
            edgecolors=palette[condition],
            linewidths=1.4,
            label=condition,
            alpha=0.9,
        )
    limit = max(
        participants["threshold CV MAE"].max(), participants["fixed-delta CV MAE"].max()
    )
    axis.plot([0, limit], [0, limit], color="#111827", linestyle="--", linewidth=1)
    axis.axhline(0.05, color="#9CA3AF", linestyle=":", linewidth=1)
    axis.set(
        xlabel="Leave-one-out MAE: fixed-change model",
        ylabel="Leave-one-out MAE: endpoint-threshold model",
        title="Participant endpoint consistency versus fixed-change consistency",
        xlim=(0, limit * 1.03),
        ylim=(0, limit * 1.03),
    )
    axis.legend(title="XAI condition", frameon=True)
    figure.tight_layout()
    figure.savefig(OUT / "threshold_vs_fixed_delta.png", dpi=200)
    plt.close(figure)

    plot_cells = cells.copy()
    plot_cells["feature / target"] = (
        plot_cells["feature"] + "\n→ " + plot_cells["target label"]
    )
    figure, axis = plt.subplots(figsize=(13, 6.5))
    sns.pointplot(
        data=plot_cells,
        x="feature / target",
        y="estimated threshold",
        hue="xai",
        order=[f"{feature}\n→ {target}" for _, feature in FEATURES for target in TARGETS],
        hue_order=CONDITIONS,
        palette=palette,
        estimator=np.median,
        ci=95,
        dodge=0.45,
        join=False,
        capsize=0.08,
        ax=axis,
    )
    axis.set(
        xlabel="Feature and target label",
        ylabel="Median participant-level endpoint",
        title="Estimated target-specific endpoints by explanation condition",
        ylim=(-0.03, 1.03),
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="XAI condition")
    figure.tight_layout()
    figure.savefig(OUT / "threshold_levels_by_condition.png", dpi=200)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    long = prepare_long(data)
    scored = add_loo_predictions(long)
    participants, by_target = summarize_participants(scored)
    cells = cell_thresholds(long)
    condition = condition_summary(participants)
    tests = condition_tests(participants)
    model_tests = model_comparison_tests(participants)
    target_consistency_summary, target_consistency_tests = target_consistency_analysis(
        by_target
    )
    sensitivity = sensitivity_table(participants)
    threshold_summary, threshold_tests = threshold_level_analysis(cells)
    separation = target_separation(cells)
    separation_summary, separation_tests = target_separation_summary(separation)
    train_reference, train_alignment, train_tests = training_alignment(data, long)

    participants = participants.sort_values(
        ["strong threshold-like", "threshold CV MAE"], ascending=[False, True]
    )
    outputs = {
        "testing_changed_endpoints.csv": scored,
        "participant_threshold_consistency.csv": participants,
        "participant_threshold_consistency_by_target.csv": by_target,
        "participant_feature_target_thresholds.csv": cells,
        "condition_consistency_summary.csv": condition,
        "condition_consistency_tests.csv": tests,
        "model_comparison_tests.csv": model_tests,
        "target_consistency_summary.csv": target_consistency_summary,
        "target_consistency_tests.csv": target_consistency_tests,
        "threshold_definition_sensitivity.csv": sensitivity,
        "threshold_levels_by_condition.csv": threshold_summary,
        "threshold_level_condition_tests.csv": threshold_tests,
        "participant_feature_target_separation.csv": separation,
        "target_separation_summary.csv": separation_summary,
        "target_separation_tests.csv": separation_tests,
        "displayed_training_endpoints.csv": train_reference,
        "participant_training_endpoint_alignment.csv": train_alignment,
        "training_endpoint_alignment_tests.csv": train_tests,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)

    plots(participants, cells)

    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "participants": int(participants["participant"].nunique()),
        "testing_rows": int(data["phase"].eq("testing").sum()),
        "changed_feature_endpoints": int(len(long)),
        "near_exact_threshold_participants": int(
            participants["near-exact threshold"].sum()
        ),
        "strong_threshold_like_participants": int(
            participants["strong threshold-like"].sum()
        ),
        "near_exact_definition": {
            "minimum_evaluable_edits": 15,
            "minimum_evaluable_coverage": 0.60,
            "maximum_threshold_cv_mae": 0.05,
            "minimum_relative_gain_over_fixed_delta": 0.20,
            "minimum_rate_within_0.05": 0.75,
        },
        "strong_threshold_like_definition": {
            "minimum_evaluable_edits": 15,
            "minimum_evaluable_coverage": 0.60,
            "maximum_threshold_cv_mae": 0.075,
            "minimum_relative_gain_over_fixed_delta": 0.20,
            "maximum_absolute_endpoint_on_original_slope": 0.25,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
