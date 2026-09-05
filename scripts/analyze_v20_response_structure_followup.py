"""Focused follow-up on v2.0 response structure and the high-global strategy.

This script uses the broad search's enriched rows and keeps participant-level,
wave-stratified randomization tests.  The high-global strategy analyses are
explicitly descriptive because the cognitive parameter was fitted from the
same test responses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import StratifiedTable

from analyze_v20_xai_effect_search import (
    COMPARISONS,
    FEATURES,
    OUTDIR,
    bh_adjust,
    numeric,
    pairwise_test,
)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    testing = pd.read_csv(OUTDIR / "testing_rows_enriched.csv", low_memory=False)
    testing["one_feature_rate"] = testing["n_changed"].eq(1).astype(float)
    testing["two_feature_rate"] = testing["n_changed"].eq(2).astype(float)
    testing["three_plus_feature_rate"] = testing["n_changed"].ge(3).astype(float)
    testing["abs_change_per_changed_feature"] = np.divide(
        testing["edit_l1"], testing["n_changed"],
        out=np.zeros(len(testing)), where=testing["n_changed"].to_numpy() > 0,
    )
    testing["log_response_time"] = np.log(numeric(testing["response_time"]).clip(lower=0.05))
    for i, feature in enumerate(FEATURES, start=1):
        testing[f"select_{feature}"] = testing[f"delta_{i}"].abs().gt(1e-10).astype(float)

    mean_columns = [
        "n_changed", "one_feature_rate", "two_feature_rate", "three_plus_feature_rate",
        "edit_l1", "abs_change_per_changed_feature", "log_response_time",
        *[f"select_{feature}" for feature in FEATURES],
    ]
    participant = testing.groupby(["participant", "xai", "wave"], as_index=False)[mean_columns].mean()
    medians = testing.groupby("participant")["response_time"].median()
    participant["median_response_time"] = participant["participant"].map(medians)

    existing = pd.read_csv(OUTDIR / "participant_summary.csv", low_memory=False)
    behavior_columns = [
        "participant", "feature_selection_entropy", "modal_subset_share", "global_sign_consistency",
        "cognitive_family", "alpha", "eta", "rho", "training_direction_agreement",
        "boundary_new", "boundary_improvement", "success", "confidence_gain", "target_centroid_progress",
        "target_helpful_shap", "cf_demo_cosine", "cf_demo_feature_jaccard",
        "attribution_feature_overlap",
    ]
    participant = participant.merge(existing[behavior_columns], on="participant", how="left")
    participant["high_global_relevance_strategy"] = (
        participant["cognitive_family"].eq("feature contribution") & participant["alpha"].eq(1)
    ).astype(int)
    return testing, participant


def structure_effects(participant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "n_changed", "one_feature_rate", "two_feature_rate", "three_plus_feature_rate",
        "edit_l1", "abs_change_per_changed_feature", "feature_selection_entropy",
        "modal_subset_share", "global_sign_consistency", "median_response_time", "log_response_time",
        *[f"select_{feature}" for feature in FEATURES],
    ]
    rows = []
    seed = 200_000
    for metric in metrics:
        for group_a, group_b in COMPARISONS:
            result = pairwise_test(
                participant, metric, "xai", group_a, group_b,
                n_perm=9_999, n_boot=4_000, seed_offset=seed,
            )
            seed += 1
            rows.append({
                "metric": metric,
                "comparison": f"{group_a} - {group_b}",
                **result,
            })
    effects = pd.DataFrame(rows)
    effects["q_global"] = bh_adjust(effects["permutation_p"])
    effects["q_within_comparison"] = effects.groupby(
        "comparison", group_keys=False
    )["permutation_p"].apply(bh_adjust)

    wave_rows = []
    for wave, wave_data in participant.groupby("wave"):
        for metric in metrics:
            for group_a, group_b in COMPARISONS:
                result = pairwise_test(
                    wave_data, metric, "xai", group_a, group_b,
                    strata_col=None, n_perm=3_999, n_boot=2_000, seed_offset=seed,
                )
                seed += 1
                wave_rows.append({
                    "wave": wave,
                    "metric": metric,
                    "comparison": f"{group_a} - {group_b}",
                    **result,
                })
    wave_effects = pd.DataFrame(wave_rows)
    wave_effects["q_within_wave"] = wave_effects.groupby(
        "wave", group_keys=False
    )["permutation_p"].apply(bh_adjust)
    return effects.sort_values(["q_global", "permutation_p"]), wave_effects


def high_global_prevalence(participant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for group_a, group_b in COMPARISONS:
        sub = participant[participant["xai"].isin([group_a, group_b])]
        table = pd.crosstab(sub["xai"], sub["high_global_relevance_strategy"]).reindex(
            index=[group_a, group_b], columns=[1, 0], fill_value=0
        )
        odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
        strata_tables = []
        wave_counts = {}
        for wave, wave_data in sub.groupby("wave"):
            wave_table = pd.crosstab(
                wave_data["xai"], wave_data["high_global_relevance_strategy"]
            ).reindex(index=[group_a, group_b], columns=[1, 0], fill_value=0)
            strata_tables.append(wave_table.to_numpy())
            wave_counts[wave] = {
                group_a: {"high": int(wave_table.loc[group_a, 1]), "total": int(wave_table.loc[group_a].sum())},
                group_b: {"high": int(wave_table.loc[group_b, 1]), "total": int(wave_table.loc[group_b].sum())},
            }
        stratified = StratifiedTable(np.stack(strata_tables, axis=2))
        rows.append({
            "comparison": f"{group_a} vs {group_b}",
            "a_high": int(table.loc[group_a, 1]),
            "a_total": int(table.loc[group_a].sum()),
            "b_high": int(table.loc[group_b, 1]),
            "b_total": int(table.loc[group_b].sum()),
            "odds_ratio_unstratified": float(odds_ratio),
            "fisher_p": float(fisher_p),
            "mantel_haenszel_common_odds_ratio": float(stratified.oddsratio_pooled),
            "mantel_haenszel_p": float(stratified.test_null_odds().pvalue),
            "equal_odds_across_waves_p": float(stratified.test_equal_odds().pvalue),
            "wave_counts": json.dumps(wave_counts, sort_keys=True),
        })
    prevalence = pd.DataFrame(rows)
    prevalence["fisher_q"] = bh_adjust(prevalence["fisher_p"])

    summary = participant.groupby(["xai", "wave", "high_global_relevance_strategy"]).agg(
        participants=("participant", "size"),
        success=("success", "mean"),
        confidence_gain=("confidence_gain", "mean"),
        boundary_improvement=("boundary_improvement", "mean"),
        target_centroid_progress=("target_centroid_progress", "mean"),
        cf_demo_cosine=("cf_demo_cosine", "mean"),
        training_direction_agreement=("training_direction_agreement", "mean"),
        n_changed=("n_changed", "mean"),
    ).reset_index()
    return prevalence, summary


def high_global_outcomes(participant: pd.DataFrame) -> pd.DataFrame:
    cf = participant[participant["xai"].eq("counterfactual")].copy()
    outcomes = [
        "boundary_new", "boundary_improvement", "success", "confidence_gain", "edit_l1",
        "target_centroid_progress", "target_helpful_shap", "cf_demo_cosine",
        "cf_demo_feature_jaccard", "training_direction_agreement", "n_changed",
        "three_plus_feature_rate", "feature_selection_entropy", "modal_subset_share",
    ]
    records = []
    for i, outcome in enumerate(outcomes):
        result = pairwise_test(
            cf, outcome, "high_global_relevance_strategy", 1, 0,
            n_perm=9_999, n_boot=4_000, seed_offset=230_000 + i,
        )
        wave_diffs = {}
        for wave, wave_data in cf.groupby("wave"):
            high = wave_data.loc[wave_data["high_global_relevance_strategy"].eq(1), outcome]
            low = wave_data.loc[wave_data["high_global_relevance_strategy"].eq(0), outcome]
            wave_diffs[wave] = float(high.mean() - low.mean()) if len(high) and len(low) else np.nan
        records.append({
            "outcome": outcome,
            **result,
            "original_wave_difference": wave_diffs.get("original_36", np.nan),
            "new_wave_difference": wave_diffs.get("new_26", np.nan),
            "same_direction_both_waves": bool(
                np.sign(wave_diffs.get("original_36", np.nan))
                == np.sign(wave_diffs.get("new_26", np.nan))
            ),
            "interpretation_warning": "Post-hoc: strategy parameter was fitted from these same test responses.",
        })
    out = pd.DataFrame(records)
    out["q"] = bh_adjust(out["permutation_p"])
    return out.sort_values(["q", "permutation_p"])


def diffuse_edit_pca(participant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Treatment-blind dimension reduction of non-outcome response structure."""
    features = [
        "n_changed", "abs_change_per_changed_feature", "feature_selection_entropy",
        "modal_subset_share", *[f"select_{feature}" for feature in FEATURES],
    ]
    x = StandardScaler().fit_transform(participant[features].apply(numeric))
    model = PCA().fit(x)
    scores = model.transform(x)
    # Orient PC1 so positive values mean more diffuse editing.
    orientation = 1.0 if model.components_[0, features.index("n_changed")] >= 0 else -1.0
    participant = participant.copy()
    for index in range(3):
        participant[f"structure_pc{index + 1}"] = scores[:, index] * (orientation if index == 0 else 1.0)
    loadings = pd.DataFrame(
        {
            "feature": features,
            "PC1_loading": model.components_[0] * orientation,
            "PC2_loading": model.components_[1],
            "PC3_loading": model.components_[2],
        }
    )
    loadings.attrs["explained_variance"] = model.explained_variance_ratio_[:3].tolist()

    records = []
    seed = 250_000
    for component in ["structure_pc1", "structure_pc2", "structure_pc3"]:
        for group_a, group_b in COMPARISONS:
            result = pairwise_test(
                participant, component, "xai", group_a, group_b,
                n_perm=9_999, n_boot=4_000, seed_offset=seed,
            )
            seed += 1
            records.append({
                "component": component,
                "comparison": f"{group_a} - {group_b}",
                "explained_variance": float(
                    model.explained_variance_ratio_[int(component[-1]) - 1]
                ),
                **result,
            })
    effects = pd.DataFrame(records)
    effects["q"] = bh_adjust(effects["permutation_p"])
    return participant, loadings, effects.sort_values(["q", "permutation_p"])


def high_global_cluster_concordance(participant: pd.DataFrame) -> pd.DataFrame:
    clusters = pd.read_csv(OUTDIR / "participant_cluster_assignments.csv", usecols=["participant", "strategy_cluster"])
    merged = participant.merge(clusters, on="participant", how="left")
    cf = merged[merged["xai"].eq("counterfactual")]
    table = pd.crosstab(cf["high_global_relevance_strategy"], cf["strategy_cluster"]).reindex(
        index=[1, 0], columns=[1, 2], fill_value=0
    )
    odds, p = stats.fisher_exact(table.to_numpy())
    return pd.DataFrame([{
        "scope": "counterfactual participants only",
        "interpretation": "odds that the high-global fitted strategy falls in the poorer-response edit-pattern cluster",
        "high_global_poor_cluster": int(table.loc[1, 1]),
        "high_global_better_cluster": int(table.loc[1, 2]),
        "other_strategy_poor_cluster": int(table.loc[0, 1]),
        "other_strategy_better_cluster": int(table.loc[0, 2]),
        "odds_ratio": float(odds),
        "fisher_p": float(p),
        "warning": "Convergent but not independent: both labels were derived from the same response patterns.",
    }])


def main() -> None:
    testing, participant = prepare()
    effects, wave_effects = structure_effects(participant)
    prevalence, strategy_summary = high_global_prevalence(participant)
    strategy_outcomes = high_global_outcomes(participant)
    participant, pca_loadings, pca_effects = diffuse_edit_pca(participant)
    cluster_concordance = high_global_cluster_concordance(participant)

    participant.to_csv(OUTDIR / "response_structure_participant_summary.csv", index=False)
    effects.to_csv(OUTDIR / "response_structure_condition_effects.csv", index=False)
    wave_effects.to_csv(OUTDIR / "response_structure_wave_effects.csv", index=False)
    prevalence.to_csv(OUTDIR / "high_global_strategy_prevalence.csv", index=False)
    strategy_summary.to_csv(OUTDIR / "high_global_strategy_summary.csv", index=False)
    strategy_outcomes.to_csv(OUTDIR / "high_global_strategy_outcomes_within_counterfactual.csv", index=False)
    pca_loadings.to_csv(OUTDIR / "diffuse_edit_pca_loadings.csv", index=False)
    pca_effects.to_csv(OUTDIR / "diffuse_edit_pca_condition_effects.csv", index=False)
    cluster_concordance.to_csv(OUTDIR / "high_global_strategy_cluster_concordance.csv", index=False)

    condition_summary = participant.groupby("xai").agg(
        participants=("participant", "size"),
        n_changed=("n_changed", "mean"),
        three_plus_feature_rate=("three_plus_feature_rate", "mean"),
        insulin_selection=("select_Insulin", "mean"),
        edit_l1=("edit_l1", "mean"),
        abs_change_per_changed_feature=("abs_change_per_changed_feature", "mean"),
        median_response_time=("median_response_time", "mean"),
        log_response_time=("log_response_time", "mean"),
        high_global_strategy=("high_global_relevance_strategy", "mean"),
    ).reset_index()
    condition_summary.to_csv(OUTDIR / "response_structure_by_condition.csv", index=False)

    print("RESPONSE STRUCTURE EFFECTS")
    print(effects.head(20).round(4).to_string(index=False))
    print("\nHIGH-GLOBAL STRATEGY PREVALENCE")
    print(prevalence.round(4).to_string(index=False))
    print("\nHIGH-GLOBAL STRATEGY OUTCOMES WITHIN COUNTERFACTUAL")
    print(strategy_outcomes.round(4).to_string(index=False))
    print("\nDIFFUSE-EDIT PCA EFFECTS")
    print(pca_effects.round(4).to_string(index=False))
    print("\nHIGH-GLOBAL / UNSUPERVISED CLUSTER CONCORDANCE")
    print(cluster_concordance.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
