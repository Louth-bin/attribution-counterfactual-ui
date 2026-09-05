"""Cluster v0.9 participants by matched-instance editing behaviour.

The unit of analysis is a participant.  Within each domain, participants are
compared only on testing instances answered by every participant in that
domain.  Each matched instance contributes two features:

* sparsity: number of attributes changed;
* proximity: summed absolute range-normalized change.

Each instance-feature column is standardized across participants before
clustering, preventing intrinsically easier or harder instances from driving
the solution.  Cluster labels are descriptive, not outcome labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "qualtrics" / "qualtrics_results_v0.9.for-plausibility.csv"
ASSIGNMENTS = ROOT / "qualtrics" / "v0.9_participant_change_clusters.csv"
DIAGNOSTICS = ROOT / "qualtrics" / "v0.9_participant_cluster_diagnostics.csv"
SUMMARY = ROOT / "qualtrics" / "v0.9_participant_cluster_summary.csv"
REPORT = ROOT / "qualtrics" / "v0.9_participant_clustering_report.md"

SEED = 20260823
MAX_K = 6
STABILITY_REPEATS = 250
INSTANCE_SUBSAMPLE_FRACTION = 0.80
SILHOUETTE_TOLERANCE = 0.02
MIN_CLUSTER_SIZE = 3
MIN_ALGORITHM_AGREEMENT = 0.75


def matched_matrix(group: pd.DataFrame) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """Return common instances, response rows, and participant feature matrix."""
    instance_sets = [set(values) for _, values in group.groupby("participant")["instance id"]]
    common_instances = sorted(set.intersection(*instance_sets))
    if not common_instances:
        raise ValueError(f"No common testing instances in domain {group['domain'].iat[0]!r}")

    matched = group[group["instance id"].isin(common_instances)].copy()
    duplicate_count = matched.duplicated(["participant", "instance id"]).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate participant-instance rows")

    matrix = matched.pivot(
        index="participant",
        columns="instance id",
        values=["sparsity", "proximity"],
    )
    ordered_columns = pd.MultiIndex.from_product(
        [["sparsity", "proximity"], common_instances],
        names=matrix.columns.names,
    )
    matrix = matrix.reindex(columns=ordered_columns).sort_index()
    if matrix.isna().any().any():
        raise ValueError("Matched participant matrix unexpectedly contains missing values")
    return common_instances, matched, matrix


def ward_instance_stability(
    matrix: pd.DataFrame,
    common_instances: list[int],
    k: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Test whether labels persist when 80% of matched instances are retained."""
    full_x = StandardScaler().fit_transform(matrix)
    full_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(full_x)
    retain = max(2, int(np.ceil(INSTANCE_SUBSAMPLE_FRACTION * len(common_instances))))
    scores: list[float] = []
    for _ in range(STABILITY_REPEATS):
        selected = set(rng.choice(common_instances, size=retain, replace=False).tolist())
        columns = [column for column in matrix.columns if column[1] in selected]
        subset_x = StandardScaler().fit_transform(matrix.loc[:, columns])
        subset_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
            subset_x
        )
        scores.append(adjusted_rand_score(full_labels, subset_labels))
    return float(np.mean(scores)), float(np.quantile(scores, 0.10))


def select_k(diagnostics: pd.DataFrame) -> int:
    """Select the smallest well-supported k near the best silhouette solution."""
    valid = diagnostics[
        (diagnostics["minimum cluster size"] >= MIN_CLUSTER_SIZE)
        & (diagnostics["kmeans-ward adjusted rand"] >= MIN_ALGORITHM_AGREEMENT)
    ]
    if valid.empty:
        valid = diagnostics[diagnostics["minimum cluster size"] >= MIN_CLUSTER_SIZE]
    if valid.empty:
        valid = diagnostics
    best = valid["silhouette"].max()
    eligible = valid[valid["silhouette"] >= best - SILHOUETTE_TOLERANCE]
    return int(eligible["k"].min())


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""
    printable = frame.copy()
    for column in printable.select_dtypes(include=["float"]).columns:
        printable[column] = printable[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in printable.columns]
    rows = [[str(value) for value in row] for row in printable.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def main() -> None:
    data = pd.read_csv(INPUT)
    required = {"participant", "domain", "phase", "instance id", "sparsity", "proximity"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    testing = data[data["phase"].astype(str).str.lower().eq("testing")].copy()
    domain_counts = testing.groupby("participant")["domain"].nunique()
    if not domain_counts.eq(1).all():
        raise ValueError("At least one participant occurs in multiple testing domains")

    all_assignments: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    matched_instance_notes: dict[str, list[int]] = {}
    selection_notes: dict[str, int] = {}
    rng = np.random.default_rng(SEED)

    for domain, group in testing.groupby("domain", sort=True):
        common_instances, matched, matrix = matched_matrix(group)
        matched_instance_notes[domain] = common_instances
        x = StandardScaler().fit_transform(matrix)
        maximum_k = min(MAX_K, len(matrix) - 1)

        diagnostic_rows: list[dict[str, float | int | str]] = []
        fitted: dict[int, KMeans] = {}
        for k in range(2, maximum_k + 1):
            kmeans = KMeans(n_clusters=k, n_init=100, random_state=SEED).fit(x)
            fitted[k] = kmeans
            ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
            stability_mean, stability_p10 = ward_instance_stability(
                matrix, common_instances, k, rng
            )
            sizes = np.bincount(kmeans.labels_, minlength=k)
            diagnostic_rows.append(
                {
                    "domain": domain,
                    "participants": len(matrix),
                    "matched instances": len(common_instances),
                    "k": k,
                    "silhouette": silhouette_score(x, kmeans.labels_),
                    "davies-bouldin": davies_bouldin_score(x, kmeans.labels_),
                    "calinski-harabasz": calinski_harabasz_score(x, kmeans.labels_),
                    "kmeans-ward adjusted rand": adjusted_rand_score(
                        kmeans.labels_, ward_labels
                    ),
                    "instance-subsample stability mean": stability_mean,
                    "instance-subsample stability p10": stability_p10,
                    "minimum cluster size": int(sizes.min()),
                    "cluster sizes": "/".join(str(int(size)) for size in sorted(sizes)),
                }
            )

        diagnostics = pd.DataFrame(diagnostic_rows)
        selected_k = select_k(diagnostics)
        diagnostics["selected"] = (diagnostics["k"] == selected_k).astype(int)
        all_diagnostics.append(diagnostics)
        selection_notes[domain] = selected_k

        raw_labels = fitted[selected_k].labels_
        participant_metrics = matched.groupby("participant", sort=True).agg(
            mean_attributes_changed=("sparsity", "mean"),
            sd_attributes_changed=("sparsity", "std"),
            mean_total_normalized_change=("proximity", "mean"),
            sd_total_normalized_change=("proximity", "std"),
        )
        participant_metrics = participant_metrics.reindex(matrix.index)
        participant_metrics["raw cluster"] = raw_labels

        raw_order = (
            participant_metrics.groupby("raw cluster")
            .agg(
                mean_total=("mean_total_normalized_change", "mean"),
                mean_count=("mean_attributes_changed", "mean"),
            )
            .sort_values(["mean_total", "mean_count"])
            .index
        )
        label_map = {int(raw): index + 1 for index, raw in enumerate(raw_order)}
        participant_metrics["participant change cluster"] = participant_metrics[
            "raw cluster"
        ].map(label_map)

        if selected_k == 2:
            strategy_map = {
                1: "Focused / smaller changes",
                2: "Broad / larger changes",
            }
        else:
            strategy_map = {
                cluster: f"Change strategy {cluster}" for cluster in range(1, selected_k + 1)
            }
        participant_metrics["participant change strategy"] = participant_metrics[
            "participant change cluster"
        ].map(strategy_map)
        participant_metrics["participant cluster silhouette"] = silhouette_samples(
            x, raw_labels
        )
        participant_metrics["domain"] = domain
        participant_metrics["matched instances used"] = len(common_instances)

        assignments = participant_metrics.reset_index()[
            [
                "participant",
                "domain",
                "participant change cluster",
                "participant change strategy",
                "participant cluster silhouette",
                "matched instances used",
                "mean_attributes_changed",
                "sd_attributes_changed",
                "mean_total_normalized_change",
                "sd_total_normalized_change",
            ]
        ]
        all_assignments.append(assignments)

        cluster_summary = (
            assignments.groupby(
                ["domain", "participant change cluster", "participant change strategy"],
                as_index=False,
            )
            .agg(
                participants=("participant", "size"),
                mean_attributes_changed=("mean_attributes_changed", "mean"),
                sd_between_participants_attributes=("mean_attributes_changed", "std"),
                mean_total_normalized_change=("mean_total_normalized_change", "mean"),
                sd_between_participants_change=("mean_total_normalized_change", "std"),
                mean_participant_silhouette=("participant cluster silhouette", "mean"),
            )
        )
        all_summaries.append(cluster_summary)

    assignments = pd.concat(all_assignments, ignore_index=True).sort_values(
        ["domain", "participant"]
    )
    diagnostics = pd.concat(all_diagnostics, ignore_index=True).sort_values(["domain", "k"])
    summaries = pd.concat(all_summaries, ignore_index=True).sort_values(
        ["domain", "participant change cluster"]
    )

    assignments.to_csv(ASSIGNMENTS, index=False, float_format="%.10g")
    diagnostics.to_csv(DIAGNOSTICS, index=False, float_format="%.10g")
    summaries.to_csv(SUMMARY, index=False, float_format="%.10g")

    report_diagnostics = diagnostics[
        [
            "domain",
            "k",
            "silhouette",
            "kmeans-ward adjusted rand",
            "instance-subsample stability mean",
            "minimum cluster size",
            "selected",
        ]
    ]
    report_summary = summaries[
        [
            "domain",
            "participant change cluster",
            "participant change strategy",
            "participants",
            "mean_attributes_changed",
            "mean_total_normalized_change",
            "mean_participant_silhouette",
        ]
    ]
    instance_lines = [
        f"- **{domain}:** {len(instances)} matched instances ({', '.join(map(str, instances))})."
        for domain, instances in matched_instance_notes.items()
    ]
    choice_lines = [
        f"- **{domain}:** k = {selection_notes[domain]}."
        for domain in sorted(selection_notes)
    ]
    report = f"""# Participant-level clustering of v0.9 editing responses

## What was clustered

The unit of clustering is the **participant**. The analysis uses testing responses only. Within each domain, every participant is represented by their number of changed attributes (`sparsity`) and total absolute range-normalized change (`proximity`) for the same matched instances. Each instance-metric column is standardized across participants before clustering, so instance difficulty does not create artificial participant groups.

Housing contains a mixed 10/20-instance design. The analysis uses the 10 instances shared by all housing participants and does not impute the other responses.

{chr(10).join(instance_lines)}

## Choosing the number of clusters

Candidate k values from 2 through {MAX_K} were compared using silhouette separation, agreement between k-means and Ward hierarchical clustering (adjusted Rand index), cluster size, and stability when 20% of matched instances were omitted. The deterministic selection rule chooses the smallest solution within {SILHOUETTE_TOLERANCE:.2f} of the best eligible silhouette, with at least {MIN_CLUSTER_SIZE} participants per cluster and k-means/Ward agreement of at least {MIN_ALGORITHM_AGREEMENT:.2f}.

{chr(10).join(choice_lines)}

{markdown_table(report_diagnostics)}

The two-cluster solution is clearly favored by silhouette for diabetes and housing. SafeLimit's three-cluster silhouette is only marginally higher, but it creates a two-person cluster and disagrees substantially across clustering algorithms; with only 12 SafeLimit participants, the simpler two-cluster solution is more defensible.

## General interpretation

{markdown_table(report_summary)}

- **Focused / smaller changes:** participants generally changed fewer attributes and made a smaller total range-normalized move.
- **Broad / larger changes:** participants generally changed more attributes and made a larger total range-normalized move.

These are descriptive editing strategies, not success or quality labels. XAI condition, labels, confidence, boundary distance, validity, and other outcomes were excluded from clustering.

## Cautions

The structure is exploratory rather than a claim that two latent psychological types truly exist. SafeLimit is especially uncertain because n = 12. Cluster assignments are domain-specific even though the two strategy names have the same general meaning. Signed change direction and which named attribute was changed were intentionally excluded because the requested clustering concerned the number and magnitude of changes.

## Reproducibility

Random seed: {SEED}. K-means uses 100 initializations. Stability uses {STABILITY_REPEATS} repetitions retaining {INSTANCE_SUBSAMPLE_FRACTION:.0%} of matched instances. All magnitude calculations use the study's existing summed range-normalized L1 `proximity` measure.
"""
    REPORT.write_text(report, encoding="utf-8")

    print(f"participants={len(assignments)}")
    print("selected=" + ",".join(f"{d}:{selection_notes[d]}" for d in sorted(selection_notes)))
    print(f"assignments={ASSIGNMENTS}")
    print(f"diagnostics={DIAGNOSTICS}")
    print(f"summary={SUMMARY}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
