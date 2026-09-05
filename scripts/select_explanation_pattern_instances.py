"""Cluster explanation patterns and select balanced experiment instances.

Clustering uses explanations only:
- absolute, L1-normalized two-feature regularized Kernel SHAP values; and
- absolute normalized changes in the generated two-feature counterfactual.

Original feature values, prediction labels, change directions, model confidence,
and boundary distance are not clustering inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_clustered_explanation_instances import (  # noqa: E402
    explain_batch,
    json_value,
    local_boundary_distances,
    normalized_signed_delta,
)
from src.pipeline import ExplanationPipeline  # noqa: E402
from src.xai_methods.counterfactual import generate_counterfactual  # noqa: E402


DOMAINS = ("diabetes",)
NONZERO_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--boundary-quantile", type=float, default=0.20)
    parser.add_argument(
        "--boundary-quantile-overrides",
        nargs="*",
        default=[],
        metavar="DOMAIN=QUANTILE",
    )
    parser.add_argument("--minimum-clusters", type=int, default=3)
    # More clusters than attributes tends to split the data into the ten exact
    # two-feature pairs.  Limiting the search to 3--5 yields broader explanation
    # patterns while still choosing K algorithmically by silhouette score.
    parser.add_argument("--maximum-clusters", type=int, default=5)
    parser.add_argument("--represented-clusters", type=int, default=3)
    parser.add_argument("--training-per-cluster-label", type=int, default=2)
    parser.add_argument("--testing-per-cluster-label", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "analysis" / "explanation_pattern_instances_v1",
    )
    args = parser.parse_args()
    if not 0 < args.boundary_quantile <= 1:
        parser.error("--boundary-quantile must be in (0, 1].")
    return args


def boundary_quantile_overrides(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        domain, separator, quantile_text = value.partition("=")
        if not separator:
            raise ValueError(
                f"Boundary quantile override {value!r} must use DOMAIN=QUANTILE."
            )
        quantile = float(quantile_text)
        if not 0 < quantile <= 1:
            raise ValueError(f"Boundary quantile must be in (0, 1], found {quantile}.")
        result[domain.strip().casefold()] = quantile
    return result


def full_source_frame(dataset: Any) -> pd.DataFrame:
    frames = []
    for source_split, frame in (
        ("train", dataset.train_df),
        ("dev", dataset.dev_df),
        ("test", dataset.test_df),
    ):
        copy = frame.copy()
        copy["__source_split"] = source_split
        copy["__source_instance_id"] = np.arange(len(copy), dtype=int)
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


def closest_quantile_by_prediction(
    predictions: np.ndarray,
    boundary_distances: np.ndarray,
    quantile: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for prediction in (0, 1):
        positions = np.flatnonzero(predictions == prediction)
        ordered = positions[
            np.argsort(boundary_distances[positions], kind="stable")
        ]
        count = int(math.ceil(quantile * len(positions)))
        chosen = ordered[:count]
        selected.extend(chosen.tolist())
        diagnostics.append({
            "prediction": prediction,
            "available": int(len(positions)),
            "boundary_quantile": quantile,
            "shortlisted": int(count),
            "maximum_local_boundary_distance": float(
                boundary_distances[chosen[-1]]
            ),
        })
    return sorted(selected), diagnostics


def explanation_pattern(
    attribution_values: np.ndarray,
    normalized_changes: np.ndarray,
) -> np.ndarray:
    attribution = np.abs(np.asarray(attribution_values, dtype=float))
    attribution_total = float(np.sum(attribution))
    if attribution_total > 0:
        attribution = attribution / attribution_total
    # Retain absolute normalized change amounts. Direction is deliberately
    # discarded, but the amount of each intervention remains available.
    changes = np.abs(np.asarray(normalized_changes, dtype=float))
    return np.concatenate([attribution, changes])


def generate_domain_candidates(
    pipeline: ExplanationPipeline,
    domain: str,
    args: argparse.Namespace,
    boundary_quantile: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    assets = pipeline.prepare_assets(domain, "mlp")
    dataset = assets.dataset
    estimator = assets.model_artifact.estimator
    feature_names = list(dataset.feature_names)
    source = full_source_frame(dataset)
    feature_frame = source[feature_names]
    predictions = np.asarray(estimator.predict(feature_frame), dtype=int)
    probabilities = np.asarray(estimator.predict_proba(feature_frame), dtype=float)
    boundary_distances = local_boundary_distances(domain, dataset, feature_frame)
    shortlisted_positions, diagnostics = closest_quantile_by_prediction(
        predictions, boundary_distances, boundary_quantile
    )

    shortlisted_frame = feature_frame.iloc[shortlisted_positions].copy()
    background = dataset.train_df[feature_names].sample(
        n=min(len(dataset.train_df), 50), random_state=42
    )
    attribution_matrix = explain_batch(estimator, background, shortlisted_frame)
    candidates: list[dict[str, Any]] = []
    failures: Counter[tuple[int, str]] = Counter()

    for position, attribution_values in zip(
        shortlisted_positions, attribution_matrix
    ):
        prediction = int(predictions[position])
        selected_indices = np.flatnonzero(
            np.abs(attribution_values) > NONZERO_TOLERANCE
        ).astype(int).tolist()
        if len(selected_indices) != 2:
            failures[(prediction, "regularized_shap_not_exactly_two")] += 1
            continue
        reference = feature_frame.iloc[[position]].copy()
        counterfactual = generate_counterfactual(
            estimator=estimator,
            reference_frame=reference,
            target_distribution_frame=dataset.train_df[
                dataset.train_df[dataset.target_column] == (1 - prediction)
            ][feature_names],
            feature_names=feature_names,
            feature_types=list(dataset.feature_types),
            feature_ranges=list(dataset.feature_ranges),
            class_labels=list(dataset.class_labels),
            shap_values=attribution_values.tolist(),
            top_k=2,
            selected_feature_indices=selected_indices,
            generation_mode="minimal",
        )
        if counterfactual is None:
            failures[(prediction, "no_valid_two-feature_counterfactual")] += 1
            continue
        original_values = [
            json_value(reference.iloc[0][name]) for name in feature_names
        ]
        counterfactual_values = [
            json_value(value) for value in counterfactual["feature_values"]
        ]
        changes = np.asarray([
            normalized_signed_delta(original, updated, feature_type, feature_range)
            for original, updated, feature_type, feature_range in zip(
                original_values,
                counterfactual_values,
                dataset.feature_types,
                dataset.feature_ranges,
            )
        ], dtype=float)
        changed_indices = np.flatnonzero(np.abs(changes) > 1e-9).astype(int).tolist()
        if set(changed_indices) != set(selected_indices):
            failures[(prediction, "changed_features_do_not_match_attribution")] += 1
            continue
        candidates.append({
            "domain": domain,
            "source_split": str(source.iloc[position]["__source_split"]),
            "instance_id": int(source.iloc[position]["__source_instance_id"]),
            "prediction": prediction,
            "prediction_label": dataset.class_labels[prediction],
            "target_label": dataset.class_labels[1 - prediction],
            "positive_class_probability": float(probabilities[position, 1]),
            "local_boundary_distance": float(boundary_distances[position]),
            "raw_feature_values": original_values,
            "counterfactual_feature_values": counterfactual_values,
            "selected_indices": selected_indices,
            "selected_feature_names": [
                feature_names[index] for index in selected_indices
            ],
            "attribution_values": attribution_values.astype(float).tolist(),
            "normalized_changes": changes.tolist(),
            "counterfactual_distance": float(np.sum(np.abs(changes))),
            "counterfactual_target_probability": float(
                counterfactual["target_probability"]
            ),
            "explanation_pattern": explanation_pattern(
                attribution_values, changes
            ).tolist(),
            "counterfactual": counterfactual,
        })

    for row in diagnostics:
        prediction = int(row["prediction"])
        row.update({
            "domain": domain,
            "valid_explanation_candidates": sum(
                candidate["prediction"] == prediction for candidate in candidates
            ),
            "failures": sum(
                count for (label, _), count in failures.items()
                if label == prediction
            ),
            "failure_reasons": json.dumps({
                reason: count for (label, reason), count in failures.items()
                if label == prediction
            }, sort_keys=True),
        })
    return candidates, diagnostics, feature_names


def choose_kmeans(
    vectors: np.ndarray,
    minimum_clusters: int,
    maximum_clusters: int,
    seed: int,
) -> tuple[KMeans, list[dict[str, Any]]]:
    unique_count = len(np.unique(np.round(vectors, 12), axis=0))
    maximum = min(maximum_clusters, unique_count - 1, len(vectors) - 1)
    diagnostics: list[dict[str, Any]] = []
    fitted: dict[int, KMeans] = {}
    minimum_size = max(7, int(math.ceil(0.01 * len(vectors))))
    for cluster_count in range(minimum_clusters, maximum + 1):
        model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=50)
        labels = model.fit_predict(vectors)
        sizes = np.bincount(labels, minlength=cluster_count)
        diagnostics.append({
            "cluster_count": cluster_count,
            "silhouette_score": float(silhouette_score(vectors, labels)),
            "minimum_cluster_size": int(np.min(sizes)),
            "passes_minimum_size": int(np.min(sizes) >= minimum_size),
        })
        fitted[cluster_count] = model
    eligible = [row for row in diagnostics if row["passes_minimum_size"]]
    considered = eligible if eligible else diagnostics
    best = max(
        considered,
        key=lambda row: (row["silhouette_score"], -row["cluster_count"]),
    )
    return fitted[int(best["cluster_count"])], diagnostics


def select_balanced_instances(
    candidates: list[dict[str, Any]],
    model: KMeans,
    args: argparse.Namespace,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    vectors = np.asarray([
        candidate["explanation_pattern"] for candidate in candidates
    ])
    labels = model.predict(vectors)
    per_label = (
        args.training_per_cluster_label + args.testing_per_cluster_label
    )
    cluster_rows = []
    for cluster in range(model.n_clusters):
        cluster_indices = np.flatnonzero(labels == cluster)
        mean_attribution = np.mean(
            vectors[cluster_indices, :len(feature_names)], axis=0
        )
        dominant_indices = np.argsort(-mean_attribution, kind="stable")[:2]
        dominant_pair = tuple(
            feature_names[index] for index in sorted(dominant_indices)
        )
        cluster_rows.append({
            "cluster": cluster,
            "total": int(np.sum(labels == cluster)),
            "prediction_0": int(np.sum(
                (labels == cluster)
                & np.asarray([candidate["prediction"] == 0 for candidate in candidates])
            )),
            "prediction_1": int(np.sum(
                (labels == cluster)
                & np.asarray([candidate["prediction"] == 1 for candidate in candidates])
            )),
            "dominant_feature_pair": " | ".join(dominant_pair),
            "_dominant_features": frozenset(dominant_pair),
        })
    eligible = [
        row for row in cluster_rows
        if row["prediction_0"] >= per_label and row["prediction_1"] >= per_label
    ]
    if len(eligible) < args.represented_clusters:
        raise RuntimeError(
            f"Only {len(eligible)} clusters naturally contain {per_label} "
            "cases from each prediction. Increase --boundary-quantile."
        )

    # Use the most represented patterns.  Diversity is introduced when choosing
    # training examples inside these broad clusters, not by replacing prevalent
    # clusters with small ones.
    chosen_clusters = [
        row["cluster"]
        for row in sorted(
            eligible,
            key=lambda row: (-row["total"], row["cluster"]),
        )[:args.represented_clusters]
    ]

    rng = np.random.default_rng(args.seed)
    selected: list[dict[str, Any]] = []
    for cluster_rank, cluster in enumerate(chosen_clusters, start=1):
        for prediction in (0, 1):
            indices = [
                index for index, candidate in enumerate(candidates)
                if labels[index] == cluster and candidate["prediction"] == prediction
            ]
            # Sample naturally from the cluster.  There is no feature-pair
            # coverage objective and no preference for centroid or outlier
            # cases.  The fixed seed makes the selection reproducible.
            chosen = rng.choice(indices, size=per_label, replace=False).tolist()
            rng.shuffle(chosen)
            training = chosen[:args.training_per_cluster_label]
            testing = chosen[args.training_per_cluster_label:]

            for phase, phase_indices in (("training", training), ("testing", testing)):
                for candidate_index in phase_indices:
                    candidate = dict(candidates[candidate_index])
                    candidate["cluster"] = int(cluster)
                    candidate["cluster_rank"] = cluster_rank
                    candidate["distance_to_cluster_centroid"] = float(np.linalg.norm(
                        vectors[candidate_index] - model.cluster_centers_[cluster]
                    ))
                    candidate["experimental_phase"] = phase
                    candidate["selection_role"] = "natural_cluster_sample"
                    selected.append(candidate)
    selected.sort(key=lambda row: (
        row["experimental_phase"], row["cluster_rank"], row["prediction"],
        row["distance_to_cluster_centroid"],
    ))
    return selected, chosen_clusters, cluster_rows


def flattened_row(candidate: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    original = dict(zip(feature_names, candidate["raw_feature_values"]))
    counterfactual = dict(zip(
        feature_names, candidate["counterfactual_feature_values"]
    ))
    attribution = dict(zip(feature_names, candidate["attribution_values"]))
    changes = dict(zip(feature_names, candidate["normalized_changes"]))
    selected_names = candidate["selected_feature_names"]
    row = {
        "domain": candidate["domain"],
        "source_split": candidate["source_split"],
        "instance_id": candidate["instance_id"],
        "prediction": candidate["prediction"],
        "prediction_label": candidate["prediction_label"],
        "target_label": candidate["target_label"],
        "local_boundary_distance": candidate["local_boundary_distance"],
        "selected_features": " | ".join(selected_names),
        "attributions": " | ".join(
            f"{name}: {attribution[name]:+.6f}" for name in selected_names
        ),
        "counterfactual_changes": " | ".join(
            f"{name}: {original[name]} -> {counterfactual[name]} "
            f"({changes[name]:+.4f} normalized)"
            for name in selected_names
        ),
        "counterfactual_distance": candidate["counterfactual_distance"],
        "counterfactual_target_probability": candidate[
            "counterfactual_target_probability"
        ],
        "original_profile": json.dumps(original, separators=(",", ":")),
        "counterfactual_profile": json.dumps(
            counterfactual, separators=(",", ":")
        ),
        "attribution_vector": json.dumps(
            candidate["attribution_values"], separators=(",", ":")
        ),
        "normalized_change_vector": json.dumps(
            candidate["normalized_changes"], separators=(",", ":")
        ),
        "explanation_pattern": json.dumps(
            candidate["explanation_pattern"], separators=(",", ":")
        ),
    }
    for optional in (
        "cluster", "cluster_rank", "distance_to_cluster_centroid",
        "experimental_phase", "selection_role",
    ):
        if optional in candidate:
            row[optional] = candidate[optional]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pipeline = ExplanationPipeline()
    all_candidates: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    all_filter_rows: list[dict[str, Any]] = []
    all_k_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    feature_names_by_domain: dict[str, list[str]] = {}
    quantile_overrides = boundary_quantile_overrides(
        args.boundary_quantile_overrides
    )

    for domain in args.domains:
        domain_quantile = quantile_overrides.get(
            domain.casefold(), args.boundary_quantile
        )
        print(f"{domain}: generating full {domain_quantile:.0%} shortlist", flush=True)
        candidates, filter_rows, feature_names = generate_domain_candidates(
            pipeline, domain, args, domain_quantile
        )
        feature_names_by_domain[domain] = feature_names
        vectors = np.asarray([
            candidate["explanation_pattern"] for candidate in candidates
        ])
        model, k_rows = choose_kmeans(
            vectors,
            minimum_clusters=args.minimum_clusters,
            maximum_clusters=args.maximum_clusters,
            seed=args.seed,
        )
        labels = model.predict(vectors)
        for candidate, cluster in zip(candidates, labels):
            candidate["cluster"] = int(cluster)
        selected, chosen_clusters, cluster_rows = select_balanced_instances(
            candidates, model, args, feature_names
        )
        for row in filter_rows:
            all_filter_rows.append(row)
        for row in k_rows:
            all_k_rows.append({
                "domain": domain,
                **row,
                "chosen": int(row["cluster_count"] == model.n_clusters),
            })
        for row in cluster_rows:
            public_row = {
                key: value for key, value in row.items()
                if not key.startswith("_")
            }
            all_cluster_rows.append({
                "domain": domain,
                **public_row,
                "selected": int(row["cluster"] in chosen_clusters),
            })
        all_candidates.extend(candidates)
        all_selected.extend(selected)
        print(
            f"{domain}: candidates={len(candidates)} k={model.n_clusters} "
            f"chosen_clusters={chosen_clusters} selected={len(selected)}",
            flush=True,
        )

    prefix = args.output_prefix
    candidate_rows = [
        flattened_row(candidate, feature_names_by_domain[candidate["domain"]])
        for candidate in all_candidates
    ]
    selected_rows = [
        flattened_row(candidate, feature_names_by_domain[candidate["domain"]])
        for candidate in all_selected
    ]
    write_csv(prefix.with_name(prefix.name + "_candidate_shortlist.csv"), candidate_rows)
    write_csv(prefix.with_suffix(".csv"), selected_rows)
    write_csv(prefix.with_name(prefix.name + "_boundary_filter.csv"), all_filter_rows)
    write_csv(prefix.with_name(prefix.name + "_k_diagnostics.csv"), all_k_rows)
    write_csv(prefix.with_name(prefix.name + "_cluster_summary.csv"), all_cluster_rows)
    prefix.with_suffix(".json").write_text(json.dumps({
        "parameters": vars(args) | {"output_prefix": str(args.output_prefix)},
        "selected_instances": all_selected,
    }, indent=2), encoding="utf-8")
    prefix.with_suffix(".md").write_text(
        "\n".join([
            "# Explanation-pattern instance selection",
            "",
            f"Default boundary shortlist: closest {args.boundary_quantile:.0%} within each prediction over the full dataset. Overrides: {quantile_overrides or 'none'}.",
            "",
            "Clustering input: absolute L1-normalized regularized-SHAP values concatenated with absolute normalized counterfactual changes. No original profile values, labels, directions, confidence, or boundary distance are included.",
            "",
            f"Selected {len(all_selected)} cases: "
            f"{args.represented_clusters * 2 * args.training_per_cluster_label} "
            "training and "
            f"{args.represented_clusters * 2 * args.testing_per_cluster_label} "
            "testing per domain, with the required number from each prediction "
            "in every cluster.",
            "",
        ]),
        encoding="utf-8",
    )
    print(prefix.with_suffix(".csv"), flush=True)


if __name__ == "__main__":
    main()
