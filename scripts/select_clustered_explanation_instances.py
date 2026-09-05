"""Select explanation-balanced experimental instances for manual review.

The clustering representation concatenates:
1. two-feature regularized Kernel SHAP values, and
2. the independently optimized changes to those same two features.

Both blocks are oriented relative to the current prediction and L1-normalized.
Consequently, prediction label, model confidence, boundary margin, original
feature values, and absolute counterfactual distance do not enter clustering.
They remain available in the review output.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ExplanationPipeline  # noqa: E402
from src.xai_methods.counterfactual import generate_counterfactual  # noqa: E402
from src.xai_methods.shap_method import (  # noqa: E402
    _decode_rows,
    _encode_frames,
    _kernel_shap_sample_count,
)
from scripts.calculate_v09_boundary_distance import (  # noqa: E402
    WachterBoundarySearch,
    normalized_value,
)


DEFAULT_DOMAINS = ("housing", "safelimit", "diabetes")
NONZERO_TOLERANCE = 1e-12
PAIR_SEPARATOR = " | "
EXPERIMENT_BUNDLE = ROOT / "static" / "experiment-data.json"


class MatchedPredictionClustering:
    """Pair label-specific K-means clusters in one canonical vector space."""

    def __init__(
        self,
        models: dict[int, KMeans],
        local_for_pair: dict[int, dict[int, int]],
        labels: np.ndarray,
    ) -> None:
        self.models = models
        self.local_for_pair = local_for_pair
        self.pair_for_local = {
            prediction: {
                local_cluster: pair
                for pair, local_cluster in pair_mapping.items()
            }
            for prediction, pair_mapping in local_for_pair.items()
        }
        self.n_clusters = models[0].n_clusters
        self.labels_ = labels
        self.cluster_centers_ = np.asarray([
            (
                models[0].cluster_centers_[local_for_pair[0][pair]]
                + models[1].cluster_centers_[local_for_pair[1][pair]]
            ) / 2.0
            for pair in range(self.n_clusters)
        ])

    def predict(self, vectors: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        result = np.empty(len(vectors), dtype=int)
        for prediction in (0, 1):
            positions = np.flatnonzero(predictions == prediction)
            local = self.models[prediction].predict(vectors[positions])
            result[positions] = [
                self.pair_for_local[prediction][int(cluster)] for cluster in local
            ]
        return result

    def distance(self, vector: np.ndarray, prediction: int, pair: int) -> float:
        local_cluster = self.local_for_pair[prediction][pair]
        center = self.models[prediction].cluster_centers_[local_cluster]
        return float(np.linalg.norm(vector - center))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--boundary-quantile", type=float, default=0.20)
    parser.add_argument("--minimum-candidates-per-label", type=int, default=24)
    parser.add_argument("--maximum-candidates-per-label", type=int, default=300)
    parser.add_argument("--minimum-clusters", type=int, default=3)
    parser.add_argument("--maximum-clusters", type=int, default=10)
    parser.add_argument("--represented-clusters", type=int, default=3)
    parser.add_argument("--training-per-cluster-label", type=int, default=2)
    parser.add_argument("--testing-per-cluster-label", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "analysis" / "clustered_explanation_instances_v1",
    )
    args = parser.parse_args()
    if not 0 < args.boundary_quantile <= 1:
        parser.error("--boundary-quantile must be in (0, 1].")
    if args.represented_clusters < 1:
        parser.error("--represented-clusters must be positive.")
    return args


def positive_class_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[min(1, len(shap_values) - 1)], dtype=float)
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        return values[:, :, min(1, values.shape[2] - 1)]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unexpected SHAP output shape {values.shape}")


def explain_batch(
    estimator: Any,
    background: pd.DataFrame,
    instances: pd.DataFrame,
) -> np.ndarray:
    encoded_background, encoded_instances, category_maps = _encode_frames(
        background, instances
    )

    def wrapped_predict(encoded_rows: np.ndarray) -> np.ndarray:
        decoded = _decode_rows(
            encoded_rows=encoded_rows,
            reference_columns=list(instances.columns),
            reference_background=background,
            category_maps=category_maps,
        )
        return estimator.predict_proba(decoded)

    explainer = shap.KernelExplainer(
        wrapped_predict,
        encoded_background.to_numpy(dtype=float),
        feature_names=list(instances.columns),
    )
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        values = explainer.shap_values(
            encoded_instances.to_numpy(dtype=float),
            nsamples=_kernel_shap_sample_count(encoded_background.shape[1]),
            l1_reg="num_features(2)",
            silent=True,
        )
    finally:
        np.random.set_state(random_state)
    return positive_class_values(values)


def held_out_frame(dataset: Any) -> pd.DataFrame:
    dev = dataset.dev_df.copy()
    dev["__source_split"] = "dev"
    dev["__source_instance_id"] = np.arange(len(dev), dtype=int)
    test = dataset.test_df.copy()
    test["__source_split"] = "test"
    test["__source_instance_id"] = np.arange(len(test), dtype=int)
    return pd.concat([dev, test], ignore_index=True)


def select_near_boundary_candidates(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    boundary_distances: np.ndarray,
    boundary_quantile: float,
    minimum_per_label: int,
    maximum_per_label: int,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for label in sorted(np.unique(predictions).tolist()):
        label_indices = np.flatnonzero(predictions == label)
        ordered = label_indices[
            np.argsort(boundary_distances[label_indices], kind="stable")
        ]
        requested_count = max(
            int(math.ceil(boundary_quantile * len(label_indices))),
            minimum_per_label,
        )
        qualified_count = min(requested_count, len(label_indices))
        qualified = ordered[:qualified_count]
        if len(qualified) > maximum_per_label:
            sampled = rng.choice(qualified, size=maximum_per_label, replace=False)
            chosen = sorted(
                sampled.tolist(), key=lambda index: boundary_distances[index]
            )
        else:
            chosen = qualified.tolist()
        selected.extend(chosen)
        threshold = (
            float(boundary_distances[ordered[qualified_count - 1]])
            if qualified_count else math.nan
        )
        diagnostics.append({
            "prediction": int(label),
            "available_held_out": int(len(label_indices)),
            "requested_boundary_quantile": boundary_quantile,
            "qualified_by_effective_filter": int(qualified_count),
            "effective_boundary_fraction": qualified_count / len(label_indices),
            "candidate_sampled_for_explanations": int(len(chosen)),
            "maximum_local_boundary_distance": threshold,
        })
    return sorted(selected), diagnostics


def _bounded_local_l1_distance(
    normalized_numeric: np.ndarray,
    logit: float,
    gradient: np.ndarray,
) -> float:
    remaining = abs(float(logit))
    if remaining <= 1e-12:
        return 0.0
    distance = 0.0
    for index in np.argsort(-np.abs(gradient)):
        coefficient = float(gradient[index])
        if abs(coefficient) <= 1e-12:
            continue
        desired_sign = -np.sign(logit * coefficient)
        capacity = (
            1.0 - float(normalized_numeric[index])
            if desired_sign > 0
            else float(normalized_numeric[index])
        )
        if capacity <= 0:
            continue
        used = min(capacity, remaining / abs(coefficient))
        distance += used
        remaining -= used * abs(coefficient)
        if remaining <= 1e-10:
            return float(distance)
    return math.inf


def local_boundary_distances(
    domain: str,
    dataset: Any,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    bundle = json.loads(EXPERIMENT_BUNDLE.read_text(encoding="utf-8"))["datasets"]
    representative_case = bundle[domain]["test_pool"][0]
    solver = WachterBoundarySearch(
        bundle[domain]["browser_model"], representative_case
    )
    profiles = np.asarray([
        [
            normalized_value(row[name], feature_type, feature_range)
            for name, feature_type, feature_range in zip(
                dataset.feature_names,
                dataset.feature_types,
                dataset.feature_ranges,
            )
        ]
        for _, row in feature_frame.iterrows()
    ], dtype=float)
    category_options = [
        range(len(solver.feature_ranges[index]))
        for index in solver.categorical_indices
    ]
    combinations = list(itertools.product(*category_options)) if category_options else [tuple()]
    distances = np.full(len(profiles), math.inf, dtype=float)
    numeric_profiles = profiles[:, solver.numeric_indices]
    for categories in combinations:
        logits, gradients = solver._numeric_logits_and_gradients(
            numeric_profiles, categories
        )
        for row_index, (numeric, logit, gradient) in enumerate(
            zip(numeric_profiles, logits, gradients)
        ):
            categorical_cost = solver._categorical_cost(
                profiles[row_index], categories
            )
            numeric_distance = _bounded_local_l1_distance(
                numeric, float(logit), gradient
            )
            distances[row_index] = min(
                distances[row_index], categorical_cost + numeric_distance
            )
    # A local ReLU region can occasionally contain no feasible boundary point
    # inside the feature box.  Use the repository's multi-region boundary
    # solver only for those rare profiles rather than assigning an arbitrary
    # large distance.
    for row_index in np.flatnonzero(~np.isfinite(distances)).tolist():
        distances[row_index] = solver.solve(profiles[row_index]).distance
    return distances


def selected_indices(values: np.ndarray) -> list[int]:
    return np.flatnonzero(np.abs(values) > NONZERO_TOLERANCE).astype(int).tolist()


def normalized_signed_delta(
    original: Any,
    updated: Any,
    feature_type: str,
    feature_range: list[Any],
) -> float:
    if feature_type == "categorical":
        categories = [str(value) for value in feature_range]
        if len(categories) <= 1:
            return 0.0
        return (
            categories.index(str(updated)) - categories.index(str(original))
        ) / (len(categories) - 1)
    span = max(float(feature_range[1]) - float(feature_range[0]), 1e-12)
    return (float(updated) - float(original)) / span


def l1_normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(np.abs(values)))
    return values / total if total > 0 else values


def explanation_vector(
    attribution_values: np.ndarray,
    normalized_changes: np.ndarray,
    prediction: int,
) -> np.ndarray:
    orientation = 1.0 if prediction == 1 else -1.0
    oriented_attribution = l1_normalize(orientation * attribution_values)
    oriented_changes = l1_normalize(orientation * normalized_changes)
    return np.concatenate([oriented_attribution, oriented_changes])


def json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def build_valid_candidates(
    pipeline: ExplanationPipeline,
    domain: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    assets = pipeline.prepare_assets(domain, "mlp")
    dataset = assets.dataset
    estimator = assets.model_artifact.estimator
    feature_names = list(dataset.feature_names)
    held_out = held_out_frame(dataset)
    feature_frame = held_out[feature_names]
    probabilities = np.asarray(estimator.predict_proba(feature_frame), dtype=float)
    predictions = np.asarray(estimator.predict(feature_frame), dtype=int)
    boundary_distances = local_boundary_distances(domain, dataset, feature_frame)
    candidate_positions, filter_diagnostics = select_near_boundary_candidates(
        frame=held_out,
        predictions=predictions,
        boundary_distances=boundary_distances,
        boundary_quantile=args.boundary_quantile,
        minimum_per_label=args.minimum_candidates_per_label,
        maximum_per_label=args.maximum_candidates_per_label,
        seed=args.seed,
    )
    candidate_frame = feature_frame.iloc[candidate_positions].copy()
    background = dataset.train_df[feature_names].sample(
        n=min(len(dataset.train_df), 50), random_state=42
    )
    attribution_matrix = explain_batch(estimator, background, candidate_frame)

    valid_candidates: list[dict[str, Any]] = []
    failure_counts_by_prediction: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for position, attribution_values in zip(candidate_positions, attribution_matrix):
        prediction = int(predictions[position])
        failure_counts = failure_counts_by_prediction[prediction]
        selected = selected_indices(attribution_values)
        if len(selected) != 2:
            failure_counts["regularized_shap_not_exactly_two"] += 1
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
            selected_feature_indices=selected,
            generation_mode="minimal",
        )
        if counterfactual is None:
            failure_counts["no_two-feature_target-supporting_counterfactual"] += 1
            continue
        if int(counterfactual["prediction"]["value"]) == prediction:
            failure_counts["counterfactual_did_not_flip"] += 1
            continue
        original_values = [json_value(reference.iloc[0][name]) for name in feature_names]
        changed_values = [json_value(value) for value in counterfactual["feature_values"]]
        changes = np.asarray([
            normalized_signed_delta(original, updated, feature_type, feature_range)
            for original, updated, feature_type, feature_range in zip(
                original_values,
                changed_values,
                dataset.feature_types,
                dataset.feature_ranges,
            )
        ], dtype=float)
        if int(np.sum(np.abs(changes) > 1e-9)) != 2:
            failure_counts["counterfactual_not_exactly_two_changes"] += 1
            continue
        vector = explanation_vector(attribution_values, changes, prediction)
        valid_candidates.append({
            "domain": domain,
            "source_split": str(held_out.iloc[position]["__source_split"]),
            "instance_id": int(held_out.iloc[position]["__source_instance_id"]),
            "prediction": prediction,
            "prediction_label": dataset.class_labels[prediction],
            "target_prediction": 1 - prediction,
            "target_label": dataset.class_labels[1 - prediction],
            "positive_class_probability": float(probabilities[position, 1]),
            "boundary_margin": float(abs(probabilities[position, 1] - 0.5)),
            "local_boundary_distance": float(boundary_distances[position]),
            "raw_feature_values": original_values,
            "counterfactual_feature_values": changed_values,
            "attribution_values": attribution_values.astype(float).tolist(),
            "selected_indices": selected,
            "selected_feature_names": [feature_names[index] for index in selected],
            "normalized_changes": changes.tolist(),
            "counterfactual_distance": float(np.sum(np.abs(changes))),
            "counterfactual_target_probability": float(counterfactual["target_probability"]),
            "counterfactual": counterfactual,
            "clustering_vector": vector.tolist(),
        })

    for row in filter_diagnostics:
        failure_counts = failure_counts_by_prediction[int(row["prediction"])]
        row.update({
            "domain": domain,
            "valid_candidates": sum(
                candidate["prediction"] == row["prediction"]
                for candidate in valid_candidates
            ),
            "generation_failures": int(sum(failure_counts.values())),
            "failure_reasons": json.dumps(dict(failure_counts), sort_keys=True),
        })
    return valid_candidates, filter_diagnostics, feature_names


def choose_clustering(
    vectors: np.ndarray,
    predictions: np.ndarray,
    minimum_clusters: int,
    maximum_clusters: int,
    represented_clusters: int,
    required_per_cluster_label: int,
    seed: int,
) -> tuple[MatchedPredictionClustering, list[dict[str, Any]]]:
    unique_counts = {
        prediction: len(np.unique(
            np.round(vectors[predictions == prediction], 12), axis=0
        ))
        for prediction in (0, 1)
    }
    group_sizes = {
        prediction: int(np.sum(predictions == prediction))
        for prediction in (0, 1)
    }
    maximum = min(
        maximum_clusters,
        *(unique_counts[prediction] - 1 for prediction in (0, 1)),
        *(group_sizes[prediction] - 1 for prediction in (0, 1)),
    )
    if maximum < minimum_clusters:
        raise RuntimeError(
            f"Only {unique_counts} unique explanation vectors by prediction; cannot fit "
            f"at least {minimum_clusters} clusters."
        )
    diagnostics: list[dict[str, Any]] = []
    fitted: dict[int, MatchedPredictionClustering] = {}
    min_cluster_size = max(
        4, int(math.ceil(0.02 * min(group_sizes.values())))
    )
    for cluster_count in range(minimum_clusters, maximum + 1):
        models: dict[int, KMeans] = {}
        local_labels: dict[int, np.ndarray] = {}
        scores: dict[int, float] = {}
        local_sizes: dict[int, np.ndarray] = {}
        for prediction in (0, 1):
            group_vectors = vectors[predictions == prediction]
            model = KMeans(
                n_clusters=cluster_count, random_state=seed, n_init=50
            )
            labels = model.fit_predict(group_vectors)
            models[prediction] = model
            local_labels[prediction] = labels
            scores[prediction] = float(silhouette_score(group_vectors, labels))
            local_sizes[prediction] = np.bincount(
                labels, minlength=cluster_count
            )

        matching_cost = np.linalg.norm(
            models[0].cluster_centers_[:, None, :]
            - models[1].cluster_centers_[None, :, :],
            axis=2,
        )
        left_clusters, right_clusters = linear_sum_assignment(matching_cost)
        matched_pairs = sorted(zip(left_clusters.tolist(), right_clusters.tolist()))
        local_for_pair = {
            0: {pair: left for pair, (left, _) in enumerate(matched_pairs)},
            1: {pair: right for pair, (_, right) in enumerate(matched_pairs)},
        }
        pair_for_local = {
            prediction: {
                local: pair for pair, local in pair_mapping.items()
            }
            for prediction, pair_mapping in local_for_pair.items()
        }
        combined_labels = np.empty(len(vectors), dtype=int)
        for prediction in (0, 1):
            positions = np.flatnonzero(predictions == prediction)
            combined_labels[positions] = [
                pair_for_local[prediction][int(local)]
                for local in local_labels[prediction]
            ]
        matched = MatchedPredictionClustering(
            models=models,
            local_for_pair=local_for_pair,
            labels=combined_labels,
        )
        paired_sizes = {
            prediction: np.bincount(
                combined_labels[predictions == prediction],
                minlength=cluster_count,
            )
            for prediction in (0, 1)
        }
        balanced_eligible_clusters = sum(
            paired_sizes[0][pair] >= required_per_cluster_label
            and paired_sizes[1][pair] >= required_per_cluster_label
            for pair in range(cluster_count)
        )
        score = float(np.mean(list(scores.values())))
        diagnostics.append({
            "cluster_count": cluster_count,
            "silhouette_score": score,
            "silhouette_prediction_0": scores[0],
            "silhouette_prediction_1": scores[1],
            "minimum_cluster_size": int(min(
                np.min(local_sizes[0]), np.min(local_sizes[1])
            )),
            "passes_minimum_size": int(
                np.min(local_sizes[0]) >= min_cluster_size
                and np.min(local_sizes[1]) >= min_cluster_size
            ),
            "balanced_eligible_clusters": int(balanced_eligible_clusters),
            "passes_balanced_selection": int(
                balanced_eligible_clusters >= represented_clusters
            ),
        })
        fitted[cluster_count] = matched
    considered = [
        row for row in diagnostics
        if row["passes_minimum_size"] and row["passes_balanced_selection"]
    ]
    if not considered:
        raise RuntimeError(
            "No algorithmic cluster count contains enough natural members from "
            "both predictions in three clusters. Increase the boundary quantile "
            "or --minimum-candidates-per-label."
        )
    best = max(
        considered,
        key=lambda row: (row["silhouette_score"], -row["cluster_count"]),
    )
    return fitted[int(best["cluster_count"])], diagnostics


def balanced_select(
    candidates: list[dict[str, Any]],
    model: MatchedPredictionClustering,
    represented_clusters: int,
    training_per_cluster_label: int,
    testing_per_cluster_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    vectors = np.asarray([candidate["clustering_vector"] for candidate in candidates])
    predictions = np.asarray([
        candidate["prediction"] for candidate in candidates
    ], dtype=int)
    natural_labels = model.predict(vectors, predictions)
    per_cluster_label = training_per_cluster_label + testing_per_cluster_label
    counts = Counter(natural_labels.tolist())
    eligible_clusters = [
        cluster
        for cluster in range(model.n_clusters)
        if sum(
            natural_labels[index] == cluster and candidate["prediction"] == 0
            for index, candidate in enumerate(candidates)
        ) >= per_cluster_label
        and sum(
            natural_labels[index] == cluster and candidate["prediction"] == 1
            for index, candidate in enumerate(candidates)
        ) >= per_cluster_label
    ]
    chosen_clusters = [
        cluster
        for cluster in sorted(
            eligible_clusters, key=lambda cluster: (-counts[cluster], cluster)
        )
    ][:represented_clusters]
    if len(chosen_clusters) < represented_clusters:
        raise RuntimeError("Fewer populated clusters than requested represented clusters.")

    selected: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for cluster in chosen_clusters:
        for prediction in (0, 1):
            available_indices = [
                index for index, candidate in enumerate(candidates)
                if candidate["prediction"] == prediction
                and natural_labels[index] == cluster
            ]
            assigned = sorted(
                (
                    (
                        candidate_index,
                        model.distance(
                            vectors[candidate_index], prediction, cluster
                        ),
                    )
                    for candidate_index in available_indices
                ),
                key=lambda item: (item[1], item[0]),
            )[:per_cluster_label]
            if len(assigned) != per_cluster_label:
                raise RuntimeError("Natural-cluster selection produced an incomplete cell.")
            rng.shuffle(assigned)
            for position, (candidate_index, distance) in enumerate(assigned):
                candidate = dict(candidates[candidate_index])
                candidate["selected_cluster"] = int(cluster)
                candidate["natural_cluster"] = int(natural_labels[candidate_index])
                candidate["natural_cluster_match"] = int(
                    cluster == natural_labels[candidate_index]
                )
                candidate["distance_to_selected_cluster_centroid"] = distance
                candidate["experimental_phase"] = (
                    "training"
                    if position < training_per_cluster_label
                    else "testing"
                )
                selected.append(candidate)

    cluster_rank = {cluster: rank + 1 for rank, cluster in enumerate(chosen_clusters)}
    for candidate in selected:
        candidate["cluster_rank"] = cluster_rank[candidate["selected_cluster"]]
    selected.sort(key=lambda row: (
        row["experimental_phase"],
        row["cluster_rank"],
        row["prediction"],
        row["distance_to_selected_cluster_centroid"],
    ))
    return selected, chosen_clusters


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review_row(candidate: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    original = dict(zip(feature_names, candidate["raw_feature_values"]))
    updated = dict(zip(feature_names, candidate["counterfactual_feature_values"]))
    attribution = dict(zip(feature_names, candidate["attribution_values"]))
    changes = dict(zip(feature_names, candidate["normalized_changes"]))
    selected = candidate["selected_feature_names"]
    change_text = PAIR_SEPARATOR.join(
        f"{feature}: {original[feature]} -> {updated[feature]} "
        f"({changes[feature]:+.4f} normalized)"
        for feature in selected
    )
    attribution_text = PAIR_SEPARATOR.join(
        f"{feature}: {attribution[feature]:+.6f}" for feature in selected
    )
    return {
        "domain": candidate["domain"],
        "experimental_phase": candidate["experimental_phase"],
        "cluster_rank": candidate["cluster_rank"],
        "selected_cluster": candidate["selected_cluster"],
        "natural_cluster": candidate["natural_cluster"],
        "natural_cluster_match": candidate["natural_cluster_match"],
        "source_split": candidate["source_split"],
        "instance_id": candidate["instance_id"],
        "prediction": candidate["prediction"],
        "prediction_label": candidate["prediction_label"],
        "target_label": candidate["target_label"],
        "positive_class_probability": candidate["positive_class_probability"],
        "boundary_margin": candidate["boundary_margin"],
        "local_boundary_distance": candidate["local_boundary_distance"],
        "selected_features": PAIR_SEPARATOR.join(selected),
        "attributions": attribution_text,
        "counterfactual_changes": change_text,
        "counterfactual_distance": candidate["counterfactual_distance"],
        "counterfactual_target_probability": candidate["counterfactual_target_probability"],
        "distance_to_selected_cluster_centroid": candidate["distance_to_selected_cluster_centroid"],
        "original_profile": json.dumps(original, ensure_ascii=False, separators=(",", ":")),
        "counterfactual_profile": json.dumps(updated, ensure_ascii=False, separators=(",", ":")),
        "attribution_vector": json.dumps(candidate["attribution_values"], separators=(",", ":")),
        "normalized_change_vector": json.dumps(candidate["normalized_changes"], separators=(",", ":")),
    }


def report_text(
    args: argparse.Namespace,
    selected_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Clustered explanation instance review",
        "",
        "## Method",
        "",
        (
            f"The initial boundary parameter is the closest {args.boundary_quantile:.0%} "
            "of held-out profiles within each predicted class, measured by a "
            "box-constrained local linear approximation to normalized L1 distance "
            "from the profile to the model boundary. A configurable minimum candidate count can "
            "expand this fraction when the experimental quotas require more cases."
        ),
        "",
        (
            "Clustering concatenates the regularized two-feature SHAP vector and "
            "the independently optimized counterfactual-change vector. Each block "
            "is oriented relative to the current prediction and L1-normalized. "
            "Prediction label, confidence, boundary margin, original values, and "
            "absolute counterfactual distance are excluded from clustering."
        ),
        "",
        (
            "K-means is fitted separately within each prediction in the same "
            "canonical explanation space. A shared cluster count between "
            f"{args.minimum_clusters} and {args.maximum_clusters} maximizes mean "
            "silhouette subject to size and sampling safeguards. Opposite-prediction "
            "centroids are matched by minimum distance; the three largest eligible "
            "matched pairs are retained and sampled equally."
        ),
        "",
        "## Domain summary",
        "",
        "| Domain | Chosen k | Silhouette | Selected | Training | Testing | Natural-cluster matches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain in args.domains:
        domain_clusters = [row for row in cluster_rows if row["domain"] == domain]
        chosen = next(row for row in domain_clusters if row["chosen_model"] == 1)
        rows = [row for row in selected_rows if row["domain"] == domain]
        lines.append(
            f"| {domain} | {chosen['cluster_count']} | {chosen['silhouette_score']:.3f} | "
            f"{len(rows)} | {sum(row['experimental_phase'] == 'training' for row in rows)} | "
            f"{sum(row['experimental_phase'] == 'testing' for row in rows)} | "
            f"{sum(row['natural_cluster_match'] for row in rows)}/{len(rows)} |"
        )
    lines.extend([
        "",
        "## Boundary-filter diagnostics",
        "",
        "| Domain | Prediction | Held out | Effective fraction | Candidates | Valid two-feature CFs | Max local boundary distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in filter_rows:
        lines.append(
            f"| {row['domain']} | {row['prediction']} | {row['available_held_out']} | "
            f"{row['effective_boundary_fraction']:.1%} | "
            f"{row['candidate_sampled_for_explanations']} | {row['valid_candidates']} | "
            f"{row['maximum_local_boundary_distance']:.3f} |"
        )
    lines.extend([
        "",
        "The CSV contains every selected profile, its two regularized attributions, "
        "the independently generated two-feature counterfactual, boundary margin, "
        "cluster assignment, and normalized vectors for review.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pipeline = ExplanationPipeline()
    all_selected: list[dict[str, Any]] = []
    all_filter_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    feature_names_by_domain: dict[str, list[str]] = {}

    for domain in args.domains:
        print(f"{domain}: generating candidates", flush=True)
        candidates, filter_rows, feature_names = build_valid_candidates(
            pipeline, domain, args
        )
        feature_names_by_domain[domain] = feature_names
        all_filter_rows.extend(filter_rows)
        vectors = np.asarray([candidate["clustering_vector"] for candidate in candidates])
        model, diagnostics = choose_clustering(
            vectors,
            predictions=np.asarray([
                candidate["prediction"] for candidate in candidates
            ], dtype=int),
            minimum_clusters=args.minimum_clusters,
            maximum_clusters=args.maximum_clusters,
            represented_clusters=args.represented_clusters,
            required_per_cluster_label=(
                args.training_per_cluster_label + args.testing_per_cluster_label
            ),
            seed=args.seed,
        )
        selected, chosen_clusters = balanced_select(
            candidates=candidates,
            model=model,
            represented_clusters=args.represented_clusters,
            training_per_cluster_label=args.training_per_cluster_label,
            testing_per_cluster_label=args.testing_per_cluster_label,
            seed=args.seed,
        )
        all_selected.extend(selected)
        natural_counts = Counter(model.labels_.tolist())
        for row in diagnostics:
            all_cluster_rows.append({
                "domain": domain,
                **row,
                "chosen_model": int(row["cluster_count"] == model.n_clusters),
                "represented_cluster_ids": json.dumps(chosen_clusters)
                if row["cluster_count"] == model.n_clusters else "",
                "represented_cluster_sizes": json.dumps({
                    str(cluster): natural_counts[cluster]
                    for cluster in chosen_clusters
                }, sort_keys=True)
                if row["cluster_count"] == model.n_clusters else "",
            })
        print(
            f"{domain}: valid={len(candidates)} k={model.n_clusters} selected={len(selected)}",
            flush=True,
        )

    review_rows = [
        review_row(candidate, feature_names_by_domain[candidate["domain"]])
        for candidate in all_selected
    ]
    prefix = args.output_prefix
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    cluster_path = prefix.with_name(prefix.name + "_cluster_diagnostics.csv")
    filter_path = prefix.with_name(prefix.name + "_boundary_filter.csv")
    report_path = prefix.with_suffix(".md")
    write_csv(csv_path, review_rows, list(review_rows[0].keys()))
    write_csv(cluster_path, all_cluster_rows, list(all_cluster_rows[0].keys()))
    write_csv(filter_path, all_filter_rows, list(all_filter_rows[0].keys()))
    json_path.write_text(json.dumps({
        "parameters": vars(args) | {"output_prefix": str(args.output_prefix)},
        "selected_instances": all_selected,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        report_text(args, review_rows, all_cluster_rows, all_filter_rows),
        encoding="utf-8",
    )
    print(csv_path, flush=True)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
