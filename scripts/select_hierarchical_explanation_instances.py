"""Select diabetes instances with hierarchical, class-oriented clustering.

Level 1 is the exact pair of non-zero regularized-SHAP features.  Within each
pair, a Gaussian mixture partitions class-oriented signed attribution values
and class-oriented signed counterfactual changes.  The number of subclusters is
chosen independently by BIC, subject to a minimum subcluster size.

No absolute values are used in the clustering representation.  Prediction 0
is Diabetes and is multiplied by -1; prediction 1 is No Diabetes and is left
unchanged.  Mirrored explanations from the two predictions therefore align.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*")


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DEFAULT_INPUT = (
    ROOT / "analysis" /
    "diabetes_explanation_pattern_natural_v3_candidate_shortlist.csv"
)
DEFAULT_PREFIX = ROOT / "analysis" / "diabetes_hierarchical_oriented_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--main-clusters", type=int, default=3)
    parser.add_argument(
        "--selected-pairs",
        nargs="*",
        help=(
            "Optional exact main pairs to use instead of automatic prevalence selection; "
            "write each as, for example, 'glucose | insulin'."
        ),
    )
    parser.add_argument("--training-per-main-label", type=int, default=2)
    parser.add_argument("--testing-per-main-label", type=int, default=3)
    parser.add_argument("--maximum-subclusters", type=int, default=6)
    parser.add_argument("--minimum-subcluster-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def canonical_pair(value: str) -> tuple[str, str]:
    selected = [part.strip() for part in value.split("|")]
    return tuple(sorted(selected, key=FEATURE_NAMES.index))  # type: ignore[return-value]


def orient_explanation(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Orient signed values so mirrored predictions have the same sign."""
    prediction_sign = -1.0 if int(row["prediction"]) == 0 else 1.0
    attribution = parse_vector(row["attribution_vector"])
    attribution_total = float(np.sum(np.abs(attribution)))
    if attribution_total > 0:
        attribution = attribution / attribution_total
    changes = parse_vector(row["normalized_change_vector"])
    return prediction_sign * attribution, prediction_sign * changes


def pair_representation(group: pd.DataFrame, pair: tuple[str, str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    indices = [FEATURE_NAMES.index(name) for name in pair]
    raw_rows: list[dict[str, Any]] = []
    attributions: list[np.ndarray] = []
    changes: list[np.ndarray] = []
    for _, row in group.iterrows():
        oriented_attribution, oriented_change = orient_explanation(row)
        pair_attribution = oriented_attribution[indices]
        pair_change = oriented_change[indices]
        attributions.append(pair_attribution)
        changes.append(pair_change)
        raw_rows.append({
            "oriented_attribution_vector": oriented_attribution.tolist(),
            "oriented_change_vector": oriented_change.tolist(),
            "pair_oriented_attributions": pair_attribution.tolist(),
            "pair_oriented_changes": pair_change.tolist(),
        })

    # Scale the two explanation blocks independently.  This prevents the
    # attribution block (which sums to one) from overwhelming smaller changes.
    attribution_scaled = StandardScaler().fit_transform(np.vstack(attributions))
    change_scaled = StandardScaler().fit_transform(np.vstack(changes))
    vectors = np.hstack([
        attribution_scaled / math.sqrt(2),
        change_scaled / math.sqrt(2),
    ])
    return vectors, raw_rows


def fit_subclusters(
    vectors: np.ndarray,
    maximum_subclusters: int,
    minimum_subcluster_size: int,
    seed: int,
) -> tuple[GaussianMixture, np.ndarray, list[dict[str, Any]]]:
    maximum = max(1, min(maximum_subclusters, len(vectors) // minimum_subcluster_size))
    candidates: list[tuple[GaussianMixture, np.ndarray, dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for count in range(1, maximum + 1):
        model = GaussianMixture(
            n_components=count,
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=30,
            random_state=seed,
        )
        labels = model.fit_predict(vectors)
        sizes = np.bincount(labels, minlength=count)
        row = {
            "subcluster_count": count,
            "bic": float(model.bic(vectors)),
            "aic": float(model.aic(vectors)),
            "minimum_size": int(np.min(sizes)),
            "passes_minimum_size": int(np.min(sizes) >= minimum_subcluster_size),
        }
        diagnostics.append(row)
        candidates.append((model, labels, row))
    eligible = [candidate for candidate in candidates if candidate[2]["passes_minimum_size"]]
    considered = eligible if eligible else candidates[:1]
    model, labels, _ = min(considered, key=lambda candidate: (candidate[2]["bic"], candidate[2]["subcluster_count"]))
    return model, labels, diagnostics


def choose_main_pairs(
    summary: pd.DataFrame,
    count: int,
    required_per_label: int,
) -> list[str]:
    eligible = summary[
        (summary["prediction_0"] >= required_per_label)
        & (summary["prediction_1"] >= required_per_label)
        & (summary["training_pattern_available"] == 1)
    ].copy()
    if len(eligible) < count:
        raise RuntimeError(f"Only {len(eligible)} main clusters meet the per-label requirement.")

    best: tuple[Any, ...] | None = None
    best_pairs: list[str] | None = None
    for rows in itertools.combinations(eligible.to_dict("records"), count):
        feature_sets = [set(row["main_pair"].split(" | ")) for row in rows]
        if set.intersection(*feature_sets):
            continue
        total = sum(int(row["total"]) for row in rows)
        feature_coverage = len(set.union(*feature_sets))
        pair_names = sorted(str(row["main_pair"]) for row in rows)
        # Prevalence is primary.  Feature coverage breaks ties only.
        score = (total, feature_coverage, tuple(reversed(pair_names)))
        if best is None or score > best:
            best = score
            best_pairs = pair_names
    if best_pairs is None:
        raise RuntimeError("No eligible main-cluster combination has an empty feature intersection.")
    return sorted(
        best_pairs,
        key=lambda pair: (
            -int(eligible.loc[eligible["main_pair"] == pair, "total"].iloc[0]),
            pair,
        ),
    )


def validate_selected_pairs(
    requested: list[str],
    summary: pd.DataFrame,
    count: int,
    required_per_label: int,
) -> list[str]:
    if len(requested) != count:
        raise RuntimeError(f"Expected {count} explicitly selected pairs, found {len(requested)}.")
    canonical = [" | ".join(canonical_pair(value)) for value in requested]
    if len(set(canonical)) != len(canonical):
        raise RuntimeError("Explicitly selected pairs must be unique.")
    indexed = summary.set_index("main_pair")
    for pair in canonical:
        if pair not in indexed.index:
            raise RuntimeError(f"Unknown selected pair: {pair}")
        row = indexed.loc[pair]
        if (
            int(row["prediction_0"]) < required_per_label
            or int(row["prediction_1"]) < required_per_label
            or int(row["training_pattern_available"]) != 1
        ):
            raise RuntimeError(f"Selected pair does not meet the sampling requirements: {pair}")
    if set.intersection(*(set(pair.split(" | ")) for pair in canonical)):
        raise RuntimeError("Explicitly selected pairs must have an empty feature intersection.")
    return sorted(canonical, key=lambda pair: (-int(indexed.loc[pair, "total"]), pair))


def select_from_main_cluster(
    group: pd.DataFrame,
    training_per_label: int,
    testing_per_label: int,
) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    subcluster_sizes = group["subcluster"].value_counts().to_dict()

    pattern_counts = group.groupby(["oriented_pattern", "prediction"]).size().unstack(fill_value=0)
    eligible_patterns = pattern_counts[
        (pattern_counts.get(0, 0) >= training_per_label)
        & (pattern_counts.get(1, 0) >= training_per_label)
    ].copy()
    if eligible_patterns.empty:
        raise RuntimeError(f"{group['main_pair'].iloc[0]} has no direction-consistent training pattern.")
    eligible_patterns["total"] = eligible_patterns.sum(axis=1)
    training_pattern = str(
        eligible_patterns.sort_values("total", ascending=False).index[0]
    )

    training_indices: list[int] = []
    for prediction in (0, 1):
        label_group = group[
            (group["prediction"] == prediction)
            & (group["oriented_pattern"] == training_pattern)
        ].copy()
        label_group["subcluster_size"] = label_group["subcluster"].map(subcluster_sizes)
        training = label_group.sort_values(
            ["subcluster_size", "distance_to_subcluster_centroid", "instance_id"],
            ascending=[False, True, True],
        ).head(training_per_label)
        if len(training) < training_per_label:
            raise RuntimeError(
                f"{group['main_pair'].iloc[0]} has too few prediction-{prediction} training cases."
            )
        training["experimental_phase"] = "training"
        training["selection_role"] = "exact_oriented_pattern_then_centroid"
        training["training_direction_pattern"] = training_pattern
        training_indices.extend(training.index.tolist())
        selected_parts.append(training)

    remaining = group.drop(index=training_indices)
    for prediction in (0, 1):
        testing = remaining[remaining["prediction"] == prediction].copy()
        testing["subcluster_size"] = testing["subcluster"].map(subcluster_sizes)
        testing = testing.sort_values(
            ["subcluster_size", "distance_to_subcluster_centroid", "instance_id"],
            ascending=[False, True, True],
        ).head(testing_per_label)
        if len(testing) < testing_per_label:
            raise RuntimeError(
                f"{group['main_pair'].iloc[0]} has too few prediction-{prediction} testing cases."
            )
        testing["experimental_phase"] = "testing"
        testing["selection_role"] = "largest_subclusters_then_centroid"
        testing["training_direction_pattern"] = training_pattern
        selected_parts.append(testing)
    return pd.concat(selected_parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    frame["main_pair"] = frame["selected_features"].map(
        lambda value: " | ".join(canonical_pair(value))
    )
    frame["oriented_attribution_vector"] = ""
    frame["oriented_change_vector"] = ""
    frame["pair_oriented_attributions"] = ""
    frame["pair_oriented_changes"] = ""
    frame["subcluster"] = -1
    frame["distance_to_subcluster_centroid"] = np.nan

    main_rows: list[dict[str, Any]] = []
    subcluster_rows: list[dict[str, Any]] = []
    bic_rows: list[dict[str, Any]] = []
    processed_groups: list[pd.DataFrame] = []

    for pair_text, group in frame.groupby("main_pair", sort=False):
        pair = tuple(pair_text.split(" | "))
        vectors, raw_rows = pair_representation(group, pair)  # type: ignore[arg-type]
        model, labels, diagnostics = fit_subclusters(
            vectors,
            maximum_subclusters=args.maximum_subclusters,
            minimum_subcluster_size=args.minimum_subcluster_size,
            seed=args.seed,
        )
        means = model.means_
        distances = np.linalg.norm(vectors - means[labels], axis=1)
        updated = group.copy()
        for local_index, details in enumerate(raw_rows):
            row_index = updated.index[local_index]
            for key, value in details.items():
                updated.at[row_index, key] = json.dumps(value, separators=(",", ":"))
        updated["subcluster"] = labels.astype(int)
        updated["distance_to_subcluster_centroid"] = distances
        processed_groups.append(updated)

        main_rows.append({
            "main_pair": pair_text,
            "total": len(updated),
            "prediction_0": int((updated["prediction"] == 0).sum()),
            "prediction_1": int((updated["prediction"] == 1).sum()),
            "chosen_subclusters": int(model.n_components),
        })
        for diagnostic in diagnostics:
            bic_rows.append({
                "main_pair": pair_text,
                **diagnostic,
                "chosen": int(diagnostic["subcluster_count"] == model.n_components),
            })
        for subcluster, subcluster_group in updated.groupby("subcluster"):
            subcluster_rows.append({
                "main_pair": pair_text,
                "subcluster": int(subcluster),
                "total": len(subcluster_group),
                "prediction_0": int((subcluster_group["prediction"] == 0).sum()),
                "prediction_1": int((subcluster_group["prediction"] == 1).sum()),
                "mean_pair_oriented_attributions": json.dumps(
                    np.mean([
                        json.loads(value) for value in subcluster_group["pair_oriented_attributions"]
                    ], axis=0).tolist(), separators=(",", ":")
                ),
                "mean_pair_oriented_changes": json.dumps(
                    np.mean([
                        json.loads(value) for value in subcluster_group["pair_oriented_changes"]
                    ], axis=0).tolist(), separators=(",", ":")
                ),
            })

    processed = pd.concat(processed_groups).sort_index()
    def sign_pattern(serialized: str) -> str:
        return "".join(
            "+" if value > 1e-12 else "-" if value < -1e-12 else "0"
            for value in json.loads(serialized)
        )

    processed["oriented_attribution_signs"] = processed["pair_oriented_attributions"].map(sign_pattern)
    processed["oriented_change_signs"] = processed["pair_oriented_changes"].map(sign_pattern)
    processed["oriented_pattern"] = (
        processed["oriented_attribution_signs"] + "/" + processed["oriented_change_signs"]
    )

    for row in main_rows:
        group = processed[processed["main_pair"] == row["main_pair"]]
        pattern_counts = group.groupby(["oriented_pattern", "prediction"]).size().unstack(fill_value=0)
        eligible_patterns = pattern_counts[
            (pattern_counts.get(0, 0) >= args.training_per_main_label)
            & (pattern_counts.get(1, 0) >= args.training_per_main_label)
        ].copy()
        row["training_pattern_available"] = int(not eligible_patterns.empty)
        if eligible_patterns.empty:
            row["largest_eligible_training_pattern"] = ""
            row["largest_eligible_training_pattern_total"] = 0
        else:
            eligible_patterns["total"] = eligible_patterns.sum(axis=1)
            largest_pattern = eligible_patterns.sort_values("total", ascending=False).iloc[0]
            row["largest_eligible_training_pattern"] = str(largest_pattern.name)
            row["largest_eligible_training_pattern_total"] = int(largest_pattern["total"])
    main_summary = pd.DataFrame(main_rows).sort_values(["total", "main_pair"], ascending=[False, True])
    required = args.training_per_main_label + args.testing_per_main_label
    selected_pairs = (
        validate_selected_pairs(
            args.selected_pairs,
            main_summary,
            args.main_clusters,
            required,
        )
        if args.selected_pairs
        else choose_main_pairs(main_summary, args.main_clusters, required)
    )
    main_summary["selected"] = main_summary["main_pair"].isin(selected_pairs).astype(int)

    selected_parts = []
    for rank, pair in enumerate(selected_pairs, start=1):
        chosen = select_from_main_cluster(
            processed[processed["main_pair"] == pair],
            args.training_per_main_label,
            args.testing_per_main_label,
        )
        chosen["main_cluster_rank"] = rank
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["experimental_phase", "main_cluster_rank", "prediction", "subcluster", "distance_to_subcluster_centroid"]
    )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(prefix.with_name(prefix.name + "_candidate_shortlist.csv"), index=False)
    selected.to_csv(prefix.with_suffix(".csv"), index=False)
    main_summary.to_csv(prefix.with_name(prefix.name + "_main_clusters.csv"), index=False)
    pd.DataFrame(subcluster_rows).to_csv(prefix.with_name(prefix.name + "_subclusters.csv"), index=False)
    pd.DataFrame(bic_rows).to_csv(prefix.with_name(prefix.name + "_bic_diagnostics.csv"), index=False)

    selected_feature_sets = [set(pair.split(" | ")) for pair in selected_pairs]
    report = {
        "parameters": {
            "input": str(args.input),
            "main_clusters": args.main_clusters,
            "training_per_main_label": args.training_per_main_label,
            "testing_per_main_label": args.testing_per_main_label,
            "maximum_subclusters": args.maximum_subclusters,
            "minimum_subcluster_size": args.minimum_subcluster_size,
            "seed": args.seed,
            "explicitly_selected_pairs": bool(args.selected_pairs),
            "orientation": "multiply signed attribution and signed CF change vectors by -1 for prediction 0 (Diabetes); prediction 1 (No Diabetes) unchanged",
            "absolute_values_used": False,
            "subcluster_selection": "diagonal Gaussian mixture; minimum BIC among solutions satisfying minimum subcluster size",
        },
        "selected_main_pairs": selected_pairs,
        "selected_main_pair_intersection": sorted(set.intersection(*selected_feature_sets)),
        "selected_main_pair_union": sorted(set.union(*selected_feature_sets)),
        "selected_counts": {
            "total": len(selected),
            "training": int((selected["experimental_phase"] == "training").sum()),
            "testing": int((selected["experimental_phase"] == "testing").sum()),
            "prediction_0": int((selected["prediction"] == 0).sum()),
            "prediction_1": int((selected["prediction"] == 1).sum()),
        },
        "selected_instances": selected.to_dict("records"),
    }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_main_pairs": selected_pairs,
        "intersection": report["selected_main_pair_intersection"],
        "counts": report["selected_counts"],
        "output": str(prefix.with_suffix('.csv')),
    }, indent=2))


if __name__ == "__main__":
    main()
