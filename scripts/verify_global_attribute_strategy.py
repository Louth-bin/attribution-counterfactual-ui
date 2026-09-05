"""Alternative checks for the fitted global-attribute counterfactual strategy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


GLOBAL = {
    "change_most_influential_attributes_by_fixed_amount",
    "change_most_influential_attributes_to_flip",
}
FAMILY = {
    "change_most_contributing_attributes": "Current-case contribution",
    "change_most_influential_attributes_to_flip": "Global importance",
    "change_most_influential_attributes_by_fixed_amount": "Global importance",
    "set_attributes_to_target_profile_values": "Target-profile mismatch",
    "change_attributes_not_matching_target_profile": "Target-profile mismatch",
    "change_largest_target_profile_mismatches": "Target-profile mismatch",
    "change_toward_remembered_exemplar": "Nearest target example",
    "copy_changes_from_remembered_example": "Similar remembered edit",
}
FEATURES = tuple(f"x_{index}_changed" for index in range(1, 6))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def f1(observed: set[int], predicted: set[int]) -> float:
    denominator = len(observed) + len(predicted)
    return 1.0 if denominator == 0 else 2 * len(observed & predicted) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    participant_rows = {row["participant"]: row for row in read_csv(args.participants)}
    trial_rows = read_csv(args.trials)
    result_rows = [row for row in read_csv(args.results) if row["phase"] == "testing"]

    trial_f1: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in trial_rows:
        trial_f1[(row["participant"], row["strategy"])].append(
            float(row["feature_selection_f1"])
        )

    feature_winner: dict[str, str] = {}
    feature_winner_gap: dict[str, float] = {}
    feature_family_winner: dict[str, str] = {}
    feature_family_gap: dict[str, float] = {}
    feature_family_score: dict[str, float] = {}
    for participant in participant_rows:
        means = {
            strategy: float(np.mean(values))
            for (current, strategy), values in trial_f1.items()
            if current == participant
        }
        ordered = sorted(means, key=lambda strategy: (-means[strategy], strategy))
        feature_winner[participant] = ordered[0]
        feature_winner_gap[participant] = means[ordered[0]] - means[ordered[1]]
        family_means: dict[str, float] = {}
        for strategy, score in means.items():
            family = FAMILY[strategy]
            family_means[family] = max(family_means.get(family, -1.0), score)
        family_order = sorted(
            family_means, key=lambda family: (-family_means[family], family)
        )
        feature_family_winner[participant] = family_order[0]
        feature_family_score[participant] = family_means[family_order[0]]
        feature_family_gap[participant] = (
            family_means[family_order[0]] - family_means[family_order[1]]
        )

    observed_sets: dict[str, list[set[int]]] = defaultdict(list)
    null_loss: dict[str, list[float]] = defaultdict(list)
    participant_domain: dict[str, str] = {}
    for row in result_rows:
        participant = row["participant"]
        participant_domain[participant] = row["domain"]
        observed_sets[participant].append(
            {index for index, name in enumerate(FEATURES) if int(float(row[name])) == 1}
        )
        null_loss[participant].append(
            sum(abs(float(row[f"x_{index}_change"])) for index in range(1, 6)) / 5.0
        )

    stability: dict[str, dict[str, float | int | str]] = {}
    for participant, sets in observed_sets.items():
        frequencies = Counter(index for selected in sets for index in selected)
        ranked = sorted(range(5), key=lambda index: (-frequencies[index], index))
        candidates = []
        for k in range(1, 6):
            fixed = set(ranked[:k])
            candidates.append((float(np.mean([f1(selected, fixed) for selected in sets])), k))
        best_fixed_f1, best_k = max(candidates, key=lambda item: (item[0], -item[1]))
        exact = Counter(tuple(sorted(selected)) for selected in sets)
        stability[participant] = {
            "best_fixed_set_f1": best_fixed_f1,
            "best_fixed_k": best_k,
            "modal_exact_subset_fraction": exact.most_common(1)[0][1] / len(sets),
            "unique_subsets": len(exact),
        }

    records = []
    for participant, row in participant_rows.items():
        best = row["best_counterfactual_strategy"]
        minimum = row["minimum_loss_counterfactual_strategy"]
        feature = feature_winner[participant]
        records.append(
            {
                "participant": participant,
                "domain": participant_domain[participant],
                "best_global": best in GLOBAL,
                "minimum_global": minimum in GLOBAL,
                "feature_global": feature in GLOBAL,
                "best": best,
                "minimum": minimum,
                "feature": feature,
                "feature_gap": feature_winner_gap[participant],
                "feature_family": feature_family_winner[participant],
                "feature_family_gap": feature_family_gap[participant],
                "feature_family_score": feature_family_score[participant],
                "null_edit_loss": float(np.mean(null_loss[participant])),
                "best_edit_loss": float(row["best_strategy_cv_mean_normalized_l1"]),
                **stability[participant],
            }
        )

    original_global = [row for row in records if row["best_global"]]
    non_global = [row for row in records if not row["best_global"]]

    def count(rows: list[dict], predicate) -> int:
        return sum(bool(predicate(row)) for row in rows)

    strict = [
        row for row in records
        if row["best_global"]
        and row["minimum_global"]
        and row["feature_global"]
        and float(row["best_fixed_set_f1"]) >= 0.70
    ]

    family_counts = Counter(row["feature_family"] for row in records)
    family_counts_gap_005 = Counter(
        row["feature_family"]
        for row in records
        if float(row["feature_family_gap"]) >= 0.05
    )
    family_counts_gap_005_f1_05 = Counter(
        row["feature_family"]
        for row in records
        if float(row["feature_family_gap"]) >= 0.05
        and float(row["feature_family_score"]) >= 0.50
    )
    original_global_improvements = [
        1.0 - float(row["best_edit_loss"]) / float(row["null_edit_loss"])
        if float(row["null_edit_loss"]) > 0
        else 0.0
        for row in original_global
    ]
    for row in records:
        row["relative_improvement_vs_no_change"] = (
            1.0 - float(row["best_edit_loss"]) / float(row["null_edit_loss"])
            if float(row["null_edit_loss"]) > 0
            else 0.0
        )
    robust_global_selection = [
        row for row in records
        if row["feature_family"] == "Global importance"
        and float(row["feature_family_gap"]) >= 0.05
        and float(row["feature_family_score"]) >= 0.50
    ]
    consensus_global = [
        row for row in robust_global_selection
        if row["best_global"]
        and float(row["relative_improvement_vs_no_change"]) >= 0.10
        and float(row["best_fixed_set_f1"]) >= 0.70
    ]
    stable_robust_global = [
        row for row in robust_global_selection
        if float(row["best_fixed_set_f1"]) >= 0.70
    ]

    output = {
        "participants": len(records),
        "original_one_se_global": len(original_global),
        "minimum_loss_global_all": count(records, lambda row: row["minimum_global"]),
        "feature_selection_only_global_all": count(records, lambda row: row["feature_global"]),
        "within_original_global": {
            "minimum_loss_also_global": count(original_global, lambda row: row["minimum_global"]),
            "feature_selection_only_also_global": count(original_global, lambda row: row["feature_global"]),
            "both_alternatives_also_global": count(
                original_global, lambda row: row["minimum_global"] and row["feature_global"]
            ),
            "best_fixed_set_f1_at_least_0.70": count(
                original_global, lambda row: float(row["best_fixed_set_f1"]) >= 0.70
            ),
            "median_best_fixed_set_f1": float(
                np.median([row["best_fixed_set_f1"] for row in original_global])
            ),
            "median_modal_exact_subset_fraction": float(
                np.median([row["modal_exact_subset_fraction"] for row in original_global])
            ),
            "median_unique_subsets": float(
                np.median([row["unique_subsets"] for row in original_global])
            ),
        },
        "non_global_comparison": {
            "median_best_fixed_set_f1": float(
                np.median([row["best_fixed_set_f1"] for row in non_global])
            ),
            "median_modal_exact_subset_fraction": float(
                np.median([row["modal_exact_subset_fraction"] for row in non_global])
            ),
            "median_unique_subsets": float(
                np.median([row["unique_subsets"] for row in non_global])
            ),
        },
        "strict_four-part_verification": {
            "definition": "one-SE global + minimum-loss global + feature-only global + fixed-set F1 >= 0.70",
            "count": len(strict),
            "by_domain": dict(sorted(Counter(row["domain"] for row in strict).items())),
        },
        "feature_selection_family_check": {
            "top_family_without_margin": dict(sorted(family_counts.items())),
            "top_family_with_f1_gap_at_least_0.05": dict(
                sorted(family_counts_gap_005.items())
            ),
            "top_family_with_gap_at_least_0.05_and_f1_at_least_0.50": dict(
                sorted(family_counts_gap_005_f1_05.items())
            ),
            "ambiguous_gap_below_0.05": count(
                records, lambda row: float(row["feature_family_gap"]) < 0.05
            ),
            "robust_global_selection_by_domain": dict(
                sorted(Counter(row["domain"] for row in robust_global_selection).items())
            ),
            "robust_and_stable_global_count": len(stable_robust_global),
            "robust_and_stable_global_by_domain": dict(
                sorted(Counter(row["domain"] for row in stable_robust_global).items())
            ),
            "robust_and_stable_global_participants": [
                row["participant"] for row in sorted(
                    stable_robust_global, key=lambda row: (row["domain"], row["participant"])
                )
            ],
        },
        "original_global_edit_vector_fit_vs_no_change": {
            "median_relative_improvement": float(np.median(original_global_improvements)),
            "participants_at_least_10_percent_better": sum(
                improvement >= 0.10 for improvement in original_global_improvements
            ),
            "participants_not_better_than_no_change": sum(
                improvement <= 0 for improvement in original_global_improvements
            ),
        },
        "consensus_global_check": {
            "definition": "robust feature-selection family + original global winner + >=10% improvement over no-change + fixed-set F1 >=0.70",
            "count": len(consensus_global),
            "by_domain": dict(
                sorted(Counter(row["domain"] for row in consensus_global).items())
            ),
        },
        "one_se_global_by_domain": dict(
            sorted(Counter(row["domain"] for row in original_global).items())
        ),
        "detailed_transition_from_one_se_global_to_minimum_loss": {
            f"{row['best']} -> {row['minimum']}": 0 for row in []
        },
    }
    output["detailed_transition_from_one_se_global_to_minimum_loss"] = dict(
        sorted(Counter(f"{row['best']} -> {row['minimum']}" for row in original_global).items())
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
