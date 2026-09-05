"""Build the complete v1.0 results CSV used to update the existing JMP table.

The script preserves the existing v1.0 column set and participant-level model
fits, adds newly completed responses, scores CRT-2 for new participants, and
recomputes the two higher-is-better editing metrics requested for every row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = {
    "housing": ("sqft_living", "bedrooms", "bathrooms", "floors", "grade"),
    "diabetes": ("glucose", "blood_pressure", "insulin", "bmi", "age"),
    "safelimit": ("units", "weight", "duration", "gender", "stomach_fullness"),
}
NORMALIZED_VALUE_RE = re.compile(
    r"\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def read_qualtrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[3:]]


def number(value: str | float | int | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def normalized_profile(attribute_text: str) -> np.ndarray:
    lines = [line for line in attribute_text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError(f"Expected five attribute lines, found {len(lines)}")
    values = []
    for line in lines:
        matches = NORMALIZED_VALUE_RE.findall(line)
        if not matches:
            raise ValueError(f"No normalized value in {line!r}")
        values.append(float(matches[-1]))
    return np.asarray(values, dtype=float)


def normalize_training_value(value: str, kind: str, levels: list[object]) -> float:
    if kind == "categorical":
        categories = [str(item).casefold() for item in levels]
        index = categories.index(str(value).casefold())
        return index / (len(categories) - 1) if len(categories) > 1 else 0.0
    low, high = float(levels[0]), float(levels[1])
    if math.isclose(low, high):
        return 0.0
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def reference_matrices(bundle_path: Path) -> dict[str, np.ndarray]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    for domain, features in FEATURE_NAMES.items():
        case = bundle["datasets"][domain]["test_pool"][0]
        kinds = list(case["feature_types"])
        ranges = list(case["raw_feature_ranges"])
        rows = []
        _, records = read_csv(ROOT / "src" / "data" / domain / "train.csv")
        for record in records:
            rows.append(
                [
                    normalize_training_value(record[name], kinds[index], ranges[index])
                    for index, name in enumerate(features)
                ]
            )
        result[domain] = np.asarray(rows, dtype=float)
    return result


def score_crt(response: dict[str, str]) -> int:
    def clean(name: str) -> str:
        return " ".join(response.get(name, "").casefold().strip().split())

    race = clean("CRT2_Race")
    sheep = clean("CRT2_Sheep")
    emily = clean("CRT2_Emily")
    hole = clean("CRT2_Hole")
    race_ok = "second" in race or bool(re.search(r"\b2(?:nd)?\b", race))
    sheep_ok = "eight" in sheep or bool(re.search(r"\b8\b", sheep))
    emily_ok = "emily" in emily
    hole_ok = (
        "no dirt" in hole
        or "none" in hole
        or "nothing" in hole
        or "zero" in hole
        or bool(re.search(r"\b0(?:\.0+)?\b", hole))
    )
    return int(race_ok) + int(sheep_ok) + int(emily_ok) + int(hole_ok)


REASONING_STRATEGY = {
    "R_1CkJOzM85nYSCk1": "Change the most influential attributes for the case",
    "R_1fpKwE5dFxzF6vi": "Change the most out-of-range attributes for the case",
    "R_32YjhrSOKPpEfJf": "Change the most influential attributes for the case",
    "R_3dnediamQ206QWR": "No consistent rule or guessed changes",
    "R_5c5e5Qjh8Qn7MSE": "Change the most influential attributes for the case",
    "R_72Hr6eK5zyGxfZ7": "Change the most influential attributes for the case",
    "R_7Kx6vyCCC3HUpXY": "Change the most influential attributes for the case",
    "R_1FwylBEbSuK0oNf": "Change the most influential attributes for the case",
    "R_3OZ3Mh6AZsBTXup": "No consistent rule or guessed changes",
    "R_3cKOdqKWicLMTCx": "Change all mismatching attributes toward a target profile",
    "R_3irfiuQFDddFGoo": "Change all mismatching attributes toward a target profile",
    "R_52EcHdeCk0DGfHz": "Change the most influential attributes for the case",
    "R_5BSmZ6kOyBcgvGO": "Change the most out-of-range attributes for the case",
    "R_5a6S4hSnNMThfbj": "Change the most influential attributes for the case",
    "R_7K1IOxgbgv3UNdW": "Change all mismatching attributes toward a target profile",
    "R_7PPDn8J2Th04jwl": "Change the most influential attributes for the case",
    "R_7PppHnKsPWMhHua": "Change all mismatching attributes toward a target profile",
    "R_7zCOYjoB3D2Mqot": "No consistent rule or guessed changes",
    "R_17kEVqhzQ2qsxRj": "Change the most influential attributes for the case",
    "R_1iD8KQ66I3AExYs": "No consistent rule or guessed changes",
    "R_1tEUS0XkXyE4GbH": "Change the most influential attributes for the case",
    "R_3dxYvw8kqqJYsmL": "Change the most influential attributes for the case",
    "R_3u3Hr91OxYQ7j2h": "Change the most influential attributes for the case",
    "R_502VKXs2OjYkMTQ": "No consistent rule or guessed changes",
    "R_5PStwIHXgmZwa9J": "Change all mismatching attributes toward a target profile",
}


FORWARD_MAP = {
    "best forward mental model": "best_forward_mental_model",
    "supported forward mental models (delta BIC < 2)": "supported_forward_mental_models_delta_bic_lt_2",
    "forward mental model fit conclusion": "forward_mental_model_fit_conclusion",
    "forward mental model delta BIC to second": "delta_bic_to_second_best",
    "exemplar forward NLL": "exemplar_nll",
    "attribution forward NLL": "attribution_nll",
    "prototype forward NLL": "prototype_nll",
}

COUNTERFACTUAL_MAP = {
    "best counterfactual strategy": "best_counterfactual_strategy",
    "minimum-loss counterfactual strategy": "minimum_loss_counterfactual_strategy",
    "selected strategy fitted parameter count": "selected_strategy_fitted_parameter_count",
    "supported counterfactual strategies (one SE)": "supported_counterfactual_strategies_one_se",
    "counterfactual strategy fit conclusion": "counterfactual_strategy_fit_conclusion",
    "counterfactual-implied mental model": "counterfactual_implied_mental_model",
    "supported counterfactual-implied mental models": "supported_counterfactual_implied_mental_models",
    "counterfactual strategy CV normalized L1": "best_strategy_cv_mean_normalized_l1",
    "counterfactual strategy feature-selection F1": "best_strategy_cv_feature_selection_f1",
    "counterfactual strategy direction accuracy": "best_strategy_cv_direction_accuracy",
    "counterfactual strategy gap to second": "counterfactual_strategy_gap_to_second_best",
    "counterfactual strategy fitted k": "best_strategy_full_data_k",
    "counterfactual strategy fixed amount": "best_strategy_full_data_fixed_amount",
}


def participant_fit(path: Path) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    return {row["participant"]: row for row in rows}


def testing_feature_count(row: dict[str, str]) -> int:
    return sum(int(round(number(row[f"x_{index}_changed"]) or 0)) for index in range(1, 6))


def assign_new_clusters(old_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> dict[str, int]:
    old_testing = [
        row for row in old_rows
        if row["domain"].casefold() == "diabetes" and row["phase"].casefold() == "testing"
    ]
    new_testing = [row for row in new_rows if row["phase"].casefold() == "testing"]
    instances = sorted({int(row["instance id"]) for row in old_testing})

    def matrices(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray]:
        grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            grouped[row["participant"]][int(row["instance id"])] = row
        participants = sorted(grouped)
        values = []
        for participant in participants:
            if set(grouped[participant]) != set(instances):
                raise ValueError(f"Participant {participant} lacks the matched diabetes instances")
            counts = [testing_feature_count(grouped[participant][instance]) for instance in instances]
            proximity = [float(grouped[participant][instance]["proximity"]) for instance in instances]
            values.append(counts + proximity)
        return participants, np.asarray(values, dtype=float)

    old_participants, old_matrix = matrices(old_testing)
    new_participants, new_matrix = matrices(new_testing)
    labels_by_participant = {
        row["participant"]: int(float(row["participant change cluster"]))
        for row in old_testing
    }
    labels = np.asarray([labels_by_participant[p] for p in old_participants])
    mean = old_matrix.mean(axis=0)
    scale = old_matrix.std(axis=0)
    scale[scale == 0] = 1.0
    old_z = (old_matrix - mean) / scale
    new_z = (new_matrix - mean) / scale
    centroids = {label: old_z[labels == label].mean(axis=0) for label in sorted(set(labels))}
    assigned = {}
    for participant, vector in zip(new_participants, new_z):
        assigned[participant] = min(
            centroids,
            key=lambda label: float(np.square(vector - centroids[label]).sum()),
        )
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--new-base", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--forward-fits", type=Path, required=True)
    parser.add_argument("--counterfactual-fits", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "static" / "experiment-data.json")
    args = parser.parse_args()

    columns, old_rows = read_csv(args.existing)
    _, base_rows = read_csv(args.new_base)
    _, boundary_rows = read_csv(args.boundary)
    if len(base_rows) != len(boundary_rows):
        raise ValueError("Boundary output does not align with new base rows")
    if "CRT-2 score (0-4)" not in columns:
        columns.insert(columns.index("participant training accuracy") + 1, "CRT-2 score (0-4)")

    converted_participants = {row["participant"] for row in base_rows}
    raw_by_participant = {
        row["ResponseId"]: row
        for row in read_qualtrics(args.raw)
        if row.get("ResponseId") in converted_participants
    }
    crt_by_participant = {
        participant: score_crt(response)
        for participant, response in raw_by_participant.items()
    }
    forward = participant_fit(args.forward_fits)
    counterfactual = participant_fit(args.counterfactual_fits)

    category_by_instance: dict[tuple[str, int], tuple[str, str]] = {}
    category_votes: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for row in old_rows:
        if row["phase"].casefold() == "testing":
            category_votes[(row["domain"].casefold(), int(row["instance id"]))].append(
                (row["close to boundary old"], row["distance to boundary"])
            )
    for key, votes in category_votes.items():
        category_by_instance[key] = Counter(votes).most_common(1)[0][0]

    new_rows: list[dict[str, str]] = []
    training_correct: dict[str, list[int]] = defaultdict(list)
    for row in base_rows:
        if row["phase"].casefold() == "training":
            training_correct[row["participant"]].append(int(float(row["training correct (0/1)"])))

    for index, base in enumerate(base_rows):
        row = {column: "" for column in columns}
        for name in (
            "participant", "domain", "xai", "phase", "original label", "target label",
            "case", "instance id", "response time (seconds)",
            "attribute values before and after", "explanation", "counterfactual label",
            "distance to closest minimal counterfactual", "confidence for target label original",
            "confidence for target label counterfactual", "delta confidence of target label",
            "training response", "training correct (0/1)",
        ):
            row[name] = base.get(name, "")
        for feature in range(1, 6):
            row[f"x_{feature}_change"] = base[f"x_{feature}_change"]
            row[f"x_{feature}_changed"] = base[f"x_{feature}_changed"]

        participant = row["participant"]
        row["participant training accuracy"] = scalar(np.mean(training_correct[participant]))
        row["CRT-2 score (0-4)"] = str(crt_by_participant[participant])
        row["reasoning strategy"] = REASONING_STRATEGY[participant]
        for output_name, input_name in FORWARD_MAP.items():
            row[output_name] = forward[participant].get(input_name, "")
        for output_name, input_name in COUNTERFACTUAL_MAP.items():
            row[output_name] = counterfactual[participant].get(input_name, "")

        if row["phase"].casefold() == "testing":
            count = int(float(base["num attributes changed"]))
            proximity = float(base["distance of counterfactual to original"])
            valid = int(float(base["valid counterfactual (0/1)"]))
            delta = float(base["delta confidence of target label"])
            edited_confidence = float(base["confidence for target label counterfactual"])
            row["sparsity"] = scalar(1.0 - count / 5.0)
            row["proximity"] = scalar(proximity)
            row["successful counterfactual (0/1)"] = str(valid)
            row["valid counterfactual (binary continuous)"] = str(valid)
            row["move towards target (0/1)"] = str(int(delta > 0))
            row["delta confidence/change amount"] = scalar(delta / proximity if proximity > 0 else None)
            row["confidence of counterfactual - 50%"] = scalar(abs(edited_confidence - 0.5))
            domain_instance = (row["domain"].casefold(), int(row["instance id"]))
            if domain_instance not in category_by_instance:
                raise ValueError(f"No old boundary category for {domain_instance}")
            row["close to boundary old"], row["distance to boundary"] = category_by_instance[domain_instance]
            row["actionability (0/1)"] = str(int(not (row["domain"] == "diabetes" and int(float(row["x_5_changed"])) == 1)))
            boundary = boundary_rows[index]
            row["boundary distance original"] = boundary["boundary distance original"]
            row["boundary distance new"] = boundary["boundary distance new"]
            row["boundary distance change (new - original)"] = boundary["boundary distance change (new - original)"]
        new_rows.append(row)

    cluster_by_participant = assign_new_clusters(old_rows, new_rows)
    for row in new_rows:
        cluster = cluster_by_participant[row["participant"]]
        row["participant change cluster"] = str(cluster)
        row["participant change strategy"] = (
            "Focused / smaller changes" if cluster == 1 else "Broad / larger changes"
        )

    matrices = reference_matrices(args.bundle)
    all_rows = old_rows + new_rows
    for row in all_rows:
        if row["phase"].casefold() == "training":
            row["sparsity"] = ""
            row["plausibility"] = ""
        else:
            count = testing_feature_count(row)
            row["sparsity"] = scalar(1.0 - count / 5.0)
            profile = normalized_profile(row["attribute values before and after"])
            # Gower distance is the mean of the five per-feature dissimilarities.
            nearest_gower = float(np.abs(matrices[row["domain"].casefold()] - profile).mean(axis=1).min())
            row["plausibility"] = scalar(min(1.0, max(0.0, 1.0 - nearest_gower)))
        if row["participant"] not in crt_by_participant:
            row["CRT-2 score (0-4)"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in all_rows)

    participants = {row["participant"] for row in all_rows}
    new_participants = {row["participant"] for row in new_rows}
    testing = [row for row in all_rows if row["phase"].casefold() == "testing"]
    summary = {
        "rows": len(all_rows),
        "columns": len(columns),
        "participants": len(participants),
        "new_rows": len(new_rows),
        "new_participants": len(new_participants),
        "crt_distribution_new_participants": dict(sorted(Counter(crt_by_participant.values()).items())),
        "sparsity_definition": "1 - number of changed features / 5; higher is sparser",
        "sparsity_range_testing": [min(float(row["sparsity"]) for row in testing), max(float(row["sparsity"]) for row in testing)],
        "plausibility_definition": "1 - Gower distance to nearest domain training record (k=1); higher is more plausible",
        "plausibility_range_testing": [min(float(row["plausibility"]) for row in testing), max(float(row["plausibility"]) for row in testing)],
        "new_reasoning_strategies": dict(sorted(Counter(REASONING_STRATEGY.values()).items())),
        "new_clusters": {
            int(label): int(count)
            for label, count in sorted(Counter(cluster_by_participant.values()).items())
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
