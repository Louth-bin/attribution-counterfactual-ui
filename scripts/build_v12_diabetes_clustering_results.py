"""Assemble the new-only diabetes-clustering results in the v1.1 column schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
NORMALIZED_VALUE_RE = re.compile(
    r"\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)

# Conservative five-category coding based jointly on the edit fits and the two
# post-task free-text responses. It is intentionally separate from the formal
# attribute-selection and amount-model columns.
REASONING_STRATEGY = {
    "R_15E5kn5aqWV9Myt": "Always change the same attributes regardless of the case",
    "R_1eVmoOsuh2kQnVU": "Change all mismatching attributes toward a target profile",
    "R_1VV8BqOrYdNpSHT": "Change all mismatching attributes toward a target profile",
    "R_3xBI9TIaQaSsWad": "Change the most out-of-range attributes for the case",
    "R_53vXmt7xog3ij55": "Change the most out-of-range attributes for the case",
    "R_56kBaagYoPMItRS": "Change the most out-of-range attributes for the case",
    "R_5Rv44YPGvhr8aCR": "Change all mismatching attributes toward a target profile",
    "R_6haS2ptrunWMddu": "Change the most out-of-range attributes for the case",
    "R_7nq1k9p3nFmxZjK": "Change all mismatching attributes toward a target profile",
    "R_3fJe3EZDB3jrLK9": "No consistent rule or guessed changes",
    "R_3gwJ7NCYNcCcCho": "No consistent rule or guessed changes",
    "R_3jfNL25GoofAHh7": "Always change the same attributes regardless of the case",
    "R_5AdxFNYPUVXDGcV": "Change the most influential attributes for the case",
    "R_5yl16n60oJpwrWu": "Change all mismatching attributes toward a target profile",
    "R_6iqiXGwFokBZXvb": "Change the most influential attributes for the case",
    "R_7dM8goblPKNq3uj": "Change the most out-of-range attributes for the case",
    "R_7PvOmZWmKNN4iL0": "Always change the same attributes regardless of the case",
    "R_3KdocVAIKFJHFu0": "Change the most out-of-range attributes for the case",
    "R_3qWWvSHayKcFHto": "Always change the same attributes regardless of the case",
    "R_3WOmvFM0WJMKuPa": "No consistent rule or guessed changes",
    "R_50bx2EyN3nLlK5b": "Change the most influential attributes for the case",
    "R_5w6IMky8ndKIb5h": "Change the most out-of-range attributes for the case",
    "R_69tQCGOCnlsf1C1": "Change the most out-of-range attributes for the case",
    "R_6a1D770kyKVlljb": "Change all mismatching attributes toward a target profile",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def read_qualtrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[3:]]


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def numeric(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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


def normalize_value(value: object, kind: str, levels: list[object]) -> float:
    if kind == "categorical":
        categories = [str(item).casefold() for item in levels]
        index = categories.index(str(value).casefold())
        return index / (len(categories) - 1) if len(categories) > 1 else 0.0
    low, high = float(levels[0]), float(levels[1])
    if math.isclose(low, high):
        return 0.0
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def plausibility_references(bundle_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    diabetes = bundle["datasets"]["diabetes"]
    case = diabetes["test_pool"][0]
    kinds = list(case["feature_types"])
    ranges = list(case["raw_feature_ranges"])

    _, training_records = read_csv(ROOT / "src" / "data" / "diabetes" / "train.csv")
    full = np.asarray(
        [
            [
                normalize_value(record[name], kinds[index], ranges[index])
                for index, name in enumerate(FEATURES)
            ]
            for record in training_records
        ],
        dtype=float,
    )
    selected_cases = list(diabetes["training_pool"])
    if len(selected_cases) != 12:
        raise ValueError(f"Expected 12 clustered training profiles, found {len(selected_cases)}")
    selected = np.asarray(
        [
            [
                normalize_value(value, kind, list(levels))
                for value, kind, levels in zip(
                    item["raw_feature_values"],
                    item["feature_types"],
                    item["raw_feature_ranges"],
                )
            ]
            for item in selected_cases
        ],
        dtype=float,
    )
    return full, selected


def score_crt(response: dict[str, str]) -> int:
    def clean(name: str) -> str:
        return " ".join(response.get(name, "").casefold().strip().split())

    race = clean("CRT2_Race")
    sheep = clean("CRT2_Sheep")
    emily = clean("CRT2_Emily")
    hole = clean("CRT2_Hole")
    return sum(
        (
            "second" in race or bool(re.search(r"\b2(?:nd)?\b", race)),
            "eight" in sheep or bool(re.search(r"\b8\b", sheep)),
            "emily" in emily,
            "no dirt" in hole
            or "none" in hole
            or "nothing" in hole
            or "zero" in hole
            or bool(re.search(r"\b0(?:\.0+)?\b", hole)),
        )
    )


def indexed(path: Path) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    return {row["participant"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--forward", type=Path)
    parser.add_argument("--strategy", type=Path)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--reasoning-json",
        type=Path,
        help="Optional participant-to-reasoning-strategy JSON mapping",
    )
    parser.add_argument(
        "--continuous-validity-column",
        default="valid counterfactual (binary continuous)",
        help="Output name for the continuous copy of counterfactual validity",
    )
    parser.add_argument(
        "--skip-strategies",
        action="store_true",
        help="Leave qualitative and fitted strategy columns blank.",
    )
    args = parser.parse_args()

    reference_columns, _ = read_csv(args.reference_csv)
    reference_columns = [
        args.continuous_validity_column
        if column == "valid counterfactual (binary continuous)"
        else column
        for column in reference_columns
    ]
    _, base_rows = read_csv(args.base)
    _, boundary_rows = read_csv(args.boundary)
    if len(base_rows) != len(boundary_rows):
        raise ValueError("Boundary rows do not align with converted rows")

    participants = sorted({row["participant"] for row in base_rows})
    reasoning_strategy = {}
    if not args.skip_strategies:
        reasoning_strategy = REASONING_STRATEGY
        if args.reasoning_json:
            reasoning_strategy = json.loads(args.reasoning_json.read_text(encoding="utf-8"))
        if set(participants) != set(reasoning_strategy):
            raise ValueError("Reasoning-strategy coding does not match converted participants")

    raw_by_participant = {
        row["ResponseId"]: row
        for row in read_qualtrics(args.raw)
        if row.get("ResponseId") in participants
    }
    crt = {participant: score_crt(raw_by_participant[participant]) for participant in participants}
    forward = indexed(args.forward) if not args.skip_strategies else {}
    strategy = indexed(args.strategy) if not args.skip_strategies else {}
    full_reference, subset_reference = plausibility_references(args.bundle)

    training_correct: dict[str, list[int]] = defaultdict(list)
    for row in base_rows:
        if row["phase"] == "training":
            training_correct[row["participant"]].append(int(float(row["training correct (0/1)"])))
    training_accuracy = {
        participant: float(np.mean(values)) for participant, values in training_correct.items()
    }

    testing_confidences = np.asarray(
        [float(row["confidence for target label original"]) for row in base_rows if row["phase"] == "testing"],
        dtype=float,
    )
    boundary_quantile = float(np.quantile(testing_confidences, 0.8))

    output_rows: list[dict[str, str]] = []
    for index, base in enumerate(base_rows):
        row = {column: "" for column in reference_columns}
        participant = base["participant"]
        phase = base["phase"]
        for target, source in (
            ("participant", "participant"),
            ("domain", "domain"),
            ("xai", "xai"),
            ("phase", "phase"),
            ("original label", "original label"),
            ("target label", "target label"),
            ("case", "case"),
            ("instance id", "instance id"),
            ("response time (seconds)", "response time (seconds)"),
            ("attribute values before and after", "attribute values before and after"),
            ("explanation", "explanation"),
            ("counterfactual label", "counterfactual label"),
            ("distance to closest minimal counterfactual", "distance to closest minimal counterfactual"),
            ("confidence for target label original", "confidence for target label original"),
            ("confidence for target label counterfactual", "confidence for target label counterfactual"),
            ("delta confidence of target label", "delta confidence of target label"),
            ("training response", "training response"),
            ("training correct (0/1)", "training correct (0/1)"),
        ):
            row[target] = base[source]
        for feature in range(1, 6):
            row[f"x_{feature}_change"] = base[f"x_{feature}_change"]
            row[f"x_{feature}_changed"] = base[f"x_{feature}_changed"]

        row["reasoning strategy"] = reasoning_strategy.get(participant, "")
        row["participant training accuracy"] = scalar(training_accuracy[participant])
        row["CRT-2 score (0-4)"] = str(crt[participant])
        row["CRT binarized"] = str(int(crt[participant] > 2))
        if not args.skip_strategies:
            row["best forward mental model"] = forward[participant]["best_forward_mental_model"]
            row["forward mental model fit conclusion"] = forward[participant]["forward_mental_model_fit_conclusion"]
            row["attribute selection method"] = strategy[participant]["attribute selection method"]
            row["attribute selection ranking score"] = strategy[participant]["attribute selection ranking score"]
            row["change amount method"] = strategy[participant]["change amount method"]
            row["change amount normalized MAE"] = strategy[participant]["change amount normalized MAE"]
            row["best-supported attribute selection method"] = strategy[participant]["best-supported attribute selection method"]
            row["strategy: xai"] = row["attribute selection method"] + " : " + row["xai"]

        if phase == "testing":
            changed_count = sum(int(float(base[f"x_{feature}_changed"])) for feature in range(1, 6))
            proximity = float(base["distance of counterfactual to original"])
            valid = int(float(base["valid counterfactual (0/1)"]))
            delta = float(base["delta confidence of target label"])
            edited_confidence = float(base["confidence for target label counterfactual"])
            original_confidence = float(base["confidence for target label original"])
            profile = normalized_profile(base["attribute values before and after"])

            row["sparsity"] = scalar(1.0 - changed_count / 5.0)
            row["proximity"] = scalar(proximity)
            row["successful counterfactual (0/1)"] = str(valid)
            row["move towards target (0/1)"] = str(int(delta > 0))
            row["delta confidence/change amount"] = scalar(delta / proximity if proximity > 0 else None)
            row["close to boundary old"] = str(int(original_confidence > boundary_quantile))
            row["distance to boundary"] = "near boundary" if original_confidence > boundary_quantile else "far from boundary"
            row["confidence of counterfactual - 50%"] = scalar(abs(edited_confidence - 0.5))
            row[args.continuous_validity_column] = str(valid)
            row["actionability (0/1)"] = str(int(int(float(base["x_5_changed"])) == 0))
            full_distance = float(np.abs(full_reference - profile).mean(axis=1).min())
            subset_distance = float(np.abs(subset_reference - profile).mean(axis=1).min())
            row["plausibility"] = scalar(1.0 - full_distance)
            row["plausibility (subset)"] = scalar(1.0 - subset_distance)
            for name in (
                "boundary distance original",
                "boundary distance new",
                "boundary distance change (new - original)",
            ):
                row[name] = boundary_rows[index][name]
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=reference_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    testing_rows = [row for row in output_rows if row["phase"] == "testing"]
    summary = {
        "rows": len(output_rows),
        "columns": len(reference_columns),
        "participants": len(participants),
        "training_rows": len(output_rows) - len(testing_rows),
        "testing_rows": len(testing_rows),
        "xai_participants": {
            xai: len({row["participant"] for row in output_rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in output_rows})
        },
        "valid_counterfactuals": sum(int(row["successful counterfactual (0/1)"]) for row in testing_rows),
        "training_correct": sum(int(float(row["training correct (0/1)"])) for row in output_rows if row["phase"] == "training"),
        "subset_reference_profiles": 12,
        "reasoning_strategy_counts": {
            name: list(reasoning_strategy.values()).count(name)
            for name in sorted(set(reasoning_strategy.values()))
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
