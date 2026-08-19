"""Convert Qualtrics training and testing logs to one row per case.

Distances use normalized L1 (the sum of absolute per-feature changes after
normalizing numerical features by their configured range and categorical
features by their ordered category index).  The closest-minimal-CF column is
the distance from the participant's edited profile to the algorithmically
optimized minimal counterfactual bundled for that same instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    REPO_ROOT / "qualtrics" / "raw_output_v0.3.csv",
    REPO_ROOT / "qualtrics" / "raw_output_v0.5.csv",
    REPO_ROOT / "qualtrics" / "raw_output_v0.6.csv",
)
DEFAULT_STATIC_DATA = REPO_ROOT / "static" / "experiment-data.json"
DEFAULT_OUTPUT = REPO_ROOT / "qualtrics" / "qualtrics_results_v0.4.csv"

OUTPUT_COLUMNS = (
    "participant",
    "domain",
    "xai",
    "phase",
    "original label",
    "target label",
    "case",
    "instance id",
    "response time (seconds)",
    "attribute values before and after",
    "explanation",
    "num attributes changed",
    "distance of counterfactual to original",
    "counterfactual label",
    "valid counterfactual (0/1)",
    "distance to closest minimal counterfactual",
    "confidence for target label original",
    "confidence for target label counterfactual",
    "delta confidence of target label",
    "training response",
    "training correct (0/1)",
)


def load_qualtrics_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    if len(rows) < 3:
        raise ValueError(f"{path} does not contain the three Qualtrics header rows")
    headers = rows[0]
    if len(headers) != len(set(headers)):
        raise ValueError(f"{path} contains duplicate CSV column names")
    data_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows[3:], start=4):
        if len(row) != len(headers):
            raise ValueError(
                f"{path}:{row_number} has {len(row)} fields; expected {len(headers)}"
            )
        data_rows.append(dict(zip(headers, row)))
    return data_rows


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def predict_binary(
    model: dict[str, Any], raw_feature_values: dict[str, Any] | list[Any]
) -> tuple[int, str, dict[int, float]]:
    if isinstance(raw_feature_values, dict):
        raw_values = [raw_feature_values[name] for name in model["feature_names"]]
    else:
        raw_values = list(raw_feature_values)

    preprocessing = model["preprocessing"]
    if preprocessing["type"] == "standard-scaler":
        values = [
            (float(value) - float(preprocessing["mean"][index]))
            / float(preprocessing["scale"][index])
            for index, value in enumerate(raw_values)
        ]
    elif preprocessing["type"] == "column-transformer-v1":
        value_by_name = dict(zip(model["feature_names"], raw_values))
        numeric = preprocessing.get("numeric", {})
        values = [
            (float(value_by_name[name]) - float(numeric["mean"][index]))
            / float(numeric["scale"][index])
            for index, name in enumerate(numeric.get("feature_names", []))
        ]
        categorical = preprocessing.get("categorical", {})
        for feature_index, name in enumerate(categorical.get("feature_names", [])):
            value = str(value_by_name[name])
            for category in categorical.get("categories", [])[feature_index]:
                values.append(1.0 if str(category) == value else 0.0)
    else:
        raise ValueError(f"Unsupported preprocessing {preprocessing['type']!r}")

    layers = model["layers"]
    for layer_index, layer in enumerate(layers):
        outputs = []
        for output_index, bias in enumerate(layer["biases"]):
            result = float(bias) + sum(
                value * float(layer["weights"][input_index][output_index])
                for input_index, value in enumerate(values)
            )
            if layer_index < len(layers) - 1:
                result = max(0.0, result)
            outputs.append(result)
        values = outputs

    positive_probability = stable_sigmoid(values[0])
    classes = [int(value) for value in model["classes"]]
    probabilities = {
        classes[0]: 1.0 - positive_probability,
        classes[1]: positive_probability,
    }
    prediction = classes[1] if positive_probability > 0.5 else classes[0]
    label_index = classes.index(prediction)
    return prediction, str(model["class_labels"][label_index]), probabilities


def category_index(value: Any, categories: list[Any]) -> int:
    normalized = str(value).casefold()
    for index, category in enumerate(categories):
        if str(category).casefold() == normalized:
            return index
    raise ValueError(f"Category {value!r} is not in {categories!r}")


def normalized_value(
    value: Any, feature_type: str, feature_range: list[Any]
) -> float:
    if feature_type == "categorical":
        if len(feature_range) <= 1:
            return 0.0
        return category_index(value, feature_range) / (len(feature_range) - 1)
    low, high = (float(feature_range[0]), float(feature_range[1]))
    span = high - low
    if abs(span) <= 1e-12:
        return 0.0
    normalized = (float(value) - low) / span
    return min(1.0, max(0.0, normalized))


def normalized_l1(
    first: Iterable[Any],
    second: Iterable[Any],
    feature_types: list[str],
    feature_ranges: list[list[Any]],
) -> float:
    return sum(
        abs(
            normalized_value(first_value, feature_type, feature_range)
            - normalized_value(second_value, feature_type, feature_range)
        )
        for first_value, second_value, feature_type, feature_range in zip(
            first, second, feature_types, feature_ranges
        )
    )


def values_differ(first: Any, second: Any, feature_type: str) -> bool:
    if feature_type == "categorical":
        return str(first).casefold() != str(second).casefold()
    return abs(float(first) - float(second)) > 1e-9


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot write non-finite number {value}")
        return f"{value:.12g}"
    return str(value)


def display_value(
    raw_value: Any,
    feature_type: str,
    raw_range: list[Any],
    display_range: list[Any],
) -> Any:
    if feature_type != "categorical":
        return raw_value
    return display_range[category_index(raw_value, raw_range)]


def format_attributes(
    case: dict[str, Any], original: list[Any], changed: list[Any]
) -> str:
    lines = []
    for index, name in enumerate(case["feature_names"]):
        feature_type = case["feature_types"][index]
        raw_range = case["raw_feature_ranges"][index]
        feature_range = case["feature_ranges"][index]
        before = display_value(original[index], feature_type, raw_range, feature_range)
        after = display_value(changed[index], feature_type, raw_range, feature_range)
        before_normalized = normalized_value(
            original[index], feature_type, raw_range
        )
        before_text = (
            f"{format_scalar(before)}({format_scalar(before_normalized)})"
        )
        if values_differ(original[index], changed[index], feature_type):
            after_normalized = normalized_value(
                changed[index], feature_type, raw_range
            )
            value_text = (
                f"{before_text} -> "
                f"{format_scalar(after)}({format_scalar(after_normalized)})"
            )
        else:
            value_text = before_text
        lines.append(f'"{name}" - {value_text}')
    return "\n".join(lines)


def format_attribution_explanation(case: dict[str, Any]) -> str:
    attribution = case["attribution"]
    values = attribution["values"]
    shown_indices = list(attribution["shown_feature_indices"][:2])
    total = sum(abs(float(value)) for value in values) or 1.0
    direction_labels = attribution["direction_labels"]
    lines = []
    for index in shown_indices:
        value = float(values[index])
        sign = "+" if value >= 0 else "-"
        percentage = round((abs(value) / total) * 100)
        direction = (
            direction_labels["right"] if value >= 0 else direction_labels["left"]
        )
        lines.append(
            f'"{case["feature_names"][index]}" - '
            f"{sign}{percentage}% toward {direction}"
        )
    return "\n".join(lines)


def format_explanation(case: dict[str, Any], xai: str, phase: str) -> str:
    if phase != "training" or xai == "none":
        return ""
    if xai == "attribution":
        return format_attribution_explanation(case)
    if xai == "counterfactual":
        return format_attributes(
            case,
            list(case["raw_feature_values"]),
            list(case["counterfactual"]["raw_feature_values"]),
        )
    raise ValueError(f"Unsupported explanation condition {xai!r}")


def ordered_raw_values(case: dict[str, Any], values: dict[str, Any]) -> list[Any]:
    missing = set(case["raw_feature_names"]) - set(values)
    extra = set(values) - set(case["raw_feature_names"])
    if missing or extra:
        raise ValueError(
            f"Instance {case['instance_id']} edited-value keys differ: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [values[name] for name in case["raw_feature_names"]]


def convert(
    input_paths: list[Path], static_data_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    static_data = json.loads(static_data_path.read_text(encoding="utf-8"))
    datasets = static_data["datasets"]
    case_index = {
        (domain, phase, int(case["instance_id"])): case
        for domain, bundle in datasets.items()
        for phase, pool_name in (("training", "training_pool"), ("testing", "test_pool"))
        for case in bundle[pool_name]
    }

    output_rows: list[dict[str, Any]] = []
    input_response_count = 0
    duplicate_response_count = 0
    skipped_response_count = 0
    seen_responses: dict[str, tuple[str, str, str]] = {}
    source_counts: dict[str, int] = {}
    phase_counts = {"training": 0, "testing": 0}
    for source_index, input_path in enumerate(input_paths):
        source_row_count = 0
        for response_index, response in enumerate(load_qualtrics_rows(input_path)):
            input_response_count += 1
            serialized_training_logs = response.get("training_log_json", "").strip()
            serialized_logs = response.get("testing_log_json", "").strip()
            response_id = response["ResponseId"]
            response_signature = (
                serialized_training_logs,
                serialized_logs,
                response.get("xaiType", "").strip(),
            )
            if response_id in seen_responses:
                if seen_responses[response_id] != response_signature:
                    raise ValueError(
                        f"Response {response_id} occurs in multiple inputs with "
                        "different experiment data"
                    )
                duplicate_response_count += 1
                continue
            seen_responses[response_id] = response_signature
            training_logs = json.loads(serialized_training_logs or "[]")
            testing_logs = json.loads(serialized_logs or "[]")
            if not training_logs and not testing_logs:
                skipped_response_count += 1
                continue
            if training_logs and len(training_logs) != 10:
                raise ValueError(
                    f"Response {response['ResponseId']} has {len(training_logs)} training cases"
                )

            seen_training_numbers: set[int] = set()
            for log in training_logs:
                domain = str(log["domain"])
                instance_id = int(log["instanceId"])
                case = case_index.get((domain, "training", instance_id))
                if case is None:
                    raise ValueError(
                        f"No static training case for {domain} instance {instance_id}"
                    )
                case_number = int(log["caseNumber"])
                if case_number in seen_training_numbers:
                    raise ValueError(
                        f"Response {response['ResponseId']} repeats training case {case_number}"
                    )
                seen_training_numbers.add(case_number)

                original_value = int(case["prediction"]["value"])
                logged_original = int(log["correctPrediction"])
                if logged_original != original_value:
                    raise ValueError(
                        f"Logged/static training prediction mismatch for {domain} {instance_id}"
                    )
                selected_value = int(log["selectedPrediction"])
                calculated_correct = int(selected_value == original_value)
                if calculated_correct != int(bool(log["correct"])):
                    raise ValueError(
                        f"Incorrect training correctness flag for {domain} {instance_id}"
                    )
                original_raw = list(case["raw_feature_values"])
                labels = case["prediction_labels"]
                xai = str(log.get("explanation") or response["xaiType"])
                output_rows.append(
                    {
                        "_source_index": source_index,
                        "_response_index": response_index,
                        "_phase_index": 0,
                        "participant": response["ResponseId"],
                        "domain": domain,
                        "xai": xai,
                        "phase": "training",
                        "explanation": format_explanation(case, xai, "training"),
                        "original label": str(case["prediction"]["label"]),
                        "target label": "",
                        "case": case_number,
                        "instance id": instance_id,
                        "response time (seconds)": float(log["responseMs"]) / 1000.0,
                        "attribute values before and after": format_attributes(
                            case, original_raw, original_raw
                        ),
                        "num attributes changed": "",
                        "distance of counterfactual to original": "",
                        "counterfactual label": "",
                        "valid counterfactual (0/1)": "",
                        "distance to closest minimal counterfactual": "",
                        "confidence for target label original": "",
                        "confidence for target label counterfactual": "",
                        "delta confidence of target label": "",
                        "training response": str(labels[selected_value]),
                        "training correct (0/1)": calculated_correct,
                    }
                )
                source_row_count += 1
                phase_counts["training"] += 1

            if not testing_logs:
                continue
            if len(testing_logs) % 2:
                raise ValueError(
                    f"Response {response['ResponseId']} has an odd number of testing cases"
                )
            direction_size = len(testing_logs) // 2
            if direction_size not in {5, 10}:
                raise ValueError(
                    f"Response {response['ResponseId']} has unexpected direction size "
                    f"{direction_size}"
                )

            seen_case_numbers: set[int] = set()
            for log in testing_logs:
                domain = str(log["domain"])
                instance_id = int(log["instanceId"])
                case = case_index.get((domain, "testing", instance_id))
                if case is None:
                    raise ValueError(
                        f"No static test case for {domain} instance {instance_id}"
                    )

                original_value = int(case["prediction"]["value"])
                logged_original = int(log["originalPrediction"]["value"])
                if logged_original != original_value:
                    raise ValueError(
                        f"Logged/static prediction mismatch for {domain} {instance_id}"
                    )
                target_value = 1 - original_value
                target_label = str(case["prediction_labels"][target_value])
                case_number = int(log["caseNumberWithinDirection"]) + (
                    original_value * direction_size
                )
                if case_number in seen_case_numbers:
                    raise ValueError(
                        f"Response {response['ResponseId']} repeats case number {case_number}"
                    )
                seen_case_numbers.add(case_number)

                original_raw = list(case["raw_feature_values"])
                changed_raw = ordered_raw_values(
                    case, log["changedRawFeatureValues"]
                )
                algorithmic_raw = list(case["counterfactual"]["raw_feature_values"])
                changed_count = sum(
                    values_differ(before, after, feature_type)
                    for before, after, feature_type in zip(
                        original_raw, changed_raw, case["feature_types"]
                    )
                )
                if changed_count < 1:
                    raise ValueError(
                        f"Response {response['ResponseId']} case {case_number} has no edit"
                    )

                model = datasets[domain]["browser_model"]
                model_original, _, original_probabilities = predict_binary(
                    model, original_raw
                )
                if model_original != original_value:
                    raise ValueError(
                        f"Browser/static model mismatch for {domain} {instance_id}"
                    )
                counterfactual_value, counterfactual_label, counterfactual_probabilities = (
                    predict_binary(model, changed_raw)
                )
                algorithmic_value, _, _ = predict_binary(model, algorithmic_raw)
                if algorithmic_value != target_value:
                    raise ValueError(
                        f"Bundled minimal CF is invalid for {domain} {instance_id}"
                    )

                original_target_confidence = original_probabilities[target_value]
                counterfactual_target_confidence = counterfactual_probabilities[
                    target_value
                ]
                xai = str(log.get("explanation") or response["xaiType"])
                output_rows.append(
                    {
                        "_source_index": source_index,
                        "_response_index": response_index,
                        "_phase_index": 1,
                        "participant": response["ResponseId"],
                        "domain": domain,
                        "xai": xai,
                        "phase": "testing",
                        "explanation": format_explanation(case, xai, "testing"),
                        "original label": str(case["prediction"]["label"]),
                        "target label": target_label,
                        "case": case_number,
                        "instance id": instance_id,
                        "response time (seconds)": float(log["responseMs"]) / 1000.0,
                        "attribute values before and after": format_attributes(
                            case, original_raw, changed_raw
                        ),
                        "num attributes changed": changed_count,
                        "distance of counterfactual to original": normalized_l1(
                            original_raw,
                            changed_raw,
                            case["feature_types"],
                            case["raw_feature_ranges"],
                        ),
                        "counterfactual label": counterfactual_label,
                        "valid counterfactual (0/1)": int(
                            counterfactual_value == target_value
                        ),
                        "distance to closest minimal counterfactual": normalized_l1(
                            changed_raw,
                            algorithmic_raw,
                            case["feature_types"],
                            case["raw_feature_ranges"],
                        ),
                        "confidence for target label original": original_target_confidence,
                        "confidence for target label counterfactual": counterfactual_target_confidence,
                        "delta confidence of target label": (
                            counterfactual_target_confidence - original_target_confidence
                        ),
                        "training response": "",
                        "training correct (0/1)": "",
                    }
                )
                source_row_count += 1
                phase_counts["testing"] += 1
        source_counts[input_path.name] = source_row_count

    output_rows.sort(
        key=lambda row: (
            row["_source_index"],
            row["_response_index"],
            row["_phase_index"],
            row["case"],
        )
    )
    report = {
        "input_responses": input_response_count,
        "unique_input_responses": len(seen_responses),
        "duplicate_input_responses": duplicate_response_count,
        "responses_without_testing_logs": skipped_response_count,
        "converted_responses": len({row["participant"] for row in output_rows}),
        "output_rows": len(output_rows),
        "phase_rows": phase_counts,
        "source_rows": source_counts,
        "valid_counterfactuals": sum(
            int(row["valid counterfactual (0/1)"])
            for row in output_rows
            if row["phase"] == "testing"
        ),
        "correct_training_responses": sum(
            int(row["training correct (0/1)"])
            for row in output_rows
            if row["phase"] == "training"
        ),
    }
    return output_rows, report


def write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: format_scalar(row[column])
                    if isinstance(row[column], (int, float, bool))
                    else row[column]
                    for column in OUTPUT_COLUMNS
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--static-data", type=Path, default=DEFAULT_STATIC_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, report = convert(args.inputs, args.static_data)
    write_output(args.output, rows)
    print(json.dumps({"output": str(args.output), **report}, indent=2))


if __name__ == "__main__":
    main()
