"""Add target-directed boundary progress to the cleaned Qualtrics results.

The existing ``margin_pp`` is the edited instance's target-class probability
minus the 50% decision boundary, expressed in percentage points.  The added
``boundary_progress_pp`` is:

    edited target probability - original target probability

also in percentage points.  It is positive for movement toward the requested
class and negative for movement away from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "qualtrics" / "qualtrics_results_v0.2.csv"
DEFAULT_RAW = ROOT / "qualtrics" / "raw_output_v0.3.csv"
DEFAULT_EXPERIMENT_DATA = ROOT / "static" / "experiment-data.js"
NEW_COLUMN = "boundary_progress_pp"


def load_experiment_data(path: Path) -> Mapping[str, Any]:
    source = path.read_text(encoding="utf-8").strip()
    prefix = "window.EXPERIMENT_DATA = "
    if not source.startswith(prefix) or not source.endswith(";"):
        raise ValueError(f"Unexpected experiment-data wrapper in {path}")
    return json.loads(source[len(prefix) : -1])


def ordered_values(model: Mapping[str, Any], values: Any) -> List[Any]:
    if isinstance(values, Mapping):
        return [values[name] for name in model["feature_names"]]
    result = list(values)
    if len(result) != len(model["feature_names"]):
        raise ValueError(
            f"Expected {len(model['feature_names'])} values, got {len(result)}"
        )
    return result


def preprocess(model: Mapping[str, Any], raw_values: Sequence[Any]) -> List[float]:
    settings = model["preprocessing"]
    if settings["type"] == "standard-scaler":
        return [
            (float(value) - float(mean)) / float(scale)
            for value, mean, scale in zip(
                raw_values, settings["mean"], settings["scale"]
            )
        ]
    if settings["type"] != "column-transformer-v1":
        raise ValueError(f"Unsupported preprocessing type: {settings['type']}")

    by_name = dict(zip(model["feature_names"], raw_values))
    numeric = settings.get("numeric", {})
    transformed = [
        (float(by_name[name]) - float(mean)) / float(scale)
        for name, mean, scale in zip(
            numeric.get("feature_names", []),
            numeric.get("mean", []),
            numeric.get("scale", []),
        )
    ]
    categorical = settings.get("categorical", {})
    for name, categories in zip(
        categorical.get("feature_names", []),
        categorical.get("categories", []),
    ):
        value = str(by_name[name])
        transformed.extend(1.0 if str(category) == value else 0.0 for category in categories)
    return transformed


def predict_positive_probability(model: Mapping[str, Any], values: Any) -> float:
    layer_values = preprocess(model, ordered_values(model, values))
    layers = model["layers"]
    for layer_index, layer in enumerate(layers):
        outputs = []
        for output_index, bias in enumerate(layer["biases"]):
            output = float(bias) + sum(
                value * float(layer["weights"][input_index][output_index])
                for input_index, value in enumerate(layer_values)
            )
            outputs.append(output)
        if layer_index < len(layers) - 1:
            if model["hidden_activation"] != "relu":
                raise ValueError(
                    f"Unsupported hidden activation: {model['hidden_activation']}"
                )
            outputs = [max(0.0, output) for output in outputs]
        layer_values = outputs

    if len(layer_values) != 1 or model["output_activation"] != "logistic":
        raise ValueError("Expected one logistic output")
    logit = layer_values[0]
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def target_probability(positive_probability: float, target_class: int) -> float:
    if target_class == 1:
        return positive_probability
    if target_class == 0:
        return 1.0 - positive_probability
    raise ValueError(f"Expected binary target class, got {target_class}")


def load_testing_logs(path: Path) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    records: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for response in csv.DictReader(handle):
            participant = response.get("ResponseId", "")
            raw_log = response.get("testing_log_json", "")
            if not participant.startswith("R_") or not raw_log.startswith("["):
                continue
            for record in json.loads(raw_log):
                key = (participant, record["domain"], int(record["instanceId"]))
                if key in records:
                    raise ValueError(f"Duplicate testing record: {key}")
                records[key] = record
    return records


def load_original_instances(
    experiment_data: Mapping[str, Any]
) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    instances: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for domain, dataset in experiment_data["datasets"].items():
        for instance in dataset["test_pool"]:
            key = (domain, int(instance["instance_id"]))
            if key in instances:
                raise ValueError(f"Duplicate test instance: {key}")
            instances[key] = instance
    return instances


def add_boundary_progress(
    results_path: Path, raw_path: Path, experiment_data_path: Path
) -> int:
    experiment_data = load_experiment_data(experiment_data_path)
    logs = load_testing_logs(raw_path)
    originals = load_original_instances(experiment_data)

    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "margin_pp" not in fieldnames:
        raise ValueError(f"margin_pp is missing from {results_path}")
    if NEW_COLUMN not in fieldnames:
        fieldnames.insert(fieldnames.index("margin_pp") + 1, NEW_COLUMN)

    for row in rows:
        participant = row["participant"]
        domain = row["domain"]
        instance_id = int(row["instance"])
        log_key = (participant, domain, instance_id)
        instance_key = (domain, instance_id)
        if log_key not in logs:
            raise ValueError(f"No raw testing record for {log_key}")
        if instance_key not in originals:
            raise ValueError(f"No original test instance for {instance_key}")

        record = logs[log_key]
        original = originals[instance_key]
        model = experiment_data["datasets"][domain]["browser_model"]
        original_class = int(record["originalPrediction"]["value"])
        target_class = 1 - original_class

        original_positive = predict_positive_probability(
            model, original["raw_feature_values"]
        )
        edited_positive = predict_positive_probability(
            model, record["changedRawFeatureValues"]
        )
        original_target = target_probability(original_positive, target_class)
        edited_target = target_probability(edited_positive, target_class)
        recomputed_margin = 100.0 * (edited_target - 0.5)
        recorded_margin = float(row["margin_pp"])
        if not math.isclose(recomputed_margin, recorded_margin, abs_tol=0.00005):
            raise ValueError(
                f"margin_pp mismatch for {log_key}: "
                f"stored={recorded_margin}, recomputed={recomputed_margin:.4f}"
            )

        progress = 100.0 * (edited_target - original_target)
        row[NEW_COLUMN] = f"{progress:.4f}"

    temp_path = results_path.with_name(f".{results_path.name}.tmp")
    try:
        with temp_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, results_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--experiment-data", type=Path, default=DEFAULT_EXPERIMENT_DATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count = add_boundary_progress(args.results, args.raw, args.experiment_data)
    print(f"Added {NEW_COLUMN} to {row_count} rows in {args.results}")


if __name__ == "__main__":
    main()
