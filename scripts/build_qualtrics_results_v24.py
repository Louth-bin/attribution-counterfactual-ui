"""Convert raw_output_v2.4 into a v2.3-compatible one-feature results table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"
RAW = QUALTRICS / "raw_output_v2.4.csv"
REFERENCE = QUALTRICS / "qualtrics_results_v2.3.csv"
BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.7_actionable.json"
OUTPUT = QUALTRICS / "qualtrics_results_v2.4.csv"
AUDIT = QUALTRICS / "qualtrics_results_v2.4.audit.json"
BASE_CONVERTER = ROOT / "scripts" / "convert_recourse_v15_to_v20_prefix.py"

FEATURES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
FEATURE_SLUGS = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
TEST_CHANGE_COLUMNS = [f"x_{index}_change" for index in range(1, 6)]
TEST_CHANGED_COLUMNS = [f"x_{index}_changed" for index in range(1, 6)]
TEST_DIRECTION_COLUMNS = [
    f"x_{index}_changed_in_training_cf_direction" for index in range(1, 6)
]
TRAIN_CHANGE_COLUMNS = [f"x_{index}_train_change" for index in range(1, 6)]
ACTIONABILITY_PREFIX = "patient can change - "


def normalized_scalar(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def read_raw_participant_data(raw_path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, object]]:
    with raw_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    headers = rows[0]
    descriptors = rows[1] if len(rows) > 1 else [""] * len(headers)
    records = [dict(zip(headers, row)) for row in rows[3:]]

    first_blocks: dict[str, str] = {}
    ratings: dict[str, dict[str, str]] = {}
    complete = []
    excluded = Counter()
    for record in records:
        participant = record.get("ResponseId", "").strip()
        if not participant:
            excluded["missing_response_id"] += 1
            continue
        try:
            training = json.loads(record.get("training_log_json") or "[]")
            testing = json.loads(record.get("testing_log_json") or "[]")
        except json.JSONDecodeError:
            excluded["invalid_log_json"] += 1
            continue
        if record.get("Finished") != "1" or len(training) != 12 or len(testing) != 20:
            excluded["not_complete_12_training_20_testing"] += 1
            continue
        first_id = int(training[0]["instanceId"])
        first_blocks[participant] = (
            "Glucose + BMI"
            if 130100 <= first_id <= 130105
            else "Blood Pressure + Insulin"
        )
        participant_ratings: dict[str, str] = {}
        for index, header in enumerate(headers):
            if not header.startswith("FeatureActionability_"):
                continue
            descriptor = descriptors[index] if index < len(descriptors) else ""
            feature = descriptor.rsplit(" - ", 1)[-1].strip()
            if feature in FEATURES:
                participant_ratings[ACTIONABILITY_PREFIX + feature] = record.get(header, "")
        ratings[participant] = participant_ratings
        complete.append(record)

    audit = {
        "raw_records": len(records),
        "complete_participants": len(complete),
        "excluded": dict(excluded),
        "condition_counts": dict(Counter(row["xaiType"] for row in complete)),
        "actionability_features_in_export": sorted(
            {column.removeprefix(ACTIONABILITY_PREFIX) for row in ratings.values() for column in row}
        ),
    }
    return first_blocks, ratings, audit


def output_columns(rating_columns: list[str]) -> list[str]:
    with REFERENCE.open(encoding="utf-8-sig", newline="") as source:
        columns = next(csv.reader(source))

    result: list[str] = []
    for column in columns:
        if column == TEST_CHANGE_COLUMNS[0]:
            result.extend(["attribute changed", "amount changed"])
        if column in TEST_CHANGE_COLUMNS or column in TEST_CHANGED_COLUMNS:
            continue
        if column == TEST_DIRECTION_COLUMNS[0]:
            result.append("changed in training cf direction")
        if column in TEST_DIRECTION_COLUMNS:
            continue
        result.append(column)

    actionability_position = result.index("actionability (0/1)") + 1
    result[actionability_position:actionability_position] = rating_columns
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    raw_path = args.raw.resolve()
    output_path = args.output.resolve()
    audit_path = output_path.with_suffix(".audit.json")

    first_blocks, ratings, raw_audit = read_raw_participant_data(raw_path)
    with tempfile.TemporaryDirectory(prefix="qualtrics_v24_") as directory:
        base_path = Path(directory) / "base.csv"
        subprocess.run(
            [
                sys.executable,
                str(BASE_CONVERTER),
                str(raw_path),
                "--bundle",
                str(BUNDLE),
                "--output",
                str(base_path),
                "--skip-cognitive-fit",
            ],
            check=True,
        )
        base = pd.read_csv(base_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = {
        int(case["instance_id"]): case
        for case in experiment["training_pool"] + experiment["test_pool"]
    }
    rating_columns = sorted(
        {column for participant in ratings.values() for column in participant},
        key=lambda column: FEATURES.index(column.removeprefix(ACTIONABILITY_PREFIX)),
    )
    columns = output_columns(rating_columns)
    output = pd.DataFrame("", index=base.index, columns=columns)
    for column in columns:
        if column in base.columns:
            output[column] = base[column]

    output["first training block"] = output["participant"].map(first_blocks).fillna("")
    for column in rating_columns:
        output[column] = [ratings.get(participant, {}).get(column, "") for participant in output["participant"]]

    is_training = output["phase"].eq("training")
    is_testing = output["phase"].eq("testing")
    for row_index in output.index[is_training]:
        case = cases[int(float(output.at[row_index, "instance id"]))]
        for feature_index, column in enumerate(TRAIN_CHANGE_COLUMNS):
            before = float(case["raw_feature_values"][feature_index])
            after = float(case["counterfactual"]["raw_feature_values"][feature_index])
            low, high = map(float, case["raw_feature_ranges"][feature_index])
            output.at[row_index, column] = normalized_scalar((after - before) / (high - low))

    changed_feature_counts = Counter()
    amount_mismatches = 0
    for row_index in output.index[is_testing]:
        flags = [float(base.at[row_index, column] or 0) > 0.5 for column in TEST_CHANGED_COLUMNS]
        if sum(flags) != 1:
            raise RuntimeError(
                f"Testing row {row_index} has {sum(flags)} changed features; expected exactly one"
            )
        feature_index = flags.index(True)
        feature = FEATURES[feature_index]
        amount = float(base.at[row_index, TEST_CHANGE_COLUMNS[feature_index]])
        output.at[row_index, "attribute changed"] = feature
        output.at[row_index, "amount changed"] = normalized_scalar(amount)
        direction_value = base.at[row_index, TEST_DIRECTION_COLUMNS[feature_index]]
        output.at[row_index, "changed in training cf direction"] = direction_value
        changed_feature_counts[feature] += 1
        proximity = float(base.at[row_index, "dist user CF from orig"])
        if not math.isclose(abs(amount), proximity, rel_tol=0.0, abs_tol=1e-9):
            amount_mismatches += 1

    output.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    participant_rows = Counter(output["participant"])
    audit = {
        "output": str(output_path),
        "reference_schema": str(REFERENCE),
        "rows": len(output),
        "columns": len(output.columns),
        "participants": len(participant_rows),
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(output["phase"])),
        "condition_participants": {
            str(condition): int(count)
            for condition, count in output[["participant", "xai"]]
            .drop_duplicates()["xai"]
            .value_counts()
            .items()
        },
        "changed_feature_counts": dict(changed_feature_counts),
        "testing_rows_with_not_exactly_one_change": 0,
        "amount_vs_normalized_proximity_mismatches": amount_mismatches,
        "amount_changed_units": "normalized feature-range units; signed",
        "replaced_columns": TEST_CHANGE_COLUMNS + TEST_CHANGED_COLUMNS,
        "replacement_columns": ["attribute changed", "amount changed"],
        "replaced_direction_columns": TEST_DIRECTION_COLUMNS,
        "replacement_direction_column": "changed in training cf direction",
        "cognitive_model_columns": "retained for v2.3 compatibility and left blank",
        "raw": raw_audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
