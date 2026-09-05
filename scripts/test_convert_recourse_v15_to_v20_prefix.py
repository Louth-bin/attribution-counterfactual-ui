"""Smoke-test the future Recourse v1.5 results converter without real responses."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_discrete_age.json"
CONVERTER = ROOT / "scripts" / "convert_recourse_v15_to_v20_prefix.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-boundary", action="store_true")
    parser.add_argument("--with-cognitive", action="store_true")
    args = parser.parse_args()
    diabetes = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    training_logs = []
    for case in diabetes["training_pool"]:
        prediction = int(case["prediction"]["value"])
        training_logs.append(
            {
                "domain": "diabetes",
                "caseNumber": 1,
                "instanceId": int(case["instance_id"]),
                "explanation": "counterfactuals",
                "selectedPrediction": prediction,
                "correctPrediction": prediction,
                "correct": True,
                "responseMs": 9000,
            }
        )

    testing_logs = []
    for case in diabetes["test_pool"]:
        original = list(case["raw_feature_values"])
        changed = list(case["counterfactual"]["raw_feature_values"])
        changes = []
        for index, (before, after) in enumerate(zip(original, changed)):
            if abs(float(after) - float(before)) > 1e-9:
                changes.append(
                    {
                        "attributeIndex": index,
                        "attributeName": case["feature_names"][index],
                        "originalValue": before,
                        "newValue": after,
                    }
                )
        testing_logs.append(
            {
                "domain": "diabetes",
                "caseNumberWithinDirection": 1,
                "instanceId": int(case["instance_id"]),
                "explanation": "counterfactuals",
                "originalPrediction": case["prediction"],
                "changedRawFeatureValues": dict(zip(case["raw_feature_names"], changed)),
                "changes": changes,
                "responseMs": 12000,
            }
        )

    headers = [
        "Response ID", "Completed", "Condition", "Training Log", "Testing Log",
        "QID405", "QID406", "QID407", "QID408",
    ]
    values = [
        "R_SYNTHETIC_V15", "True", "counterfactuals",
        json.dumps(training_logs), json.dumps(testing_logs),
        "second", "eight", "Emily", "no dirt",
    ]

    with tempfile.TemporaryDirectory(prefix="test_v15_converter_") as temporary:
        work = Path(temporary)
        raw = work / "renamed_raw.csv"
        output = work / "converted.csv"
        with raw.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(headers)
            writer.writerow(headers)
            writer.writerow(headers)
            writer.writerow(values)

        command = [
            sys.executable,
            str(CONVERTER),
            str(raw),
            "--output", str(output),
        ]
        if not args.with_cognitive:
            command.append("--skip-cognitive-fit")
        if not args.with_boundary:
            command.append("--skip-boundary")
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )
        with output.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            columns = list(reader.fieldnames or [])

        assert len(columns) == 65
        assert columns[-1] == "dist to nearest training instance (L1 normalized)"
        assert "nearest training instance id" not in columns
        assert len(rows) == 32
        training_numbers = {
            int(row["trial number"]) for row in rows if row["phase"] == "training"
        }
        testing_numbers = {
            int(row["trial number"]) for row in rows if row["phase"] == "testing"
        }
        assert training_numbers == set(range(1, 13))
        assert testing_numbers == set(range(1, 21))
        assert {row["xai"] for row in rows} == {"counterfactual"}
        assert {row["CRT-2 score (0-4)"] for row in rows} == {"4"}
        assert all(
            row["dist to nearest training instance (L1 normalized)"]
            for row in rows if row["phase"] == "testing"
        )
        if args.with_boundary:
            assert all(
                row["abs dist original to boundary"]
                and row["abs dist user CF to boundary"]
                and row["sign dist user CF to boundary"]
                for row in rows if row["phase"] == "testing"
            )
        if args.with_cognitive:
            assert all(row["cognitive model v0.1 family"] for row in rows)
            assert all(row["explanation reliance (η)"] for row in rows)
            assert all(row["additive change margin (ρ)"] for row in rows)
            assert all(row["age treated as actionable (0/1)"] for row in rows)
        print("status=pass rows=32 columns=65 renamed_raw_headers=pass")


if __name__ == "__main__":
    main()
