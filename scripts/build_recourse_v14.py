"""Build Recourse v1.4 with preserved v1.3 cases and 20 testing trials."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_recourse_v12 import build_payload, replace_diabetes_ids
from scripts.generate_static_experiment import build_metadata
from src.pipeline import ExplanationPipeline


SOURCE_QSF = ROOT / "qualtrics" / "Recourse_v1.3.qsf"
OUTPUT_QSF = ROOT / "qualtrics" / "Recourse_v1.4.qsf"
SELECTION = ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_preserved.csv"
V13_MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.3.json"
MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.4.json"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
ANALYSIS = ROOT / "analysis" / "diabetes_experiment_bundle_v1.4.json"
NEW_IDS = {0: list(range(130206, 130210)), 1: list(range(130306, 130310))}


def read_rows() -> list[dict[str, str]]:
    with SELECTION.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if Counter(row["experimental_phase"] for row in rows) != {"training": 12, "testing": 20}:
        raise RuntimeError("Unexpected phase counts")
    for phase, expected in (("training", 6), ("testing", 10)):
        if Counter(int(row["prediction"]) for row in rows if row["experimental_phase"] == phase) != {0: expected, 1: expected}:
            raise RuntimeError(f"Unexpected {phase} prediction counts")
    for row in rows:
        row["cluster"] = row["subcluster"]
        row["cluster_rank"] = row["main_cluster_rank"]
    return rows


def source_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_split"]), int(row["instance_id"])


def build_cases(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_manifest = json.loads(V13_MANIFEST.read_text(encoding="utf-8"))["cases"]
    old_ids = {
        (str(row["source_split"]), int(row["source_instance_id"])): int(row["qualtrics_instance_id"])
        for row in old_manifest
    }
    pipeline = ExplanationPipeline()
    training_rows = [row for row in rows if row["experimental_phase"] == "training"]
    testing_rows = [row for row in rows if row["experimental_phase"] == "testing"]
    training_rows.sort(key=lambda row: old_ids[source_key(row)])
    training = [build_payload(pipeline, row, old_ids[source_key(row)]) for row in training_rows]

    new_positions = Counter()
    testing = []
    for prediction in (0, 1):
        group = [row for row in testing_rows if int(row["prediction"]) == prediction]
        preserved = sorted(
            (row for row in group if source_key(row) in old_ids),
            key=lambda row: old_ids[source_key(row)],
        )
        additions = sorted(
            (row for row in group if source_key(row) not in old_ids),
            key=lambda row: (int(row["main_cluster_rank"]), float(row["distance_to_subcluster_centroid"]), source_key(row)),
        )
        if len(preserved) != 6 or len(additions) != 4:
            raise RuntimeError(f"Unexpected preserved/additional testing counts for prediction {prediction}")
        for row in preserved:
            testing.append(build_payload(pipeline, row, old_ids[source_key(row)]))
        for row in additions:
            assigned = NEW_IDS[prediction][new_positions[prediction]]
            new_positions[prediction] += 1
            testing.append(build_payload(pipeline, row, assigned))
    return training, testing


def loop_rows(count: int) -> dict[str, dict[str, str]]:
    return {str(index + 1): {"1": str(index)} for index in range(count)}


def build_qsf(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8"))
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.4"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes-warning recourse study with two hidden randomized training blocks, "
        "12 training cases, 20 testing cases, three explanation conditions, and four CRT-2 questions."
    )
    document["SurveyEntry"]["LastModified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    training_ids = [int(case["instance_id"]) for case in training]
    testing_ids = {
        prediction: [
            int(case["instance_id"])
            for case in testing
            if int(case["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    for question_id in ("QID12", "QID409", "QID376", "QID379"):
        questions[question_id]["QuestionJS"] = replace_diabetes_ids(
            questions[question_id]["QuestionJS"], training_ids, testing_ids
        )

    blocks = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks_by_id = {block["ID"]: block for block in blocks}
    for block_id, description in (
        ("BL_5aT4RYQul04UaHA", "Testing: Label 0 to Label 1 (10 Cases)"),
        ("BL_3W1PIk3tbtEX0BE", "Testing: Label 1 to Label 0 (10 Cases)"),
    ):
        block = blocks_by_id[block_id]
        block["Description"] = description
        block["Options"]["LoopingOptions"]["Static"] = loop_rows(10)
        block["Options"]["LoopingOptions"]["Randomization"] = "All"

    questions["QID286"]["QuestionText"] = questions["QID286"]["QuestionText"].replace(
        "A testing session involving 12 trials.", "A testing session involving 20 trials."
    )
    questions["QID390"]["QuestionJS"] = questions["QID390"]["QuestionJS"].replace(
        "two sessions with six profiles each", "two sessions with ten profiles each"
    )
    questions["QID19"]["QuestionText"] = questions["QID19"]["QuestionText"].replace(
        "following six people", "following ten people"
    )
    questions["QID20"]["QuestionText"] = questions["QID20"]["QuestionText"].replace(
        "following six people", "following ten people"
    )
    OUTPUT_QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def write_data(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    diabetes = bundle["datasets"]["diabetes"]
    added_ids = set(NEW_IDS[0] + NEW_IDS[1])
    diabetes["test_pool"] = [
        case for case in diabetes["test_pool"] if int(case["instance_id"]) not in added_ids
    ] + [case for case in testing if int(case["instance_id"]) in added_ids]
    metadata = diabetes["metadata"]
    metadata["static_training_pool_count"] = len(diabetes["training_pool"])
    metadata["static_test_pool_count"] = len(diabetes["test_pool"])
    metadata["qualtrics_v1_4_training_ids"] = [case["instance_id"] for case in training]
    metadata["qualtrics_v1_4_testing_ids_by_prediction"] = {
        str(prediction): [
            case["instance_id"] for case in testing
            if int(case["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    metadata["qualtrics_v1_4_training_blocks"] = metadata["qualtrics_v1_3_training_blocks"]
    bundle["version"] = "static-experiment-v14-diabetes-two-cluster-20-test"
    bundle["generated_at"] = date.today().isoformat()
    STATIC_JSON.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATIC_JS.write_text(
        "window.EXPERIMENT_DATA = " + json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    pipeline = ExplanationPipeline()
    analysis_dataset = {
        "metadata": build_metadata(pipeline, "diabetes", training, testing),
        "browser_model": diabetes["browser_model"],
        "training_pool": training,
        "test_pool": testing,
    }
    ANALYSIS.write_text(
        json.dumps({
            "version": "diabetes-experiment-v1.4",
            "generated_at": date.today().isoformat(),
            "default_model": "mlp",
            "datasets": {"diabetes": analysis_dataset},
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    rows = []
    for case in training + testing:
        rows.append({
            "domain": "diabetes",
            "experimental_phase": case["experimental_phase"],
            "qualtrics_instance_id": case["instance_id"],
            "source_split": case["source_split"],
            "source_instance_id": case["source_instance_id"],
            "prediction": int(case["prediction"]["value"]),
            "prediction_label": case["prediction"]["label"],
            "selection_cluster": case["selection_cluster"],
            "selection_cluster_rank": case["selection_cluster_rank"],
            "feature_pair_key": case["feature_pair_key"],
        })
    MANIFEST.write_text(json.dumps({"version": "1.4", "cases": rows}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    training, testing = build_cases(read_rows())
    build_qsf(training, testing)
    write_data(training, testing)
    write_manifest(training, testing)
    print(json.dumps({
        "qsf": str(OUTPUT_QSF),
        "training_ids": [case["instance_id"] for case in training],
        "testing_ids": {
            str(prediction): [case["instance_id"] for case in testing if int(case["prediction"]["value"]) == prediction]
            for prediction in (0, 1)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
