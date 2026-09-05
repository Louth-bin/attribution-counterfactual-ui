"""Validate the preserved-case 20-test Recourse v1.4 build."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QSF = ROOT / "qualtrics" / "Recourse_v1.4.qsf"
STATIC = ROOT / "static" / "experiment-data.json"
ANALYSIS = ROOT / "analysis" / "diabetes_experiment_bundle_v1.4.json"
V13 = ROOT / "qualtrics" / "case-manifest-v1.3.json"
V14 = ROOT / "qualtrics" / "case-manifest-v1.4.json"


def changed(case: dict) -> int:
    return sum(
        str(before) != str(after) if kind == "categorical" else abs(float(before) - float(after)) > 1e-9
        for before, after, kind in zip(
            case["raw_feature_values"],
            case["counterfactual"]["raw_feature_values"],
            case["feature_types"],
        )
    )


def main() -> None:
    qsf = json.loads(QSF.read_text(encoding="utf-8"))
    assert qsf["SurveyEntry"]["SurveyName"] == "Recourse v1.4"
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in qsf["SurveyElements"] if element.get("Element") == "SQ"
    }
    scripts = 0
    for question_id, question in questions.items():
        source = question.get("QuestionJS", "")
        if not source:
            continue
        scripts += 1
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
            handle.write(source)
            path = Path(handle.name)
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        path.unlink(missing_ok=True)
        assert result.returncode == 0, f"{question_id}: {result.stderr}"

    blocks = next(e["Payload"] for e in qsf["SurveyElements"] if e.get("Element") == "BL")
    by_id = {block["ID"]: block for block in blocks}
    for block_id in ("BL_5aT4RYQul04UaHA", "BL_3W1PIk3tbtEX0BE"):
        options = by_id[block_id]["Options"]["LoopingOptions"]
        assert options["Randomization"] == "All"
        assert [row["1"] for row in options["Static"].values()] == [str(i) for i in range(10)]
    assert "20 trials" in questions["QID286"]["QuestionText"]
    assert "ten profiles each" in questions["QID390"]["QuestionJS"]
    assert "following ten people" in questions["QID19"]["QuestionText"]
    assert "following ten people" in questions["QID20"]["QuestionText"]

    static = json.loads(STATIC.read_text(encoding="utf-8"))
    diabetes = static["datasets"]["diabetes"]
    metadata = diabetes["metadata"]
    training_ids = list(map(int, metadata["qualtrics_v1_4_training_ids"]))
    testing_ids = {
        int(label): list(map(int, ids))
        for label, ids in metadata["qualtrics_v1_4_testing_ids_by_prediction"].items()
    }
    assert training_ids == list(range(130100, 130112))
    assert testing_ids == {0: list(range(130200, 130210)), 1: list(range(130300, 130310))}
    for question_id in ("QID12", "QID409", "QID376", "QID379"):
        js = questions[question_id]["QuestionJS"]
        assert "training: [" + ", ".join(map(str, training_ids)) + "]" in js
        assert "0: [" + ", ".join(map(str, testing_ids[0])) + "]" in js
        assert "1: [" + ", ".join(map(str, testing_ids[1])) + "]" in js

    old = json.loads(V13.read_text(encoding="utf-8"))["cases"]
    new = json.loads(V14.read_text(encoding="utf-8"))["cases"]
    old_map = {(row["source_split"], int(row["source_instance_id"])): int(row["qualtrics_instance_id"]) for row in old}
    new_map = {(row["source_split"], int(row["source_instance_id"])): int(row["qualtrics_instance_id"]) for row in new}
    assert all(new_map[key] == value for key, value in old_map.items())
    assert len(new) == 32 and len(new_map) == 32

    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = analysis["training_pool"] + analysis["test_pool"]
    assert len(analysis["training_pool"]) == 12 and len(analysis["test_pool"]) == 20
    assert Counter(int(case["prediction"]["value"]) for case in analysis["test_pool"]) == {0: 10, 1: 10}
    assert all(changed(case) == 2 for case in cases)
    assert all(int(case["counterfactual"]["prediction"]["value"]) != int(case["prediction"]["value"]) for case in cases)
    static_cases = {int(case["instance_id"]): case for case in diabetes["training_pool"] + diabetes["test_pool"]}
    assert all(int(case["instance_id"]) in static_cases for case in cases)

    print(json.dumps({
        "status": "pass",
        "question_scripts_checked": scripts,
        "training": 12,
        "testing": 20,
        "prediction_counts": {"0": 10, "1": 10},
        "v1_3_cases_preserved": len(old),
        "new_testing_cases": len(new) - len(old),
    }, indent=2))


if __name__ == "__main__":
    main()
