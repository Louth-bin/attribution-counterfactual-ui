"""Validate the discrete-LIME, age-cluster Recourse v1.5 build."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QSF = ROOT / "qualtrics" / "Recourse_v1.5.qsf"
BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_discrete_age.json"
SELECTION = ROOT / "outputs" / "v15-discrete-lime-age" / "selected_clusters.json"
MODEL = ROOT / "analysis" / "diabetes_mlp_regularized.joblib"
IFRAME_JS = ROOT / "iframe.js"
FRAME_JS = ROOT / "qualtrics" / "qualtrics-frame.js"
FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]


def changed_indices(case: dict) -> list[int]:
    return [
        index
        for index, (before, after) in enumerate(
            zip(case["raw_feature_values"], case["counterfactual"]["raw_feature_values"])
        )
        if abs(float(after) - float(before)) > 1e-9
    ]


def check_js(source: str, label: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
        handle.write(source)
        path = Path(handle.name)
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    path.unlink(missing_ok=True)
    assert result.returncode == 0, f"{label}: {result.stderr}"


def find_values(node, key: str) -> list:
    found = []
    if isinstance(node, dict):
        if key in node:
            found.append(node[key])
        for value in node.values():
            found.extend(find_values(value, key))
    elif isinstance(node, list):
        for value in node:
            found.extend(find_values(value, key))
    return found


def main() -> None:
    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    dataset = experiment["datasets"]["diabetes"]
    training = dataset["training_pool"]
    testing = dataset["test_pool"]
    cases = training + testing
    assert len(training) == 12 and len(testing) == 20
    assert Counter(int(case["prediction"]["value"]) for case in training) == {0: 6, 1: 6}
    assert Counter(int(case["prediction"]["value"]) for case in testing) == {0: 10, 1: 10}
    assert Counter(case["feature_pair_key"] for case in training) == {
        "age|glucose": 6,
        "glucose|insulin": 6,
    }
    assert Counter(case["feature_pair_key"] for case in testing) == {
        "age|glucose": 10,
        "glucose|insulin": 10,
    }
    assert all(len(changed_indices(case)) == 2 for case in cases)
    assert all(case["attribution"]["feature_selection"]["discretize_continuous"] for case in cases)
    assert all(case["attribution"]["featureConditions"] for case in cases)
    assert all(
        set(changed_indices(case)) == set(case["attribution"]["shown_feature_indices"])
        for case in cases
    )
    assert all(4 in changed_indices(case) for case in cases if case["feature_pair_key"] == "age|glucose")
    assert all("age" not in case["counterfactual_settings"]["raw_controllable_feature_names"] for case in cases)
    assert len({(case["source_split"], case["source_instance_id"]) for case in cases}) == 32

    model = joblib.load(MODEL)
    for case in cases:
        original = pd.DataFrame([case["raw_feature_values"]], columns=FEATURES)
        edited = pd.DataFrame([case["counterfactual"]["raw_feature_values"]], columns=FEATURES)
        assert int(model.predict(original)[0]) == int(case["prediction"]["value"])
        assert int(model.predict(edited)[0]) == int(case["counterfactual"]["prediction"]["value"])
        assert int(case["prediction"]["value"]) != int(case["counterfactual"]["prediction"]["value"])

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    transfer = selection["copy_strategy_validation"]
    assert transfer["testing_cases"] == 20
    assert transfer["success_rate"] >= 0.90
    assert transfer["mean_normalized_delta_distance"] <= 0.04
    assert any("Age" in cluster["pair"] for cluster in selection["selected_clusters"])

    qsf = json.loads(QSF.read_text(encoding="utf-8"))
    assert qsf["SurveyEntry"]["SurveyName"] == "Recourse v1.5"
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in qsf["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    for question_id, question in questions.items():
        if question.get("QuestionJS"):
            check_js(question["QuestionJS"], question_id)

    assignment_values = find_values(
        next(element["Payload"] for element in qsf["SurveyElements"] if element.get("Element") == "FL"),
        "Value",
    )
    assert "counterfactual" in assignment_values
    assert "counterfactuals" not in assignment_values
    for qid in ("QID18", "QID21", "QID22", "QID23", "QID24", "QID25"):
        assert questions[qid]["Choices"]["1"]["Display"]
        assert "What does" not in questions[qid]["QuestionText"] or qid in {"QID23", "QID25"}
    assert questions["QID18"]["QuestionText"] == "What AI output is selected for the tutorial profile?"
    assert questions["QID21"]["QuestionText"] == "What is the Glucose value in the tutorial profile?"
    assert questions["QID22"]["QuestionText"] == (
        "For the current case, which are the two most influential attributes?"
    )
    assert questions["QID23"]["QuestionText"] == "What does a red influence bar mean?"
    assert questions["QID24"]["QuestionText"].endswith("both counter-example changes exactly?")
    assert questions["QID25"]["QuestionText"] == "What does a red change marker mean?"
    for first, second in (
        ("QID18", "QID27"),
        ("QID21", "QID28"),
        ("QID22", "QID29"),
        ("QID23", "QID30"),
        ("QID24", "QID31"),
        ("QID25", "QID32"),
    ):
        assert questions[first]["QuestionText"] == questions[second]["QuestionText"]
        assert questions[first]["Choices"] == questions[second]["Choices"]

    tutorial_text = questions["QID9"]["QuestionText"]
    assert "The numbered explanation interface shows:" in tutorial_text
    assert "Red bars</span> contribute towards a <b>Diabetes</b> warning" in tutorial_text
    assert "Blue bars</span> contribute towards a <b>No Diabetes</b> warning" in tutorial_text
    assert 'Glucose, <span class="tutorial-red">−69%</span>' in tutorial_text
    assert 'Age, <span class="tutorial-red">−31%</span>' in tutorial_text
    assert 'Glucose</span> decreases from 173 to 141' in tutorial_text
    assert 'Age</span> increases from 31 to 35' in tutorial_text
    assert 'data-instance-id="130100"' in tutorial_text
    assert '"diabetes":{"attribution":130100,"counterfactual":130100}' in questions["QID9"]["QuestionJS"]
    basic_tutorial_text = questions["QID271"]["QuestionText"]
    assert "The five <b>attributes</b> describing the person." in basic_tutorial_text
    assert "The <b>values</b> of each attribute." in basic_tutorial_text
    assert "Bars indicating how <b>low/high</b> the value is" in basic_tutorial_text
    assert "The selected box shows whether the AI issues" in basic_tutorial_text
    assert '"diabetes":[130100,' in questions["QID271"]["QuestionJS"]
    assert questions["QID18"]["Choices"]["1"]["Display"] == "Diabetes"
    assert questions["QID21"]["Choices"]["1"]["Display"] == "173"
    assert questions["QID22"]["Choices"]["1"]["Display"] == "Glucose and Age"
    assert "These are the changes the person should avoid." in questions["QID20"]["QuestionText"]
    assert "These are the changes the person should avoid." in questions["QID379"]["QuestionText"]

    frame_source = FRAME_JS.read_text(encoding="utf-8")
    check_js(frame_source, "qualtrics-frame.js")
    assert "TRAINING_EXPLANATION_DELAY_MS = 8000" in frame_source
    assert "data.screenState.explanationType === explanation" in frame_source
    assert "Review the explanation carefully" in frame_source
    assert "cf-training-check" not in frame_source
    assert "cf-testing-check" not in frame_source
    assert "Now answer the quick check" not in frame_source
    assert "Answer the quick check" not in frame_source
    assert 'title.textContent = "Training case";' in frame_source
    assert questions["QID12"]["QuestionJS"] == frame_source
    assert questions["QID409"]["QuestionJS"] == frame_source
    preview_js = questions["QID26"]["QuestionJS"]
    assert "question.disableNextButton()" in preview_js
    assert "counterfactual-ui:simulation-change" in preview_js
    assert "question.enableNextButton()" in preview_js

    iframe_source = IFRAME_JS.read_text(encoding="utf-8")
    check_js(iframe_source, "iframe.js")
    assert "featureConditions?.[index]" not in iframe_source
    assert "joinClauses(changes)" in iframe_source
    assert "signedPercent" in iframe_source

    print(
        json.dumps(
            {
                "status": "pass",
                "question_scripts_checked": sum(bool(q.get("QuestionJS")) for q in questions.values()),
                "training_cases": len(training),
                "testing_cases": len(testing),
                "age_changing_cases": sum(4 in changed_indices(case) for case in cases),
                "copy_transfer_success": transfer["success_rate"],
                "mean_copy_delta_distance": transfer["mean_normalized_delta_distance"],
                "screening_questions_per_condition": {"none": 2, "attribution": 4, "counterfactual": 4},
                "attempts": 2,
                "training_delay_seconds": 8,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
