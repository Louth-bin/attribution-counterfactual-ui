"""Validate the promoted v1.7 QSF and static instance bundle."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"


def main() -> None:
    qsf = json.loads((QUALTRICS / "Recourse_v1.7.qsf").read_text(encoding="utf-8"))
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in qsf["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    frame = (QUALTRICS / "qualtrics-frame.js").read_text(encoding="utf-8")
    assert qsf["SurveyEntry"]["SurveyName"] == "Recourse v1.7"
    assert all(
        questions[question_id]["QuestionJS"] == frame
        for question_id in ("QID12", "QID409", "QID376", "QID379")
    )
    assert 'parameters.set("immutableFeatures", "Glucose")' in frame
    assert 'parameters.set("maxChangedFeatures", "1")' in frame
    assert (
        'data-explanation="counterfactual" data-instance-id="130100"'
        in questions["QID9"]["QuestionText"]
    )

    bundle = json.loads((ROOT / "static" / "experiment-data.json").read_text(encoding="utf-8"))
    diabetes = bundle["datasets"]["diabetes"]
    training = [
        case
        for case in diabetes["training_pool"]
        if 130100 <= int(case["instance_id"]) <= 130111
    ]
    testing_ids = set(range(130200, 130210)) | set(range(130300, 130310))
    testing = [
        case for case in diabetes["test_pool"] if int(case["instance_id"]) in testing_ids
    ]
    assert len(training) == 12
    assert len(testing) == 20
    assert Counter(case["feature_pair_key"] for case in training) == {
        "glucose|bmi": 6,
        "blood_pressure|insulin": 6,
    }
    assert Counter(int(case["prediction"]["value"]) for case in testing) == {0: 10, 1: 10}
    assert all(
        case["counterfactual_settings"]["preview_max_changed_features"] == 1
        and "Glucose"
        in case["counterfactual_settings"]["preview_immutable_feature_names"]
        for case in testing
    )
    assert (
        "profile distance <= 0.20"
        in diabetes["metadata"]["qualtrics_v1_7_testing_selection"]
    )

    print(
        json.dumps(
            {
                "qsf_json_valid": True,
                "embedded_frame_scripts_current": True,
                "training_cases": len(training),
                "testing_cases": len(testing),
                "training_pairs": dict(Counter(case["feature_pair_key"] for case in training)),
                "testing_prediction_counts": dict(
                    Counter(case["prediction"]["label"] for case in testing)
                ),
                "testing_glucose_locked": True,
                "testing_max_changed_features": 1,
                "bundle_version": bundle["version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
