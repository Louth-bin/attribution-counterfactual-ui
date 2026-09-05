"""Promote the validated actionable preview into a new Qualtrics v1.7 package."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"
SOURCE_QSF = QUALTRICS / "Recourse_v1.5.qsf"
OUTPUT_QSF = QUALTRICS / "Recourse_v1.7.qsf"
PREVIEW = QUALTRICS / "experimental-actionable-preview-data.json"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
FRAME_SCRIPT = QUALTRICS / "qualtrics-frame.js"
ANALYSIS_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.7_actionable.json"
MANIFEST = QUALTRICS / "case-manifest-v1.7.json"

TRAIN_IDS = list(range(130100, 130112))
TEST_IDS = {0: list(range(130200, 130210)), 1: list(range(130300, 130310))}


def assign_slots(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training = sorted(
        (copy.deepcopy(case) for case in source["training_pool"]),
        key=lambda case: (
            int(case["selection_cluster"]),
            int(case["prediction"]["value"]),
            str(case["source_split"]),
            int(case["source_instance_id"]),
        ),
    )
    if len(training) != 12:
        raise RuntimeError(f"Expected 12 training cases, found {len(training)}")
    for assigned_id, case in zip(TRAIN_IDS, training):
        case["instance_id"] = assigned_id

    testing: list[dict[str, Any]] = []
    for prediction in (0, 1):
        group = sorted(
            (
                copy.deepcopy(case)
                for case in source["test_pool"]
                if int(case["prediction"]["value"]) == prediction
            ),
            key=lambda case: (
                int(case["selection_cluster"]),
                str(case["source_split"]),
                int(case["source_instance_id"]),
            ),
        )
        if len(group) != 10:
            raise RuntimeError(
                f"Expected 10 testing cases for prediction {prediction}, found {len(group)}"
            )
        for assigned_id, case in zip(TEST_IDS[prediction], group):
            case["instance_id"] = assigned_id
        testing.extend(group)
    return training, testing


def tutorial_section(case: dict[str, Any], explanation: str) -> str:
    if explanation == "counterfactual":
        changes = []
        for name, before, after in zip(
            case["feature_names"],
            case["raw_feature_values"],
            case["counterfactual"]["raw_feature_values"],
        ):
            if abs(float(after) - float(before)) <= 1e-9:
                continue
            color = "tutorial-red" if float(after) < float(before) else "tutorial-blue"
            changes.append(
                f'<li><span class="{color}">{name}</span> changes from {before:g} to {after:g}.</li>'
            )
        return (
            '<section class="tutorial-copy" data-domain="diabetes" '
            'data-explanation="counterfactual" data-instance-id="130100" hidden>\n'
            '  <p>To learn how the AI predicts, you will see an explanation for the AI\'s '
            'diabetes warning. This explanation shows a <b>counter-example</b>, where '
            '<b>two attributes are changed</b> to alter the warning.</p>\n'
            '  <p>The numbered explanation interface shows:</p>\n'
            '  <ol class="tutorial-bullets tutorial-bullets-compact">\n'
            f'    <li>The changes in the two attributes in the counter-example.<ul>{"".join(changes)}</ul></li>\n'
            '    <li>A sentence describing the warning and the effect of the changes.</li>\n'
            '  </ol>\n</section>'
        )

    shown = list(case["attribution"]["shown_feature_indices"])
    values = [float(case["attribution"]["values"][index]) for index in shown]
    denominator = sum(abs(value) for value in values) or 1.0
    descriptions = []
    for index, value in zip(shown, values):
        percentage = round(100 * abs(value) / denominator)
        color = "tutorial-red" if value < 0 else "tutorial-blue"
        sign = "−" if value < 0 else "+"
        descriptions.append(
            f'{case["feature_names"][index]}, <span class="{color}">{sign}{percentage}%</span>'
        )
    return (
        '<section class="tutorial-copy" data-domain="diabetes" '
        'data-explanation="attribution" data-instance-id="130100" hidden>\n'
        '  <p>To learn how the AI predicts, you will see an explanation for the AI\'s '
        'diabetes warning. This explanation shows the <b>two most important attributes</b> '
        'for that profile.</p>\n'
        '  <p>The numbered explanation interface shows:</p>\n'
        '  <ol class="tutorial-bullets tutorial-bullets-compact">\n'
        f'    <li>The influence of the two most important attributes ({" and ".join(descriptions)}). '
        'The higher the number, the stronger the influence.\n'
        '      <ul><li><span class="tutorial-red">Red bars</span> contribute towards a '
        '<b>Diabetes</b> warning.</li><li><span class="tutorial-blue">Blue bars</span> '
        'contribute towards a <b>No Diabetes</b> warning.</li></ul>\n'
        '    </li>\n'
        '    <li>A sentence describing the warning and the influences.</li>\n'
        '  </ol>\n</section>'
    )


def write_qsf(training: list[dict[str, Any]]) -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8"))
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.7"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes recourse study with actionable two-feature training and "
        "one-feature glucose-locked testing"
    )

    elements = document["SurveyElements"]
    blocks = next(element["Payload"] for element in elements if element.get("Element") == "BL")
    blocks_by_id = {block["ID"]: block for block in blocks}
    blocks_by_id["BL_3fJJUKH6EXaeRIq"]["Description"] = (
        "Training block A (hidden): Glucose + BMI"
    )
    blocks_by_id["BL_6Hb2Nq8Tx4Lm7Wp"]["Description"] = (
        "Training block B (hidden): Blood Pressure + Insulin"
    )

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in elements
        if element.get("Element") == "SQ"
    }
    frame_script = FRAME_SCRIPT.read_text(encoding="utf-8")
    for question_id in ("QID12", "QID409", "QID376", "QID379"):
        questions[question_id]["QuestionJS"] = frame_script

    tutorial = questions["QID9"]
    text = tutorial["QuestionText"]
    for explanation in ("attribution", "counterfactual"):
        pattern = re.compile(
            rf'<section class="tutorial-copy" data-domain="diabetes" '
            rf'data-explanation="{explanation}".*?</section>',
            flags=re.DOTALL,
        )
        text, count = pattern.subn(tutorial_section(training[0], explanation), text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not replace diabetes {explanation} tutorial")
    tutorial["QuestionText"] = text

    OUTPUT_QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def main() -> None:
    preview = json.loads(PREVIEW.read_text(encoding="utf-8"))
    source = preview["datasets"]["diabetes"]
    training, testing = assign_slots(source)

    if Counter(case["feature_pair_key"] for case in training) != {
        "glucose|bmi": 6,
        "blood_pressure|insulin": 6,
    }:
        raise RuntimeError("Unexpected training pair distribution")
    if any(case["counterfactual_settings"].get("preview_max_changed_features") != 1 for case in testing):
        raise RuntimeError("A testing case is missing the one-feature constraint")
    if any("Glucose" not in case["counterfactual_settings"].get("preview_immutable_feature_names", []) for case in testing):
        raise RuntimeError("A testing case is missing the Glucose lock")

    bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    diabetes = bundle["datasets"]["diabetes"]
    experiment_ids = set(TRAIN_IDS + TEST_IDS[0] + TEST_IDS[1])
    diabetes["training_pool"] = [
        case for case in diabetes["training_pool"] if int(case["instance_id"]) not in experiment_ids
    ] + copy.deepcopy(training)
    diabetes["test_pool"] = [
        case for case in diabetes["test_pool"] if int(case["instance_id"]) not in experiment_ids
    ] + copy.deepcopy(testing)
    diabetes["browser_model"] = copy.deepcopy(source["browser_model"])
    diabetes["metadata"].update(
        {
            "model": "smoother_regularized_mlp_alpha_1.0_hidden_8x4",
            "xai_methods": ["discrete_lime"],
            "qualtrics_v1_7_training_ids": TRAIN_IDS,
            "qualtrics_v1_7_testing_ids_by_prediction": {
                str(prediction): ids for prediction, ids in TEST_IDS.items()
            },
            "qualtrics_v1_7_training_blocks": {
                "glucose_bmi": TRAIN_IDS[:6],
                "blood_pressure_insulin": TRAIN_IDS[6:],
            },
            "qualtrics_v1_7_training_change_range": [0.15, 0.25],
            "qualtrics_v1_7_testing_minimum_change": 0.10,
            "qualtrics_v1_7_testing_immutable_features": ["Glucose"],
            "qualtrics_v1_7_testing_max_changed_features": 1,
            "qualtrics_v1_7_objective": source["metadata"]["preview_objective"],
            "qualtrics_v1_7_plausibility_reference": source["metadata"]["preview_plausibility_reference"],
            "qualtrics_v1_7_testing_selection": source["metadata"]["preview_testing_selection"],
            "static_training_pool_count": len(diabetes["training_pool"]),
            "static_test_pool_count": len(diabetes["test_pool"]),
        }
    )
    bundle["version"] = "static-experiment-v17-actionable-transfer-consistent"
    bundle["generated_at"] = date.today().isoformat()

    STATIC_JSON.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATIC_JS.write_text(
        "window.EXPERIMENT_DATA = "
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    promoted = copy.deepcopy(preview)
    promoted["version"] = "diabetes-experiment-v1.7-actionable-transfer-consistent"
    promoted["datasets"]["diabetes"]["training_pool"] = copy.deepcopy(training)
    promoted["datasets"]["diabetes"]["test_pool"] = copy.deepcopy(testing)
    ANALYSIS_BUNDLE.write_text(
        json.dumps(promoted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = []
    for case in training + testing:
        manifest.append(
            {
                "domain": "diabetes",
                "experimental_phase": case["experimental_phase"],
                "qualtrics_instance_id": int(case["instance_id"]),
                "source_split": case["source_split"],
                "source_instance_id": int(case["source_instance_id"]),
                "prediction": int(case["prediction"]["value"]),
                "prediction_label": case["prediction"]["label"],
                "selection_cluster": int(case["selection_cluster"]),
                "feature_pair_key": case["feature_pair_key"],
                "nearest_training_source": case.get("nearest_training_source"),
            }
        )
    MANIFEST.write_text(
        json.dumps({"version": "1.7", "cases": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_qsf(training)

    print(
        json.dumps(
            {
                "qsf": str(OUTPUT_QSF),
                "bundle_version": bundle["version"],
                "training_cases": len(training),
                "testing_cases": len(testing),
                "training_pairs": dict(Counter(case["feature_pair_key"] for case in training)),
                "testing_labels": dict(Counter(case["prediction"]["label"] for case in testing)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
