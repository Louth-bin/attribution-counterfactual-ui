"""Update only the v1.7 screening items to match displayed case 130100."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QSF = ROOT / "qualtrics" / "Recourse_v1.7.qsf"
BACKUP = ROOT / "qualtrics" / "Recourse_v1.7_before_screening_update.qsf"
STATIC = ROOT / "static" / "experiment-data.json"

SCREENING_IDS = {
    "QID18", "QID21", "QID22", "QID23", "QID24", "QID25",
    "QID27", "QID28", "QID29", "QID30", "QID31", "QID32",
}


def displayed_number(value: float) -> str:
    return f"{float(value):g}"


def update_mcq(
    questions: dict[str, dict[str, Any]],
    question_id: str,
    text: str,
    choices: list[str],
) -> None:
    question = questions[question_id]
    if len(question.get("Choices", {})) < len(choices):
        raise RuntimeError(f"{question_id} does not have four existing choices")
    question["QuestionText"] = text
    question["QuestionDescription"] = text
    for choice_number, display in enumerate(choices, start=1):
        question["Choices"][str(choice_number)]["Display"] = display


def extract_diabetes_tutorial_section(text: str, explanation: str) -> str:
    pattern = re.compile(
        rf'<section class="tutorial-copy" data-domain="diabetes" '
        rf'data-explanation="{explanation}".*?</section>',
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"Missing diabetes {explanation} tutorial section")
    return match.group(0)


def main() -> None:
    original_text = QSF.read_text(encoding="utf-8")
    document = json.loads(original_text)
    before = copy.deepcopy(document)
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    required = SCREENING_IDS | {"QID9", "QID35"}
    missing = sorted(required - set(questions))
    if missing:
        raise RuntimeError(f"Missing required questions: {missing}")

    static = json.loads(STATIC.read_text(encoding="utf-8"))
    diabetes = static["datasets"]["diabetes"]
    case = next(
        item for item in diabetes["training_pool"] if int(item["instance_id"]) == 130100
    )
    names = list(case["feature_names"])
    originals = [float(value) for value in case["raw_feature_values"]]
    counterfactuals = [
        float(value) for value in case["counterfactual"]["raw_feature_values"]
    ]
    values = dict(zip(names, originals))
    prediction = str(case["prediction"]["label"])
    alternative_prediction = next(
        label for label in case["prediction_labels"] if label != prediction
    )
    shown_names = [names[index] for index in case["attribution"]["shown_feature_indices"]]
    changes = [
        (name, before_value, after_value)
        for name, before_value, after_value in zip(names, originals, counterfactuals)
        if abs(after_value - before_value) > 1e-9
    ]
    if len(shown_names) != 2 or len(changes) != 2:
        raise RuntimeError("Tutorial case no longer has two shown influences and two changes")

    output_choices = [prediction, alternative_prediction, "The interface does not show an output", "Both outputs"]
    for question_id in ("QID18", "QID27"):
        update_mcq(
            questions,
            question_id,
            "What AI output is selected for the tutorial profile?",
            output_choices,
        )

    glucose = displayed_number(values["Glucose"])
    glucose_distractors = [
        displayed_number(counterfactuals[names.index("Glucose")]),
        displayed_number(values["Blood Pressure"]),
        displayed_number(values["Age"]),
    ]
    for question_id in ("QID21", "QID28"):
        update_mcq(
            questions,
            question_id,
            "What is the Glucose value in the tutorial profile?",
            [glucose, *glucose_distractors],
        )

    influence_pair = f"{shown_names[0]} and {shown_names[1]}"
    for question_id in ("QID22", "QID29"):
        update_mcq(
            questions,
            question_id,
            "For the current case, which are the two most influential attributes?",
            [
                influence_pair,
                "Glucose and Age",
                "Blood Pressure and Insulin",
                "Insulin and BMI",
            ],
        )

    for question_id in ("QID23", "QID30"):
        update_mcq(
            questions,
            question_id,
            "What does a red influence bar mean?",
            [
                "The attribute's current value contributes toward a Diabetes warning",
                "The attribute's current value contributes toward a No Diabetes warning",
                "The attribute value decreased",
                "The attribute was ignored by the AI",
            ],
        )

    change_text = []
    reverse_text = []
    for name, before_value, after_value in changes:
        delta = after_value - before_value
        sign = "+" if delta > 0 else "−"
        change_text.append(
            f"{name}: {displayed_number(before_value)} → {displayed_number(after_value)} "
            f"({sign}{displayed_number(abs(delta))})"
        )
        reverse_sign = "+" if -delta > 0 else "−"
        reverse_text.append(
            f"{name}: {displayed_number(after_value)} → {displayed_number(before_value)} "
            f"({reverse_sign}{displayed_number(abs(delta))})"
        )
    for question_id in ("QID24", "QID31"):
        update_mcq(
            questions,
            question_id,
            "Which option gives both counter-example changes exactly?",
            [
                "; ".join(change_text),
                "; ".join(reverse_text),
                f"Only {changes[0][0]} changed",
                "No attribute values changed",
            ],
        )

    for question_id in ("QID25", "QID32"):
        update_mcq(
            questions,
            question_id,
            "What does a red change marker mean?",
            [
                "The attribute value decreased",
                "The attribute value increased",
                "The attribute caused a Diabetes warning",
                "The attribute is the most important feature",
            ],
        )

    # Attempt 2 displays QID35. Replace only its two diabetes panels with the
    # already-current panels from QID9, preserving every other tutorial panel.
    attempt_two = questions["QID35"]["QuestionText"]
    for explanation in ("attribution", "counterfactual"):
        current_section = extract_diabetes_tutorial_section(
            questions["QID9"]["QuestionText"], explanation
        )
        pattern = re.compile(
            rf'<section class="tutorial-copy" data-domain="diabetes" '
            rf'data-explanation="{explanation}".*?</section>',
            flags=re.DOTALL,
        )
        attempt_two, replacement_count = pattern.subn(
            lambda _match: current_section,
            attempt_two,
            count=1,
        )
        if replacement_count != 1:
            raise RuntimeError(f"Could not synchronize QID35 {explanation} tutorial")
    questions["QID35"]["QuestionText"] = attempt_two

    # Assert that no question outside the explicit screening scope (plus QID35)
    # changed. This protects edits the user has already made elsewhere in the QSF.
    before_questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in before["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    allowed = SCREENING_IDS | {"QID35"}
    unexpectedly_changed = sorted(
        question_id
        for question_id in set(questions) & set(before_questions)
        if questions[question_id] != before_questions[question_id]
        and question_id not in allowed
    )
    if unexpectedly_changed:
        raise RuntimeError(f"Unexpectedly changed questions: {unexpectedly_changed}")

    if not BACKUP.exists():
        BACKUP.write_text(original_text, encoding="utf-8")
    QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "updated_qsf": str(QSF),
                "backup": str(BACKUP),
                "updated_question_ids": sorted(SCREENING_IDS),
                "synchronized_attempt_two_tutorial": "QID35 diabetes panels only",
                "reference_instance_id": 130100,
                "prediction": prediction,
                "glucose": glucose,
                "influential_attributes": shown_names,
                "counterfactual_changes": change_text,
                "other_questions_changed": [],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
