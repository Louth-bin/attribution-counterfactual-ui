"""Build Recourse_v10 from v09 while preserving the existing survey design."""

from __future__ import annotations

import copy
import html
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_QSF = REPO_ROOT / "qualtrics" / "Recourse_v09.qsf"
OUTPUT_QSF = REPO_ROOT / "qualtrics" / "Recourse_v10.qsf"
STATIC_DATA = REPO_ROOT / "static" / "experiment-data.json"
FRAME_JS = REPO_ROOT / "qualtrics" / "qualtrics-frame.js"
SCREENING_BLOCK_ID = "BL_9ScreenAttempt2"


def question_elements(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        element["PrimaryAttribute"]: element
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }


def blocks(document: dict[str, Any]) -> list[dict[str, Any]]:
    payload = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    return list(payload.values()) if isinstance(payload, dict) else payload


def format_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        rounded = round(float(value), 1)
        return str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"
    return str(value)


def case_ids(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain, bundle in data["datasets"].items():
        result[domain] = {
            "training": [int(case["instance_id"]) for case in bundle["training_pool"]],
            "test": {
                str(label): [
                    int(case["instance_id"])
                    for case in bundle["test_pool"]
                    if int(case["prediction"]["value"]) == label
                ]
                for label in (0, 1)
            },
        }
    return result


def insert_before(text: str, marker: str, addition: str) -> str:
    if marker not in text:
        raise RuntimeError(f"Could not find insertion marker: {marker!r}")
    return text.replace(marker, addition + marker, 1)


def update_scenario(payload: dict[str, Any]) -> None:
    diabetes_panel = """
  <section class="scenario-card" data-domain="diabetes" hidden>
    <p class="scenario-eyebrow">DIABETES WARNING DOMAIN</p>
    <h1>Understanding an AI diabetes warning</h1>
    <p class="scenario-lead">Imagine that you are reviewing health profiles. An AI uses five details to decide whether to issue a <b>Diabetes</b> or <b>No Diabetes</b> warning.</p>
    <h2>Information available to the AI</h2>
    <table class="domain-table"><tr><th>Attribute</th><th>Description</th><th>Typical range</th></tr>
      <tr><td>Blood Glucose</td><td>Blood glucose concentration</td><td>74–192 mg/dL</td></tr>
      <tr><td>Diastolic Blood Pressure</td><td>Diastolic blood pressure</td><td>48–94 mmHg</td></tr>
      <tr><td>Serum Insulin</td><td>Two-hour serum insulin</td><td>36–521 µU/mL</td></tr>
      <tr><td>BMI</td><td>Body mass index</td><td>20.4–48.9 kg/m²</td></tr>
      <tr><td>Age</td><td>Age in years</td><td>21–58 years</td></tr>
    </table>
  </section>
"""
    payload["QuestionText"] = insert_before(
        payload["QuestionText"], "</div>\n<style>", diabetes_panel
    )


def update_basic_tutorial(payload: dict[str, Any], ids: dict[str, Any]) -> None:
    basic_case_id = ids["diabetes"]["training"][0]
    panel = """<section class="tutorial-copy" data-domain="diabetes" hidden>
  <p>Each health profile is shown using the following interface.</p>
  <ol class="tutorial-bullets">
    <li>The five <b>attributes</b> describing the person.</li>
    <li>The <b>values</b> of each attribute.</li>
    <li>Bars indicating how <b>low/high</b> the value is for that attribute (for example, Blood Glucose is shown relative to the range from about 74 to 192 mg/dL).</li>
    <li>The selected box shows whether the AI issues a <b>Diabetes</b> or <b>No Diabetes</b> warning.</li>
  </ol>
</section>"""
    payload["QuestionText"] = insert_before(
        payload["QuestionText"],
        "</div>\n    <div class=\"tutorial-preview\">",
        panel,
    )
    training_ids = {
        domain: values["training"] for domain, values in ids.items()
    }
    payload["QuestionJS"] = re.sub(
        r"var ids = \{.*?\};",
        "var ids = " + json.dumps(training_ids, separators=(",", ":")) + ";",
        payload["QuestionJS"],
        count=1,
    )
    if str(basic_case_id) not in payload["QuestionJS"]:
        raise RuntimeError("Diabetes basic tutorial ID was not embedded")


def attribution_panel(case: dict[str, Any]) -> str:
    selected = list(case["counterfactual"]["raw_selected_feature_names"])
    raw_names = list(case["raw_feature_names"])
    values = list(case["attribution"]["values"])
    indices = [raw_names.index(name) for name in selected]
    magnitudes = [abs(float(values[index])) for index in indices]
    total = sum(magnitudes) or 1.0
    percentages = [round(value * 100 / total) for value in magnitudes]
    percentages[-1] = 100 - sum(percentages[:-1])
    items = []
    for index, percentage in zip(indices, percentages):
        contribution = float(values[index])
        css_class = "tutorial-blue" if contribution >= 0 else "tutorial-red"
        sign = "+" if contribution >= 0 else "-"
        items.append(
            f"{html.escape(str(case['feature_names'][index]))}, "
            f'<span class="{css_class}">{sign}{percentage}%</span>'
        )
    return f"""<section class="tutorial-copy" data-domain="diabetes" data-explanation="attribution" data-instance-id="{case['instance_id']}" hidden>
  <p>To learn how the AI predicts, you will see an explanation for the AI's diabetes warning. This explanation shows the <b>two most important attributes</b> for that profile.</p>
  <p>The numbered explanation interface shows:</p>
  <ol class="tutorial-bullets tutorial-bullets-compact">
    <li>The influence of the two most important attributes ({items[0]} and {items[1]}). The higher the number, the stronger the influence.
      <ul><li><span class="tutorial-red">Red bars</span> contribute towards a <b>Diabetes</b> warning.</li><li><span class="tutorial-blue">Blue bars</span> contribute towards a <b>No Diabetes</b> warning.</li></ul>
    </li>
    <li>A sentence describing the warning and the influences.</li>
  </ol>
</section>"""


def counterfactual_panel(case: dict[str, Any]) -> str:
    raw_names = list(case["raw_feature_names"])
    selected = list(case["counterfactual"]["raw_selected_feature_names"])
    changes = []
    for position, raw_name in enumerate(selected):
        index = raw_names.index(raw_name)
        display_name = html.escape(str(case["feature_names"][index]))
        original = format_value(case["raw_feature_values"][index])
        updated = format_value(case["counterfactual"]["raw_feature_values"][index])
        css_class = "tutorial-red" if position == 0 else "tutorial-blue"
        verb = "increases" if float(updated) > float(original) else "decreases"
        changes.append(
            f'<li><span class="{css_class}">{display_name}</span> '
            f"{verb} from {original} to {updated}.</li>"
        )
    return f"""<section class="tutorial-copy" data-domain="diabetes" data-explanation="counterfactual" data-instance-id="{case['instance_id']}" hidden>
  <p>To learn how the AI predicts, you will see an explanation for the AI's diabetes warning. This explanation shows a <b>counter-example</b>, where <b>two attributes are changed</b> to alter the warning.</p>
  <p>The numbered explanation interface shows:</p>
  <ol class="tutorial-bullets tutorial-bullets-compact">
    <li>The changes in the two attributes in the counter-example.<ul>{''.join(changes)}</ul></li>
    <li>A sentence describing the warning and the effect of the changes.</li>
  </ol>
</section>"""


def update_explanation_tutorial(
    payload: dict[str, Any],
    data: dict[str, Any],
) -> tuple[int, int]:
    training = data["datasets"]["diabetes"]["training_pool"]
    attribution_case = training[0]
    counterfactual_case = next(
        case for case in training if int(case["prediction"]["value"]) == 0
    )
    panels = attribution_panel(attribution_case) + counterfactual_panel(counterfactual_case)
    payload["QuestionText"] = insert_before(
        payload["QuestionText"],
        "</div>\n    <div class=\"tutorial-preview\">",
        panels,
    )
    tutorial_ids = {
        "housing": {"attribution": 14693, "counterfactual": 2968},
        "safelimit": {"attribution": 960, "counterfactual": 610},
        "diabetes": {
            "attribution": int(attribution_case["instance_id"]),
            "counterfactual": int(counterfactual_case["instance_id"]),
        },
    }
    payload["QuestionJS"] = re.sub(
        r"var ids = \{.*?\};",
        "var ids = " + json.dumps(tutorial_ids, separators=(",", ":")) + ";",
        payload["QuestionJS"],
        count=1,
    )
    return (
        int(attribution_case["instance_id"]),
        int(counterfactual_case["instance_id"]),
    )


def screening_javascript() -> str:
    return """Qualtrics.SurveyEngine.addOnload(function () {
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  if (['housing', 'safelimit', 'diabetes'].indexOf(domain) < 0) domain = 'housing';
  var container = this.getQuestionContainer();
  Array.prototype.forEach.call(container.querySelectorAll('[data-screening-domain]'), function (item) {
    item.style.display = item.getAttribute('data-screening-domain') === domain ? '' : 'none';
  });
});"""


def domain_choice(housing: str, safelimit: str, diabetes: str) -> str:
    return (
        f'<span data-screening-domain="housing">{html.escape(housing)}</span>'
        f'<span data-screening-domain="safelimit" style="display:none">{html.escape(safelimit)}</span>'
        f'<span data-screening-domain="diabetes" style="display:none">{html.escape(diabetes)}</span>'
    )


def update_domain_screening(payload: dict[str, Any], attempt: int) -> None:
    payload["QuestionText"] = "Which pair are the two possible AI outputs in this task?"
    payload["QuestionDescription"] = (
        "Which pair are the two possible AI outputs in this task?"
        + (" (Attempt 2)" if attempt == 2 else "")
    )
    payload["Choices"] = {
        "1": {"Display": domain_choice(
            "Cheap or Expensive",
            "Above Limit or Below Limit",
            "Diabetes or No Diabetes",
        )},
        "2": {"Display": domain_choice(
            "Small or Large",
            "Safe Driver or Unsafe Driver",
            "High Glucose or Low Glucose",
        )},
        "3": {"Display": domain_choice(
            "Affordable or Unaffordable",
            "Drunk or Sober",
            "Healthy or Unhealthy",
        )},
        "4": {"Display": domain_choice(
            "Accepted or Rejected",
            "High Risk or Low Risk",
            "Treatment or No Treatment",
        )},
    }
    payload["QuestionJS"] = screening_javascript()


def set_intro_scripts(questions: dict[str, dict[str, Any]]) -> None:
    questions["QID11"]["QuestionJS"] = """Qualtrics.SurveyEngine.addOnload(function () {
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  var explanation = String(Qualtrics.SurveyEngine.getEmbeddedData('xaiType') || 'attribution').toLowerCase();
  var text = document.getElementById('cf-training-intro-copy');
  if (!text) return;
  var review = explanation === 'attribution'
    ? 'review the correct answer and the influence explanation'
    : explanation === 'counterfactual'
    ? 'review the correct answer and the counter-example explanation'
    : 'review the correct answer';
  var copy = {
    housing: 'Before moving to the final task, your task is to learn how the AI predicts Cheap or Expensive Houses. You will see <b>10 house profiles</b>. Predict Cheap or Expensive, then ',
    safelimit: 'Before moving to the final task, your task is to learn how the AI predicts Below or Above Limit. You will see <b>10 driver profiles</b>. Predict Above Limit or Below Limit, then ',
    diabetes: 'Before moving to the final task, your task is to learn how the AI predicts Diabetes or No Diabetes. You will see <b>10 health profiles</b>. Predict Diabetes or No Diabetes, then '
  };
  text.innerHTML = (copy[domain] || copy.housing) + review + '.';
});
Qualtrics.SurveyEngine.addOnReady(function () {});
Qualtrics.SurveyEngine.addOnUnload(function () {});"""
    questions["QID390"]["QuestionJS"] = """Qualtrics.SurveyEngine.addOnload(function () {
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  var text = document.getElementById('cf-test-intro-copy');
  if (!text) return;
  var copy = {
    housing: 'You will complete two sessions with ten profiles each. You will have to change the AI predictions from Cheap to Expensive or vice versa.',
    safelimit: 'You will complete two sessions with ten profiles each. You will have to change the AI predictions from Above Limit to Below Limit or vice versa.',
    diabetes: 'You will complete two sessions with ten profiles each. You will have to change the AI predictions from Diabetes to No Diabetes or vice versa.'
  };
  text.innerHTML = copy[domain] || copy.housing;
});
Qualtrics.SurveyEngine.addOnReady(function () {});
Qualtrics.SurveyEngine.addOnUnload(function () {});"""


def update_direction_introductions(questions: dict[str, dict[str, Any]]) -> None:
    additions = {
        "QID19": """
  <div data-domain="diabetes" hidden>
    <h2>Diabetes to No Diabetes</h2>
    <p>Each of the following ten people receives a <b>Diabetes</b> warning. What minimal changes should they make to receive a <b>No Diabetes</b> warning in the future?</p>
  </div>
""",
        "QID20": """
  <div data-domain="diabetes" hidden>
    <h2>No Diabetes to Diabetes</h2>
    <p>Each of the following ten people receives a <b>No Diabetes</b> warning. What minimal changes should they avoid that would make them receive a <b>Diabetes</b> warning in the future?</p>
  </div>
""",
    }
    for question_id, addition in additions.items():
        payload = questions[question_id]
        payload["QuestionText"] = insert_before(
            payload["QuestionText"], "</div>\n\n<style>", addition
        )
        payload["QuestionJS"] = payload["QuestionJS"].replace(
            'domain !== "housing" && domain !== "safelimit"',
            'domain !== "housing" && domain !== "safelimit" && domain !== "diabetes"',
        )


def update_test_prompts(questions: dict[str, dict[str, Any]]) -> None:
    prompts = {
        "QID376": """
<p class="cf-test-prompt" data-domain="diabetes" hidden>
  This person receives a <b>Diabetes</b> warning. To avoid such a warning in the future, what <b>minimal</b> changes should they make?
</p>
""",
        "QID379": """
<p class="cf-test-prompt" data-domain="diabetes" hidden>
  This person receives a <b>No Diabetes</b> warning. To avoid receiving a <b>Diabetes</b> warning in the future, describe what <b>minimal</b> changes they should avoid.
</p>
""",
    }
    marker = '  <iframe id="cf-case-frame"'
    for question_id, prompt in prompts.items():
        questions[question_id]["QuestionText"] = insert_before(
            questions[question_id]["QuestionText"], marker, prompt
        )


def duplicate_reference_question(
    document: dict[str, Any],
    source: dict[str, Any],
    new_id: str,
    export_tag: str,
    description: str,
) -> dict[str, Any]:
    element = copy.deepcopy(source)
    element["PrimaryAttribute"] = new_id
    element["SecondaryAttribute"] = description
    payload = element["Payload"]
    payload["QuestionID"] = new_id
    payload["DataExportTag"] = export_tag
    payload["QuestionDescription"] = description
    document["SurveyElements"].append(element)
    return payload


def update_attempt_two(
    document: dict[str, Any],
    elements: dict[str, dict[str, Any]],
) -> None:
    duplicate_reference_question(
        document,
        elements["QID271"],
        "QID34",
        "BasicInterfaceAttempt2",
        "Basic interface tutorial (Attempt 2)",
    )
    duplicate_reference_question(
        document,
        elements["QID9"],
        "QID35",
        "ExplanationInterfaceAttempt2",
        "AI explanation tutorial (Attempt 2)",
    )
    screening_block = next(block for block in blocks(document) if block["ID"] == SCREENING_BLOCK_ID)
    screening_block["BlockElements"] = [
        {"Type": "Question", "QuestionID": "QID33"},
        {"Type": "Question", "QuestionID": "QID34"},
        {"Type": "Question", "QuestionID": "QID27"},
        {"Type": "Question", "QuestionID": "QID28"},
        {"Type": "Page Break"},
        {"Type": "Question", "QuestionID": "QID35"},
        {"Type": "Question", "QuestionID": "QID29"},
        {"Type": "Question", "QuestionID": "QID30"},
        {"Type": "Question", "QuestionID": "QID31"},
        {"Type": "Question", "QuestionID": "QID32"},
    ]


def update_task_preview(payload: dict[str, Any], preview_id: int) -> None:
    payload["QuestionJS"] = payload["QuestionJS"].replace(
        "var previewIds = { housing: 14693, safelimit: 610 };",
        f"var previewIds = {{ housing: 14693, safelimit: 610, diabetes: {preview_id} }};",
    )


def update_question_count(document: dict[str, Any]) -> None:
    count = sum(
        element.get("Element") == "SQ" for element in document["SurveyElements"]
    )
    qc = next(
        element for element in document["SurveyElements"]
        if element.get("Element") == "QC"
    )
    qc["SecondaryAttribute"] = str(count)


def main() -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8-sig"))
    data = json.loads(STATIC_DATA.read_text(encoding="utf-8"))
    if set(data["datasets"]) != {"housing", "safelimit", "diabetes"}:
        raise RuntimeError("The static UI bundle must contain housing, safelimit, and diabetes")
    ids = case_ids(data)
    for domain, values in ids.items():
        if len(values["training"]) != 10:
            raise RuntimeError(f"{domain} must have 10 training cases")
        if any(len(values["test"][str(label)]) != 10 for label in (0, 1)):
            raise RuntimeError(f"{domain} must have 10 test cases per label")

    elements = question_elements(document)
    questions = {question_id: element["Payload"] for question_id, element in elements.items()}
    update_scenario(questions["QID280"])
    update_basic_tutorial(questions["QID271"], ids)
    _, preview_id = update_explanation_tutorial(questions["QID9"], data)
    update_domain_screening(questions["QID18"], attempt=1)
    update_domain_screening(questions["QID27"], attempt=2)
    elements["QID18"]["SecondaryAttribute"] = questions["QID18"]["QuestionDescription"]
    elements["QID27"]["SecondaryAttribute"] = questions["QID27"]["QuestionDescription"]
    set_intro_scripts(questions)
    update_direction_introductions(questions)
    update_test_prompts(questions)
    update_task_preview(questions["QID26"], preview_id)

    frame_js = FRAME_JS.read_text(encoding="utf-8")
    for question_id in ("QID12", "QID376", "QID379"):
        questions[question_id]["QuestionJS"] = frame_js

    update_attempt_two(document, elements)
    update_question_count(document)
    document["SurveyEntry"]["SurveyName"] = "Recourse v0.10"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Qualtrics-hosted study for housing, drink-driving, and diabetes warning domains; "
        "10 fixed training and 20 fixed test cases, with two-attempt screening."
    )
    OUTPUT_QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(OUTPUT_QSF)


if __name__ == "__main__":
    main()
