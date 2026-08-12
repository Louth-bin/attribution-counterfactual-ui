"""Create the streamlined Housing / Drink Driving Qualtrics survey.

This intentionally reuses stable question and block IDs from the supplied QSF,
but replaces its active flow, random instance selection, and domain-specific
copy. The original source QSF is never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads" / "Summative_Study_-_Global_Explanation_-_wine_quality.qsf"
DEFAULT_OUTPUT = REPO_ROOT / "qualtrics" / "UPLOAD_THIS_Qualtrics_Starter.qsf"
FRAME_JS = REPO_ROOT / "qualtrics" / "qualtrics-frame.js"
STATIC_DATA = REPO_ROOT / "static" / "experiment-data.json"

INTRO_BLOCK = "BL_5cCx0dF9dBRyR4W"
CONSENT_BLOCK = "BL_bOEr2QTL2wESbqu"
SCENARIO_BLOCK = "BL_eXxzjmRQ8gb7FGu"
TRAINING_INTRO_BLOCK = "BL_aUXleLP5DOWgqR8"
TRAINING_BLOCK = "BL_3fJJUKH6EXaeRIq"
TEST_INTRO_BLOCK = "BL_6S8eO7ATYBdYM3I"
TEST_LABEL_0_BLOCK = "BL_5aT4RYQul04UaHA"
TEST_LABEL_1_BLOCK = "BL_3W1PIk3tbtEX0BE"
DEMOGRAPHICS_BLOCK = "BL_d07rL5ybaNgQsMm"
TRASH_BLOCK = "BL_0j1M5799dsLpL5c"


def question_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }


def block(
    block_id: str,
    description: str,
    question_ids: list[str],
    *,
    default: bool = False,
    trash: bool = False,
    loop_count: int = 0,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "BlockLocking": "false",
        "RandomizeQuestions": "false",
        "BlockVisibility": "Collapsed",
    }
    if loop_count:
        options.update({
            "Looping": "Static",
            "LoopingOptions": {
                "Static": {
                    str(index + 1): {"1": str(index)} for index in range(loop_count)
                },
                "Randomization": "None",
            },
        })
    return {
        "Type": "Trash" if trash else ("Default" if default else "Standard"),
        "SubType": "",
        "Description": description,
        "ID": block_id,
        "BlockElements": [
            {"Type": "Question", "QuestionID": question_id}
            for question_id in question_ids
        ],
        "Options": options,
    }


def embedded_field(field: str, value: str = "") -> dict[str, Any]:
    item: dict[str, Any] = {
        "Description": field,
        "Type": "Custom",
        "Field": field,
        "VariableType": "String",
        "DataVisibility": [],
        "AnalyzeText": False,
    }
    if value:
        item["Value"] = value
    return item


def embedded_flow(flow_id: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {"Type": "EmbeddedData", "FlowID": flow_id, "EmbeddedData": fields}


def standard_flow(flow_id: str, block_id: str) -> dict[str, Any]:
    return {"Type": "Standard", "ID": block_id, "FlowID": flow_id, "Autofill": []}


def randomized_block_order(
    flow_id: str,
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    """Present every child once, in random order."""
    return {
        "Type": "BlockRandomizer",
        "FlowID": flow_id,
        "SubSet": len(children),
        "Flow": children,
    }


def scenario_html() -> str:
    return """
<div id="cf-scenario-root" class="scenario-root">
  <section class="scenario-card" data-domain="housing">
    <p class="scenario-eyebrow">HOUSING DOMAIN</p>
    <h1>Understanding an AI house-price classification</h1>
    <p class="scenario-lead">Imagine that you are reviewing residential property listings. An AI examines five details about each property and classifies the house as <b>Cheap</b> or <b>Expensive</b>.</p>
    <div class="scenario-outcomes"><div><b>Cheap</b><small>AI prediction 0</small></div><div><b>Expensive</b><small>AI prediction 1</small></div></div>
    <h2>Information available to the AI</h2>
    <table class="domain-table"><tr><th>Attribute</th><th>Description</th><th>Typical range</th></tr>
      <tr><td>Living Area</td><td>Interior living area</td><td>800–4,320 sq ft</td></tr>
      <tr><td>Bedrooms</td><td>Number of bedrooms</td><td>2–5</td></tr>
      <tr><td>Bathrooms</td><td>Bathroom equivalent</td><td>1–3.75</td></tr>
      <tr><td>Floors</td><td>Number of floors</td><td>1–2.5</td></tr>
      <tr><td>Construction Grade</td><td>Overall construction and design quality</td><td>6–11</td></tr>
    </table>
    <div class="scenario-task"><h2>What you will do</h2><ol><li>During training, predict whether the AI will say <b>Cheap</b> or <b>Expensive</b>.</li><li>Review the AI answer and the assigned explanation.</li><li>During testing, make the smallest changes that reverse the AI prediction.</li></ol></div>
  </section>
  <section class="scenario-card" data-domain="safelimit" hidden>
    <p class="scenario-eyebrow">DRINK-DRIVING DOMAIN</p>
    <h1>Understanding an AI drink-driving risk classification</h1>
    <p class="scenario-lead">Imagine that you are reviewing driver profiles. An AI uses five details to estimate whether the driver's blood-alcohol level is <b>Above Limit</b> or <b>Below Limit</b>, using 0.08 as the threshold.</p>
    <div class="scenario-outcomes"><div><b>Above Limit</b><small>AI prediction 0</small></div><div><b>Below Limit</b><small>AI prediction 1</small></div></div>
    <h2>Information available to the AI</h2>
    <table class="domain-table"><tr><th>Attribute</th><th>Description</th><th>Typical range/options</th></tr>
      <tr><td>Alcohol Units</td><td>Standard alcohol units consumed</td><td>1.496–9.5</td></tr>
      <tr><td>Weight</td><td>Body weight</td><td>46.788–102.408 kg</td></tr>
      <tr><td>Drinking Duration</td><td>Time spent drinking</td><td>15–290 minutes</td></tr>
      <tr><td>Gender</td><td>Gender used by the BAC calculation</td><td>Female / Male</td></tr>
      <tr><td>Stomach Fullness</td><td>Whether alcohol was consumed on an empty or full stomach</td><td>Empty / Full</td></tr>
    </table>
    <div class="scenario-task"><h2>What you will do</h2><ol><li>During training, predict whether the AI will say <b>Above Limit</b> or <b>Below Limit</b>.</li><li>Review the AI answer and the assigned explanation.</li><li>During testing, make the smallest changes that reverse the AI prediction.</li></ol></div>
    <p class="scenario-note">The profiles and predictions are for this research task only and should not be used to decide whether it is safe to drive.</p>
  </section>
</div>
<style>
  .scenario-root{font-size:17px;max-width:980px;margin:0 auto;color:#18222d;line-height:1.55}.scenario-card{border:1px solid #d8e0e8;border-radius:12px;padding:28px;background:#fff;box-shadow:0 4px 18px rgba(36,52,71,.08)}.scenario-card h1{font-size:28px;line-height:1.2;margin:4px 0 14px}.scenario-card h2{font-size:20px;margin:26px 0 10px}.scenario-eyebrow{margin:0;color:#376b91;font-size:13px;font-weight:700;letter-spacing:.08em}.scenario-lead{font-size:18px;margin:0 0 20px}.scenario-outcomes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:18px 0 24px}.scenario-outcomes div{display:flex;flex-direction:column;padding:13px;border:1px solid #d8e0e8;border-radius:8px;background:#f8fafc}.scenario-outcomes small{color:#617181}.domain-table{width:100%;border-collapse:collapse}.domain-table th,.domain-table td{padding:10px;border:1px solid #d8dee5;text-align:left}.domain-table th{background:#243447;color:#fff}.scenario-task{margin-top:24px;padding:2px 18px 12px;border-left:4px solid #4387b8;background:#f3f8fc}.scenario-task h2{margin-top:14px}.scenario-task li{margin:7px 0}.scenario-note{margin:18px 0 0;padding:10px 12px;background:#fff6df;border-radius:6px;font-size:14px}@media(max-width:650px){.scenario-card{padding:18px}.scenario-outcomes{grid-template-columns:1fr}.domain-table{font-size:14px}}
</style>
""".strip()


SCENARIO_JS = """
Qualtrics.SurveyEngine.addOnload(function () {
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  var root = document.getElementById('cf-scenario-root');
  if (!root) return;
  Array.prototype.forEach.call(root.querySelectorAll('[data-domain]'), function (panel) {
    panel.hidden = panel.getAttribute('data-domain') !== domain;
  });
});
""".strip()


def tutorial_js(training_ids: dict[str, list[int]]) -> str:
    ids = json.dumps(training_ids, separators=(",", ":"))
    return f"""
Qualtrics.SurveyEngine.addOnload(function () {{
  var ids = {ids};
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  if (!ids[domain]) domain = 'housing';
  var base = String(Qualtrics.SurveyEngine.getEmbeddedData('ui_base_url') || 'https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html');
  var params = new URLSearchParams({{
    appId: domain,
    xaiType: 'none',
    split: 'train',
    instanceId: String(ids[domain][0]),
    showPrediction: '1',
    counterfactualSimulation: '0'
  }});
  var frame = document.getElementById('cf-tutorial-frame');
  frame.src = base + (base.indexOf('?') >= 0 ? '&' : '?') + params.toString();
  window.addEventListener('message', function (event) {{
    if (event.source !== frame.contentWindow || !event.data || event.data.type !== 'counterfactual-ui:iframe-height') return;
    frame.style.height = Math.max(320, Math.min(850, Number(event.data.height) || 0)) + 'px';
  }}, false);
}});
Qualtrics.SurveyEngine.addOnReady(function () {{}});
Qualtrics.SurveyEngine.addOnUnload(function () {{}});
""".strip()


def intro_js(phase: str) -> str:
    return f"""
Qualtrics.SurveyEngine.addOnload(function () {{
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  var text = document.getElementById('cf-{phase}-intro-copy');
  if (!text) return;
  if ('{phase}' === 'training') {{
    text.innerHTML = domain === 'safelimit'
      ? 'You will see <b>10 driver profiles</b>. Predict Above Limit or Below Limit, then review the answer and explanation.'
      : 'You will see <b>10 house profiles</b>. Predict Cheap or Expensive, then review the answer and explanation.';
  }} else {{
    text.innerHTML = domain === 'safelimit'
      ? 'You will complete two five-profile phases: Above Limit to Below Limit and Below Limit to Above Limit. Qualtrics randomizes which phase comes first.'
      : 'You will complete two five-profile phases: Cheap to Expensive and Expensive to Cheap. Qualtrics randomizes which phase comes first.';
  }}
}});
Qualtrics.SurveyEngine.addOnReady(function () {{}});
Qualtrics.SurveyEngine.addOnUnload(function () {{}});
""".strip()


def replace_active_design(document: dict[str, Any]) -> None:
    questions = question_map(document)
    data = json.loads(STATIC_DATA.read_text(encoding="utf-8"))
    training_ids = {
        name: [case["instance_id"] for case in bundle["training_pool"]]
        for name, bundle in data["datasets"].items()
    }

    questions["QID286"]["QuestionText"] = (
        "<h1>Research Study on AI Understanding</h1>"
        "<p>You will complete a short tutorial, 10 training cases, 10 testing cases, "
        "and demographic questions. The task concerns either housing prices or "
        "drink-driving risk.</p>"
    )
    questions["QID286"]["QuestionJS"] = ""
    questions["QID280"]["QuestionText"] = scenario_html()
    questions["QID280"]["QuestionJS"] = SCENARIO_JS
    questions["QID280"]["QuestionDescription"] = "Application scenario"
    questions["QID280"].pop("DisplayLogic", None)
    questions["QID271"]["QuestionText"] = (
        "<div style=\"font-size:18px\"><h2>Tutorial: basic interface</h2>"
        "<p>The frame shows five attributes, their values and relative low/high positions, "
        "followed by the AI prediction.</p>"
        "<iframe id=\"cf-tutorial-frame\" title=\"Interface tutorial\" "
        "style=\"width:100%;height:420px;border:0\"></iframe></div>"
    )
    questions["QID271"]["QuestionJS"] = tutorial_js(training_ids)
    questions["QID11"]["QuestionText"] = (
        "<h1>Training</h1><p id=\"cf-training-intro-copy\"></p>"
    )
    questions["QID11"]["QuestionJS"] = intro_js("training")
    questions["QID390"]["QuestionText"] = (
        "<h1>Testing</h1><p id=\"cf-test-intro-copy\"></p>"
    )
    questions["QID390"]["QuestionJS"] = intro_js("test")

    training_html = """
<div id="cf-study-root" data-phase="training" class="cf-study-root">
  <h2 id="cf-case-title">Training case</h2><p>Study the profile, then predict the AI output.</p>
  <iframe id="cf-case-frame" title="Training profile" style="width:100%;height:420px;border:0"></iframe>
  <div id="cf-training-answer" class="cf-training-answer" hidden></div>
  <p id="cf-status" class="cf-status" aria-live="polite"></p>
</div><style>.cf-study-root{font-size:18px;max-width:1050px;margin:0 auto}.cf-training-answer{display:flex;gap:12px;justify-content:center;margin:18px 0}.cf-answer-button{border:1px solid #243447;border-radius:6px;background:#fff;padding:11px 22px;cursor:pointer;font-size:17px}.cf-answer-button:disabled{cursor:default;opacity:.7}.cf-status{min-height:1.5em;font-weight:600}.cf-correct{color:#217a45}.cf-incorrect{color:#b33a3a}</style>
""".strip()

    def test_html(source_label: int) -> str:
        prompts = {
            0: {
                "housing": (
                    "This house is predicted as <b>Cheap</b>. Use the Changes "
                    "column to make the smallest changes that make the AI "
                    "predict <b>Expensive</b>."
                ),
                "safelimit": (
                    "This driver is predicted as <b>Above Limit</b>. Use the "
                    "Changes column to make the smallest changes that make the "
                    "AI predict <b>Below Limit</b>."
                ),
            },
            1: {
                "housing": (
                    "This house is predicted as <b>Expensive</b>. Use the Changes "
                    "column to make the smallest changes that make the AI "
                    "predict <b>Cheap</b>."
                ),
                "safelimit": (
                    "This driver is predicted as <b>Below Limit</b>. Use the "
                    "Changes column to make the smallest changes that make the "
                    "AI predict <b>Above Limit</b>."
                ),
            },
        }
        return f"""
<div id="cf-study-root" data-phase="test" data-test-label="{source_label}" class="cf-study-root">
  <h2 id="cf-case-title">Testing case</h2>
  <!-- Edit these two prompts directly in the Qualtrics question HTML. -->
  <p class="cf-test-prompt" data-domain="housing" hidden>{prompts[source_label]["housing"]}</p>
  <p class="cf-test-prompt" data-domain="safelimit" hidden>{prompts[source_label]["safelimit"]}</p>
  <iframe id="cf-case-frame" title="Testing profile" style="width:100%;height:650px;border:0"></iframe>
  <p id="cf-status" class="cf-status" aria-live="polite"></p>
</div><style>.cf-study-root{{font-size:18px;max-width:1050px;margin:0 auto}}.cf-test-prompt{{padding:12px 14px;border-left:4px solid #243447;background:#f5f7f9}}.cf-status{{min-height:1.5em;font-weight:600}}</style>
""".strip()
    frame_js = FRAME_JS.read_text(encoding="utf-8")
    questions["QID12"]["QuestionText"] = training_html
    questions["QID12"]["QuestionJS"] = frame_js

    # Reuse an unused question ID for the second testing direction while
    # preserving the source survey's valid Qualtrics identifiers.
    questions["QID379"].clear()
    questions["QID379"].update(copy.deepcopy(questions["QID376"]))
    questions["QID379"]["QuestionID"] = "QID379"
    questions["QID379"]["DataExportTag"] = "TestDirection1"
    questions["QID376"]["DataExportTag"] = "TestDirection0"
    questions["QID376"]["QuestionText"] = test_html(0)
    questions["QID376"]["QuestionJS"] = frame_js
    questions["QID379"]["QuestionText"] = test_html(1)
    questions["QID379"]["QuestionJS"] = frame_js

    active_questions = {
        "QID286", "QID5", "QID280", "QID271", "QID11", "QID12",
        "QID390", "QID376", "QID379", "QID266", "QID71", "QID70", "QID72",
        "QID73", "QID74", "QID75",
    }
    # Remove the dozens of unused questions inherited from the old experiment.
    # Keeping an empty Trash block preserves the conventional QSF block layout.
    document["SurveyElements"] = [
        element
        for element in document["SurveyElements"]
        if element.get("Element") != "SQ"
        or element.get("PrimaryAttribute") in active_questions
    ]
    for element in document["SurveyElements"]:
        if element.get("Element") == "QC":
            element["SecondaryAttribute"] = str(len(active_questions))

    blocks = [
        block(INTRO_BLOCK, "Introduction", ["QID286"], default=True),
        block(TRASH_BLOCK, "Trash / Unused Questions", [], trash=True),
        block(CONSENT_BLOCK, "Consent", ["QID5"]),
        block(SCENARIO_BLOCK, "Domain and Basic Interface", ["QID280", "QID271"]),
        block(TRAINING_INTRO_BLOCK, "Training Introduction", ["QID11"]),
        block(TRAINING_BLOCK, "Training: 10 Fixed Cases", ["QID12"], loop_count=10),
        block(TEST_INTRO_BLOCK, "Testing Introduction", ["QID390"]),
        block(
            TEST_LABEL_0_BLOCK,
            "Testing: Label 0 to Label 1 (5 Cases)",
            ["QID376"],
            loop_count=5,
        ),
        block(
            TEST_LABEL_1_BLOCK,
            "Testing: Label 1 to Label 0 (5 Cases)",
            ["QID379"],
            loop_count=5,
        ),
        block(
            DEMOGRAPHICS_BLOCK,
            "Demographic Questions",
            ["QID266", "QID71", "QID70", "QID72", "QID73", "QID74", "QID75"],
        ),
    ]
    block_element = next(
        element for element in document["SurveyElements"] if element.get("Element") == "BL"
    )
    block_element["Payload"] = {str(index): value for index, value in enumerate(blocks)}

    flow = {
        "Type": "Root",
        "FlowID": "FL_1",
        "Flow": [
            embedded_flow("FL_2", [
                embedded_field("ui_base_url", "https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html"),
                embedded_field("xaiType", "attribution"),
                embedded_field("training_log_json", "[]"),
                embedded_field("testing_log_json", "[]"),
            ]),
            embedded_flow("FL_3", [embedded_field("appId", "housing")]),
            standard_flow("FL_6", INTRO_BLOCK),
            standard_flow("FL_7", CONSENT_BLOCK),
            standard_flow("FL_8", SCENARIO_BLOCK),
            standard_flow("FL_9", TRAINING_INTRO_BLOCK),
            standard_flow("FL_10", TRAINING_BLOCK),
            standard_flow("FL_11", TEST_INTRO_BLOCK),
            randomized_block_order("FL_12", [
                standard_flow("FL_13", TEST_LABEL_0_BLOCK),
                standard_flow("FL_14", TEST_LABEL_1_BLOCK),
            ]),
            standard_flow("FL_15", DEMOGRAPHICS_BLOCK),
        ],
        "Properties": {"Count": 15},
    }
    flow_element = next(
        element for element in document["SurveyElements"] if element.get("Element") == "FL"
    )
    flow_element["Payload"] = flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = json.loads(args.source.read_text(encoding="utf-8-sig"))
    document["SurveyEntry"]["SurveyName"] = "Summative Study - Housing and Drink Driving"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Qualtrics-hosted study using the basic counterfactual UI frame; "
        "10 fixed training and 10 fixed test cases."
    )
    replace_active_design(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
