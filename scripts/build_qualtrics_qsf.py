"""Create the streamlined Housing / Drink Driving Qualtrics survey.

This intentionally reuses stable question and block IDs from the supplied QSF,
but replaces its active flow, random instance selection, and domain-specific
copy. The original source QSF is never modified.
"""

from __future__ import annotations

import argparse
import copy
import html
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
    loop_randomization: str = "None",
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
                "Randomization": loop_randomization,
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
    <div class="scenario-outcomes"><div><b>Cheap</b></div><div><b>Expensive</b></div></div>
    <h2>Information available to the AI</h2>
    <table class="domain-table"><tr><th>Attribute</th><th>Description</th><th>Typical range</th></tr>
      <tr><td>Living Area</td><td>Interior living area</td><td>800–4,320 sq ft</td></tr>
      <tr><td>Bedrooms</td><td>Number of bedrooms</td><td>2–5</td></tr>
      <tr><td>Bathrooms</td><td>Bathroom equivalent</td><td>1–3.75</td></tr>
      <tr><td>Floors</td><td>Number of floors</td><td>1–2.5</td></tr>
      <tr><td>Construction Grade</td><td>Overall construction and design quality</td><td>6–11</td></tr>
    </table>
    <div class="scenario-task"><h2>What you will do</h2><ol><li>During training, predict whether the AI will say <b>Cheap</b> or <b>Expensive</b>.</li><li data-explanation="attribution" hidden>Review the AI answer and an explanation showing the influence of the two most important attributes.</li><li data-explanation="counterfactual" hidden>Review the AI answer and a counter-example showing two attribute changes.</li><li data-explanation="none" hidden>Review the correct AI answer.</li><li>During testing, make the smallest changes that reverse the AI prediction.</li></ol></div>
  </section>
  <section class="scenario-card" data-domain="safelimit" hidden>
    <p class="scenario-eyebrow">DRINK-DRIVING DOMAIN</p>
    <h1>Understanding an AI drink-driving risk classification</h1>
    <p class="scenario-lead">Imagine that you are reviewing driver profiles. An AI uses five details to estimate whether the driver's blood-alcohol level is <b>Above Limit</b> or <b>Below Limit</b>, using 0.08 as the threshold.</p>
    <div class="scenario-outcomes"><div><b>Above Limit</b></div><div><b>Below Limit</b></div></div>
    <h2>Information available to the AI</h2>
    <table class="domain-table"><tr><th>Attribute</th><th>Description</th><th>Typical range/options</th></tr>
      <tr><td>Alcohol Units</td><td>Standard alcohol units consumed</td><td>1.496–9.5</td></tr>
      <tr><td>Weight</td><td>Body weight</td><td>46.788–102.408 kg</td></tr>
      <tr><td>Drinking Duration</td><td>Time spent drinking</td><td>15–290 minutes</td></tr>
      <tr><td>Gender</td><td>Gender used by the BAC calculation</td><td>Female / Male</td></tr>
      <tr><td>Stomach Fullness</td><td>Whether alcohol was consumed on an empty or full stomach</td><td>Empty / Full</td></tr>
    </table>
    <div class="scenario-task"><h2>What you will do</h2><ol><li>During training, predict whether the AI will say <b>Above Limit</b> or <b>Below Limit</b>.</li><li data-explanation="attribution" hidden>Review the AI answer and an explanation showing the influence of the two most important attributes.</li><li data-explanation="counterfactual" hidden>Review the AI answer and a counter-example showing two attribute changes.</li><li data-explanation="none" hidden>Review the correct AI answer.</li><li>During testing, make the smallest changes that reverse the AI prediction.</li></ol></div>
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
  var explanation = String(Qualtrics.SurveyEngine.getEmbeddedData('xaiType') || 'attribution').toLowerCase();
  if (!['attribution', 'counterfactual', 'none'].includes(explanation)) explanation = 'attribution';
  var root = document.getElementById('cf-scenario-root');
  if (!root) return;
  Array.prototype.forEach.call(root.querySelectorAll('[data-domain]'), function (panel) {
    panel.hidden = panel.getAttribute('data-domain') !== domain;
  });
  Array.prototype.forEach.call(root.querySelectorAll('[data-explanation]'), function (item) {
    item.hidden = item.getAttribute('data-explanation') !== explanation;
  });
});
""".strip()


def _tutorial_value(value: Any) -> str:
    if isinstance(value, float):
        rounded = round(value, 1)
        return str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"
    return str(value)


def _feature_index(payload: dict[str, Any], feature_name: str) -> int:
    normalized = feature_name.lower()
    return next(
        (
            index
            for index, name in enumerate(payload["feature_names"])
            if str(name).lower() == normalized
        ),
        -1,
    )


def _feature_range_example(payload: dict[str, Any], feature_name: str) -> str:
    index = _feature_index(payload, feature_name)
    if index < 0:
        return ""
    feature_range = payload["feature_ranges"][index]
    value = payload["feature_values"][index]
    minimum, maximum = [float(item) for item in feature_range[:2]]
    ratio = (float(value) - minimum) / (maximum - minimum)
    position = (
        "well over half"
        if ratio >= 0.7
        else "just over half"
        if ratio >= 0.52
        else "about half"
        if ratio >= 0.45
        else "under half"
    )
    return (
        f"for {html.escape(feature_name)}, the lowest value is "
        f"{html.escape(_tutorial_value(minimum))} and the highest is "
        f"{html.escape(_tutorial_value(maximum))}, so "
        f"{html.escape(_tutorial_value(value))} is {position} of the bar"
    )


def basic_tutorial_html(data: dict[str, Any]) -> str:
    sections = []
    settings = {
        "housing": {
            "subject": "house",
            "feature": "Living Area",
            "prediction": "The selected box shows the AI's house-price <b>prediction</b>.",
        },
        "safelimit": {
            "subject": "driver",
            "feature": "Alcohol Units",
            "prediction": "The selected box shows the AI's blood-alcohol-limit <b>prediction</b>.",
        },
    }
    for domain, setting in settings.items():
        payload = data["datasets"][domain]["training_pool"][0]
        hidden = "" if domain == "housing" else " hidden"
        example = _feature_range_example(payload, setting["feature"])
        sections.append(f"""
<section class="tutorial-copy" data-domain="{domain}"{hidden}>
  <p>Each {setting["subject"]} profile is shown using the following interface.</p>
  <ol class="tutorial-bullets">
    <li>The five <b>attributes</b> describing the {setting["subject"]}.</li>
    <li>The <b>values</b> of each attribute.</li>
    <li>Bars indicating how <b>low/high</b> the value is for that attribute (for example, {example}).</li>
    <li>{setting["prediction"]}</li>
  </ol>
</section>""".strip())
    return f"""
<div id="cf-basic-tutorial-root" class="tutorial-root">
  <h1>Basic Interface</h1>
  <div class="tutorial-two-column">
    <div>{''.join(sections)}</div>
    <div class="tutorial-preview"><iframe id="cf-basic-tutorial-frame" title="Basic interface example"></iframe></div>
  </div>
</div>
{tutorial_css()}
""".strip()


def _select_attribution_tutorial_case(bundle: dict[str, Any]) -> dict[str, Any]:
    pool = bundle["training_pool"]
    for candidate in pool:
        shown = candidate["attribution"].get("shown_feature_indices", [])
        values = [float(candidate["attribution"]["values"][index]) for index in shown]
        if any(value > 0 for value in values) and any(value < 0 for value in values):
            return candidate
    return pool[0]


def _select_counterfactual_tutorial_case(bundle: dict[str, Any]) -> dict[str, Any]:
    pool = bundle["training_pool"]
    for candidate in pool:
        changes = []
        for name in candidate["counterfactual"].get("selected_feature_names", []):
            index = _feature_index(candidate, name)
            if index < 0 or candidate["feature_types"][index] == "categorical":
                continue
            delta = (
                float(candidate["counterfactual"]["feature_values"][index])
                - float(candidate["feature_values"][index])
            )
            changes.append(delta)
        if any(value > 0 for value in changes) and any(value < 0 for value in changes):
            return candidate
    return next(
        (
            candidate
            for candidate in pool
            if candidate["counterfactual"]["prediction"]["value"]
            != candidate["prediction"]["value"]
        ),
        pool[0],
    )


def _attribution_tutorial_copy(payload: dict[str, Any], domain: str) -> str:
    attribution = payload["attribution"]
    total = sum(abs(float(value)) for value in attribution["values"]) or 1.0
    examples = []
    for index in attribution.get("shown_feature_indices", []):
        value = float(attribution["values"][index])
        color = "tutorial-red" if value < 0 else "tutorial-blue"
        sign = "-" if value < 0 else "+"
        percent = round((abs(value) / total) * 100)
        examples.append(
            f'{html.escape(payload["feature_names"][index])}, '
            f'<span class="{color}">{sign}{percent}%</span>'
        )
    decision = "price prediction" if domain == "housing" else "blood-alcohol-limit prediction"
    left = html.escape(str(attribution["direction_labels"]["left"]).lower())
    right = html.escape(str(attribution["direction_labels"]["right"]).lower())
    joined = " and ".join(examples)
    return f"""
<p>To learn how the AI predicts, you will see an explanation for the AI's {decision}. This explanation shows the <b>two most important attributes</b> for that profile.</p>
<p>The numbered explanation interface shows:</p>
<ol class="tutorial-bullets tutorial-bullets-compact">
  <li>The influence of the two most important attributes ({joined}). The higher the number, the stronger the influence.
    <ul><li><span class="tutorial-red">Red bars</span> contribute towards a <b>{left}</b> decision.</li><li><span class="tutorial-blue">Blue bars</span> contribute towards a <b>{right}</b> decision.</li></ul>
  </li>
  <li>A sentence describing the {decision} and the influences.</li>
</ol>
""".strip()


def _counterfactual_tutorial_copy(payload: dict[str, Any], domain: str) -> str:
    details = []
    for name in payload["counterfactual"].get("selected_feature_names", []):
        index = _feature_index(payload, name)
        if index < 0:
            continue
        original = payload["feature_values"][index]
        updated = payload["counterfactual"]["feature_values"][index]
        if payload["feature_types"][index] == "categorical":
            details.append(
                f'<span class="tutorial-blue">Blue markers</span> show that '
                f'{html.escape(name)} changes from {html.escape(_tutorial_value(original))} '
                f'to {html.escape(_tutorial_value(updated))}.'
            )
        else:
            delta = float(updated) - float(original)
            color = "tutorial-red" if delta < 0 else "tutorial-blue"
            direction = "decreases" if delta < 0 else "increases"
            details.append(
                f'<span class="{color}">{html.escape(name)}</span> {direction} '
                f'from {html.escape(_tutorial_value(original))} to '
                f'{html.escape(_tutorial_value(updated))}.'
            )
    decision = "price prediction" if domain == "housing" else "blood-alcohol-limit prediction"
    detail_items = "".join(f"<li>{item}</li>" for item in details)
    return f"""
<p>To learn how the AI predicts, you will see an explanation for the AI's {decision}. This explanation shows a <b>counter-example</b>, where <b>two attributes are changed</b> to alter the {decision}.</p>
<p>The numbered explanation interface shows:</p>
<ol class="tutorial-bullets tutorial-bullets-compact">
  <li>The changes in the two attributes in the counter-example.<ul>{detail_items}</ul></li>
  <li>A sentence describing the {decision} and the effect of the changes.</li>
</ol>
""".strip()


def explanation_tutorial_html(data: dict[str, Any]) -> str:
    sections = []
    for domain in ("housing", "safelimit"):
        bundle = data["datasets"][domain]
        attribution_case = _select_attribution_tutorial_case(bundle)
        counterfactual_case = _select_counterfactual_tutorial_case(bundle)
        for explanation, payload, copy_html in (
            (
                "attribution",
                attribution_case,
                _attribution_tutorial_copy(attribution_case, domain),
            ),
            (
                "counterfactual",
                counterfactual_case,
                _counterfactual_tutorial_copy(counterfactual_case, domain),
            ),
        ):
            hidden = "" if domain == "housing" and explanation == "attribution" else " hidden"
            sections.append(f"""
<section class="tutorial-copy" data-domain="{domain}" data-explanation="{explanation}" data-instance-id="{payload['instance_id']}"{hidden}>
  {copy_html}
</section>""".strip())
    return f"""
<div id="cf-explanation-tutorial-root" class="tutorial-root">
  <h1>AI Explanation</h1>
  <div class="tutorial-two-column">
    <div>{''.join(sections)}</div>
    <div class="tutorial-preview"><iframe id="cf-explanation-tutorial-frame" title="Explanation interface example"></iframe></div>
  </div>
</div>
{tutorial_css()}
""".strip()


def tutorial_css() -> str:
    return """
<style>
  .tutorial-root{font-size:18px;line-height:1.3;max-width:1100px;margin:0 auto}.tutorial-root h1{font-size:30px;margin:0 0 22px}.tutorial-two-column{display:grid;grid-template-columns:minmax(280px,.9fr) minmax(430px,1.25fr);gap:28px;align-items:start}.tutorial-copy p{margin:0 0 18px}.tutorial-bullets{margin:12px 0 0;padding-left:28px}.tutorial-bullets li{margin-bottom:12px}.tutorial-bullets ul{margin-top:8px;padding-left:24px}.tutorial-bullets-compact{margin-top:8px}.tutorial-preview iframe{width:100%;min-height:420px;border:0}.tutorial-red{color:#ea3335}.tutorial-blue{color:#3c88e8}@media(max-width:800px){.tutorial-two-column{grid-template-columns:1fr}.tutorial-preview iframe{min-height:460px}}
</style>
""".strip()


def tutorial_js(training_ids: dict[str, list[int]]) -> str:
    ids = json.dumps(training_ids, separators=(",", ":"))
    return f"""
Qualtrics.SurveyEngine.addOnload(function () {{
  var ids = {ids};
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  if (!ids[domain]) domain = 'housing';
  var root = document.getElementById('cf-basic-tutorial-root');
  if (!root) return;
  Array.prototype.forEach.call(root.querySelectorAll('[data-domain]'), function (panel) {{
    panel.hidden = panel.getAttribute('data-domain') !== domain;
  }});
  var base = String(Qualtrics.SurveyEngine.getEmbeddedData('ui_base_url') || 'https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html');
  var params = new URLSearchParams({{
    appId: domain,
    xaiType: 'none',
    split: 'train',
    instanceId: String(ids[domain][0]),
    showPrediction: '1',
    counterfactualSimulation: '0',
    tutorialCallouts: 'basic'
  }});
  var frame = document.getElementById('cf-basic-tutorial-frame');
  frame.src = base + (base.indexOf('?') >= 0 ? '&' : '?') + params.toString();
  window.addEventListener('message', function (event) {{
    if (event.source !== frame.contentWindow || !event.data || event.data.type !== 'counterfactual-ui:iframe-height') return;
    frame.style.height = Math.max(320, Math.min(850, Number(event.data.height) || 0)) + 'px';
  }}, false);
}});
""".strip()


def explanation_tutorial_js(data: dict[str, Any]) -> str:
    tutorial_ids = {}
    for domain in ("housing", "safelimit"):
        bundle = data["datasets"][domain]
        tutorial_ids[domain] = {
            "attribution": _select_attribution_tutorial_case(bundle)["instance_id"],
            "counterfactual": _select_counterfactual_tutorial_case(bundle)["instance_id"],
        }
    ids = json.dumps(tutorial_ids, separators=(",", ":"))
    return f"""
Qualtrics.SurveyEngine.addOnload(function () {{
  var root = document.getElementById('cf-explanation-tutorial-root');
  if (!root) return;
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  var explanation = String(Qualtrics.SurveyEngine.getEmbeddedData('xaiType') || 'attribution').toLowerCase();
  var ids = {ids};
  if (!ids[domain]) domain = 'housing';
  if (!['attribution', 'counterfactual'].includes(explanation)) {{
    this.getQuestionContainer().style.display = 'none';
    return;
  }}
  Array.prototype.forEach.call(root.querySelectorAll('[data-domain][data-explanation]'), function (panel) {{
    panel.hidden = panel.getAttribute('data-domain') !== domain ||
      panel.getAttribute('data-explanation') !== explanation;
  }});
  var base = String(Qualtrics.SurveyEngine.getEmbeddedData('ui_base_url') || 'https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html');
  var params = new URLSearchParams({{
    appId: domain,
    xaiType: explanation,
    split: 'train',
    instanceId: String(ids[domain][explanation]),
    showPrediction: '1',
    counterfactualSimulation: '0',
    tutorialCallouts: 'explanation'
  }});
  var frame = document.getElementById('cf-explanation-tutorial-frame');
  frame.src = base + (base.indexOf('?') >= 0 ? '&' : '?') + params.toString();
  window.addEventListener('message', function (event) {{
    if (event.source !== frame.contentWindow || !event.data || event.data.type !== 'counterfactual-ui:iframe-height') return;
    frame.style.height = Math.max(360, Math.min(900, Number(event.data.height) || 0)) + 'px';
  }}, false);
}});
""".strip()


def intro_js(phase: str) -> str:
    return f"""
Qualtrics.SurveyEngine.addOnload(function () {{
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || 'housing').toLowerCase();
  var explanation = String(Qualtrics.SurveyEngine.getEmbeddedData('xaiType') || 'attribution').toLowerCase();
  var text = document.getElementById('cf-{phase}-intro-copy');
  if (!text) return;
  if ('{phase}' === 'training') {{
    var review = explanation === 'attribution'
      ? 'review the correct answer and the feature-influence explanation'
      : explanation === 'counterfactual'
      ? 'review the correct answer and the counter-example explanation'
      : 'review the correct answer';
    text.innerHTML = domain === 'safelimit'
      ? 'You will see <b>10 driver profiles</b>. Predict Above Limit or Below Limit, then ' + review + '.'
      : 'You will see <b>10 house profiles</b>. Predict Cheap or Expensive, then ' + review + '.';
  }} else {{
    text.innerHTML = domain === 'safelimit'
      ? 'You will complete two ten-profile phases: Above Limit to Below Limit and Below Limit to Above Limit. Qualtrics randomizes which phase comes first.'
      : 'You will complete two ten-profile phases: Cheap to Expensive and Expensive to Cheap. Qualtrics randomizes which phase comes first.';
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
        "<p>You will complete a short tutorial, 10 training cases, 20 testing cases, "
        "and demographic questions. The task concerns either housing prices or "
        "drink-driving risk.</p>"
    )
    questions["QID286"]["QuestionJS"] = ""
    questions["QID280"]["QuestionText"] = scenario_html()
    questions["QID280"]["QuestionJS"] = SCENARIO_JS
    questions["QID280"]["QuestionDescription"] = "Application scenario"
    questions["QID280"].pop("DisplayLogic", None)
    questions["QID271"]["QuestionText"] = basic_tutorial_html(data)
    questions["QID271"]["QuestionJS"] = tutorial_js(training_ids)
    questions["QID271"]["QuestionDescription"] = "Basic interface tutorial"

    questions["QID9"].clear()
    questions["QID9"].update(copy.deepcopy(questions["QID271"]))
    questions["QID9"]["QuestionID"] = "QID9"
    questions["QID9"]["DataExportTag"] = "ExplanationTutorial"
    questions["QID9"]["QuestionDescription"] = "Explanation interface tutorial"
    questions["QID9"]["QuestionText"] = explanation_tutorial_html(data)
    questions["QID9"]["QuestionJS"] = explanation_tutorial_js(data)
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
        "QID286", "QID5", "QID280", "QID271", "QID9", "QID11", "QID12",
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
        block(
            SCENARIO_BLOCK,
            "Domain and Interface Tutorials",
            ["QID280", "QID271", "QID9"],
        ),
        block(TRAINING_INTRO_BLOCK, "Training Introduction", ["QID11"]),
        block(
            TRAINING_BLOCK,
            "Training: 10 Fixed Cases",
            ["QID12"],
            loop_count=10,
            loop_randomization="All",
        ),
        block(TEST_INTRO_BLOCK, "Testing Introduction", ["QID390"]),
        block(
            TEST_LABEL_0_BLOCK,
            "Testing: Label 0 to Label 1 (10 Cases)",
            ["QID376"],
            loop_count=10,
            loop_randomization="All",
        ),
        block(
            TEST_LABEL_1_BLOCK,
            "Testing: Label 1 to Label 0 (10 Cases)",
            ["QID379"],
            loop_count=10,
            loop_randomization="All",
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
        "10 fixed training and 20 fixed test cases."
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
