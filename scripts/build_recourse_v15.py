"""Build the discrete-LIME, age-cluster Recourse v1.5 survey."""

from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_QSF = ROOT / "qualtrics" / "Recourse_v1.4.qsf"
OUTPUT_QSF = ROOT / "qualtrics" / "Recourse_v1.5.qsf"
SELECTION = ROOT / "outputs" / "v15-discrete-lime-age" / "selected_clusters.json"
MODEL_PATH = ROOT / "analysis" / "diabetes_mlp_regularized.joblib"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
ANALYSIS_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_discrete_age.json"
MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.5-discrete-age.json"
FRAME_SCRIPT = ROOT / "qualtrics" / "qualtrics-frame.js"

RAW_FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DISPLAY_FEATURES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
LABELS = ["Diabetes", "No Diabetes"]
TRAIN_IDS = list(range(130100, 130112))
TEST_IDS = {0: list(range(130200, 130210)), 1: list(range(130300, 130310))}


def browser_model_payload(model: Any) -> dict[str, Any]:
    preprocessor = model.named_steps["preprocessor"]
    mlp = model.named_steps["model"]
    scaler = preprocessor.named_transformers_["numeric"]
    return {
        "format": "sklearn-mlp-binary-v1",
        "feature_names": RAW_FEATURES,
        "class_labels": LABELS,
        "classes": [int(value) for value in mlp.classes_.tolist()],
        "preprocessing": {
            "type": "column-transformer-v1",
            "numeric": {
                "feature_names": list(preprocessor.transformers_[0][2]),
                "mean": [float(value) for value in scaler.mean_.tolist()],
                "scale": [float(value) for value in scaler.scale_.tolist()],
            },
            "categorical": {"feature_names": [], "categories": []},
        },
        "hidden_activation": str(mlp.activation),
        "output_activation": str(mlp.out_activation_),
        "layers": [
            {
                "weights": [[float(value) for value in row] for row in weights.tolist()],
                "biases": [float(value) for value in biases.tolist()],
            }
            for weights, biases in zip(mlp.coefs_, mlp.intercepts_)
        ],
    }


def payload_for(
    row: dict[str, Any], assigned_id: int, model: Any, ranges: list[list[float]]
) -> dict[str, Any]:
    original = np.asarray([float(row["original"][name]) for name in DISPLAY_FEATURES])
    counterfactual = np.asarray(
        [float(row["counterfactual"][name]) for name in DISPLAY_FEATURES]
    )
    original_probs = model.predict_proba(pd.DataFrame([original], columns=RAW_FEATURES))[0]
    original_prediction = int(np.argmax(original_probs))
    expected_prediction = LABELS.index(row["prediction"])
    if original_prediction != expected_prediction:
        raise ValueError(f"Prediction mismatch for {row['source']}")
    counterfactual_probs = model.predict_proba(
        pd.DataFrame([counterfactual], columns=RAW_FEATURES)
    )[0]
    target_prediction = 1 - original_prediction
    if int(np.argmax(counterfactual_probs)) != target_prediction:
        raise ValueError(f"Counterfactual does not flip for {row['source']}")

    lime_values = [float(value) for value in row["lime_values_no_diabetes_direction"]]
    conditions = [str(value) for value in row["lime_feature_conditions"]]
    shown_indices = sorted(
        range(len(lime_values)), key=lambda index: -abs(lime_values[index])
    )[:2]
    changed_names = list(row["counterfactual_changes"])
    changed_indices = [DISPLAY_FEATURES.index(name) for name in changed_names]
    if set(changed_indices) != set(shown_indices):
        raise ValueError(f"LIME/counterfactual pair mismatch for {row['source']}")

    phase = row["role"]
    return {
        "dataset": "diabetes",
        "model": "mlp",
        "xai_method": "lime",
        "xai_type": "attribution",
        "split": "train" if phase == "training" else "test",
        "explanation_feature_count": 2,
        "instance_id": assigned_id,
        "available_instance_count": 392,
        "feature_names": DISPLAY_FEATURES,
        "raw_feature_names": RAW_FEATURES,
        "feature_types": ["numerical"] * 5,
        "feature_ranges": ranges,
        "raw_feature_ranges": ranges,
        "feature_values": original.tolist(),
        "raw_feature_values": original.tolist(),
        "prediction": {
            "value": original_prediction,
            "label": LABELS[original_prediction],
            "probabilities": [
                {"label": label, "value": float(original_probs[index])}
                for index, label in enumerate(LABELS)
            ],
        },
        "prediction_labels": LABELS,
        "feature_importance_by_name": {
            RAW_FEATURES[index]: lime_values[index] for index in range(5)
        },
        "counterfactual_settings": {
            "mode": "minimal",
            "controllable_only": False,
            "controllable_feature_names": DISPLAY_FEATURES[:-1],
            "raw_controllable_feature_names": RAW_FEATURES[:-1],
        },
        "attribution": {
            "method": "lime",
            "feature_selection": {
                "method": "discrete_lime_quartile_bins_lasso_path",
                "num_features": 2,
                "num_samples": 500,
                "discretize_continuous": True,
            },
            "values": lime_values,
            "ranking_values": [abs(value) for value in lime_values],
            "raw_values": lime_values,
            "max_abs_value": max(abs(value) for value in lime_values),
            "shown_feature_count": 2,
            "shown_feature_indices": shown_indices,
            "direction_labels": {"left": "Diabetes", "right": "No Diabetes"},
            "feature_conditions": conditions,
            "featureConditions": conditions,
            "local_fidelity": float(row["lime_fidelity"]),
        },
        "counterfactual": {
            "feature_values": counterfactual.tolist(),
            "raw_feature_values": counterfactual.tolist(),
            "prediction": {
                "value": target_prediction,
                "label": LABELS[target_prediction],
            },
            "target_prediction": {
                "value": target_prediction,
                "label": LABELS[target_prediction],
            },
            "target_probability": float(counterfactual_probs[target_prediction]),
            "selected_feature_names": changed_names,
            "raw_selected_feature_names": [RAW_FEATURES[index] for index in changed_indices],
            "source": "discrete_lime_top2_independent_change_optimization",
            "generation_mode": "minimal",
        },
        "feature_pair_key": "|".join(
            sorted(RAW_FEATURES[index] for index in changed_indices)
        ),
        "feature_pair_names": changed_names,
        "source_split": row["source_split"],
        "source_instance_id": int(row["source_instance_id"]),
        "source_row_id": int(row["source_row_id"]),
        "experimental_phase": phase,
        "selection_cluster": int(row["cluster"]),
        "selection_role": (
            "discrete_lime_cf_cluster_training"
            if phase == "training"
            else "copy_transfer_matched_testing"
        ),
        "nearest_training_source": row.get("nearest_training_source"),
        "profile_distance_to_nearest_training": row.get(
            "profile_distance_to_nearest_training"
        ),
    }


def build_cases() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    rows = selection["selected_instances"]
    if Counter(row["role"] for row in rows) != {"training": 12, "testing": 20}:
        raise ValueError("Selection must contain 12 training and 20 testing cases")
    model = joblib.load(MODEL_PATH)
    static = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    ranges = static["datasets"]["diabetes"]["training_pool"][0]["raw_feature_ranges"]

    training_rows = sorted(
        (row for row in rows if row["role"] == "training"),
        key=lambda row: (int(row["cluster"]), LABELS.index(row["prediction"]), row["source"]),
    )
    testing_rows = sorted(
        (row for row in rows if row["role"] == "testing"),
        key=lambda row: (LABELS.index(row["prediction"]), int(row["cluster"]), row["source"]),
    )
    training = [
        payload_for(row, assigned_id, model, ranges)
        for row, assigned_id in zip(training_rows, TRAIN_IDS)
    ]
    testing = []
    for prediction in (0, 1):
        group = [row for row in testing_rows if LABELS.index(row["prediction"]) == prediction]
        testing.extend(
            payload_for(row, assigned_id, model, ranges)
            for row, assigned_id in zip(group, TEST_IDS[prediction])
        )
    return selection, training, testing


def format_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def signed(value: float) -> str:
    return f"{'+' if value > 0 else '−'}{format_value(abs(value))}"


def pair_text(names: list[str]) -> str:
    return f"{names[0]} and {names[1]}"


def set_mcq(question: dict[str, Any], prompt: str, choices: list[str]) -> None:
    question["QuestionText"] = prompt
    question["Choices"] = {
        str(index): {"Display": choice} for index, choice in enumerate(choices, start=1)
    }
    question["ChoiceOrder"] = [str(index) for index in range(1, len(choices) + 1)]


def basic_tutorial_html() -> str:
    return """<div id="cf-basic-tutorial-root" class="tutorial-root">
  <h1>Basic Interface</h1>
  <div class="tutorial-two-column">
    <div>
      <p>Read the specific profile shown. Pay attention to:</p>
      <ol class="tutorial-bullets">
        <li>the five <b>attribute names</b> and their exact <b>values</b>;</li>
        <li>the position of each value on its Low–High range; and</li>
        <li>the selected box showing the AI's <b>Diabetes</b> or <b>No Diabetes</b> output.</li>
      </ol>
      <p>The screening questions ask about this exact profile.</p>
    </div>
    <div class="tutorial-preview"><iframe id="cf-basic-tutorial-frame" title="Basic interface example"></iframe></div>
  </div>
</div>
<style>.tutorial-root{font-size:18px;line-height:1.35;max-width:1100px;margin:0 auto}.tutorial-root h1{font-size:30px;margin:0 0 20px}.tutorial-two-column{display:grid;grid-template-columns:minmax(280px,.9fr) minmax(430px,1.25fr);gap:28px;align-items:start}.tutorial-bullets{padding-left:28px}.tutorial-bullets li{margin-bottom:12px}.tutorial-preview iframe{width:100%;min-height:420px;border:0}@media(max-width:800px){.tutorial-two-column{grid-template-columns:1fr}}</style>"""


def basic_tutorial_js(tutorial_id: int) -> str:
    return f"""Qualtrics.SurveyEngine.addOnload(function () {{
  var frame = document.getElementById('cf-basic-tutorial-frame');
  if (!frame) return;
  var base = String(Qualtrics.SurveyEngine.getEmbeddedData('ui_base_url') || 'https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html');
  var params = new URLSearchParams({{appId:'diabetes',xaiType:'none',split:'train',instanceId:'{tutorial_id}',showPrediction:'1',counterfactualSimulation:'0',tutorialCallouts:'basic'}});
  frame.src = base + (base.indexOf('?') >= 0 ? '&' : '?') + params.toString();
  window.addEventListener('message', function (event) {{
    if (event.source === frame.contentWindow && event.data && event.data.type === 'counterfactual-ui:iframe-height') frame.style.height = Math.max(320, Math.min(900, Number(event.data.height) || 0)) + 'px';
  }}, false);
}});"""


def explanation_tutorial_html(case: dict[str, Any]) -> str:
    attribution = case["attribution"]
    indices = attribution["shown_feature_indices"]
    total = sum(abs(attribution["values"][index]) for index in indices) or 1
    influence_items = []
    for index in indices:
        value = attribution["values"][index]
        percent = round(abs(value) / total * 100)
        color = "tutorial-blue" if value >= 0 else "tutorial-red"
        direction = "No Diabetes" if value >= 0 else "Diabetes"
        feature_name = DISPLAY_FEATURES[index]
        influence_items.append(
            f'<li><b>{feature_name}</b>: <span class="{color}"><b>{"+" if value >= 0 else "−"}{percent}%</b></span> toward <b>{direction}</b>.</li>'
        )
    changed = []
    for index, (before, after) in enumerate(
        zip(case["feature_values"], case["counterfactual"]["feature_values"])
    ):
        delta = float(after) - float(before)
        if abs(delta) <= 1e-9:
            continue
        color = "tutorial-blue" if delta > 0 else "tutorial-red"
        changed.append(
            f'<li><b>{DISPLAY_FEATURES[index]}</b>: {format_value(float(before))} → {format_value(float(after))} '
            f'(<span class="{color}"><b>{signed(delta)}</b></span>).</li>'
        )
    return f"""<div id="cf-explanation-tutorial-root" class="tutorial-root">
  <h1>AI Explanation</h1>
  <div class="tutorial-two-column">
    <div>
      <section data-explanation="attribution">
        <p>The attribution explains how each highlighted attribute's <b>current value</b> contributed to this prediction. It does not say that changing the value will necessarily reverse the prediction.</p>
        <ul class="tutorial-bullets">{''.join(influence_items)}</ul>
        <p><span class="tutorial-red"><b>Red influence bars</b></span> contribute toward <b>Diabetes</b>; <span class="tutorial-blue"><b>blue influence bars</b></span> contribute toward <b>No Diabetes</b>.</p>
      </section>
      <section data-explanation="counterfactual" hidden>
        <p>The counter-example changes exactly two values and shows how those changes alter the AI output:</p>
        <ul class="tutorial-bullets">{''.join(changed)}</ul>
        <p><span class="tutorial-red"><b>Red change markers</b></span> mean a value decreased; <span class="tutorial-blue"><b>blue change markers</b></span> mean a value increased.</p>
      </section>
    </div>
    <div class="tutorial-preview"><iframe id="cf-explanation-tutorial-frame" title="Explanation interface example"></iframe></div>
  </div>
</div>
<style>.tutorial-root{{font-size:18px;line-height:1.35;max-width:1100px;margin:0 auto}}.tutorial-root h1{{font-size:30px;margin:0 0 20px}}.tutorial-two-column{{display:grid;grid-template-columns:minmax(300px,.95fr) minmax(430px,1.25fr);gap:28px;align-items:start}}.tutorial-bullets{{padding-left:28px}}.tutorial-bullets li{{margin-bottom:12px}}.tutorial-preview iframe{{width:100%;min-height:420px;border:0}}.tutorial-red{{color:#ea3335}}.tutorial-blue{{color:#3c88e8}}@media(max-width:800px){{.tutorial-two-column{{grid-template-columns:1fr}}}}</style>"""


def explanation_tutorial_js(tutorial_id: int) -> str:
    return f"""Qualtrics.SurveyEngine.addOnload(function () {{
  var root = document.getElementById('cf-explanation-tutorial-root');
  var frame = document.getElementById('cf-explanation-tutorial-frame');
  if (!root || !frame) return;
  var explanation = String(Qualtrics.SurveyEngine.getEmbeddedData('xaiType') || 'none').toLowerCase();
  if (explanation === 'counterfactuals') explanation = 'counterfactual';
  if (['attribution','counterfactual'].indexOf(explanation) < 0) {{ this.getQuestionContainer().style.display = 'none'; return; }}
  Array.prototype.forEach.call(root.querySelectorAll('[data-explanation]'), function (panel) {{ panel.hidden = panel.getAttribute('data-explanation') !== explanation; }});
  var base = String(Qualtrics.SurveyEngine.getEmbeddedData('ui_base_url') || 'https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html');
  var params = new URLSearchParams({{appId:'diabetes',xaiType:explanation,split:'train',instanceId:'{tutorial_id}',showPrediction:'1',counterfactualSimulation:'0',tutorialCallouts:'explanation'}});
  frame.src = base + (base.indexOf('?') >= 0 ? '&' : '?') + params.toString();
  window.addEventListener('message', function (event) {{
    if (event.source === frame.contentWindow && event.data && event.data.type === 'counterfactual-ui:iframe-height') frame.style.height = Math.max(360, Math.min(900, Number(event.data.height) || 0)) + 'px';
  }}, false);
}});"""


def preview_js(tutorial_id: int) -> str:
    return f"""Qualtrics.SurveyEngine.addOnload(function () {{
  var question = this;
  var frame = document.getElementById('cf-task-preview-frame');
  var status = document.getElementById('cf-task-preview-status');
  if (!frame) return;
  question.disableNextButton();
  var base = String(Qualtrics.SurveyEngine.getEmbeddedData('ui_base_url') || 'https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html');
  var params = new URLSearchParams({{appId:'diabetes',xaiType:'none',split:'train',instanceId:'{tutorial_id}',showPrediction:'1',counterfactualSimulation:'1'}});
  frame.src = base + (base.indexOf('?') >= 0 ? '&' : '?') + params.toString();
  window.addEventListener('message', function (event) {{
    if (event.source !== frame.contentWindow || !event.data) return;
    if (event.data.type === 'counterfactual-ui:iframe-height') {{
      frame.style.height = Math.max(320, Math.min(900, Number(event.data.height) || 0)) + 'px';
    }} else if (event.data.type === 'counterfactual-ui:simulation-change') {{
      var changes = Array.isArray(event.data.changes) ? event.data.changes : [];
      if (changes.length > 0) {{
        question.enableNextButton();
        if (status) status.textContent = 'Your preview change is registered. You may continue.';
      }} else {{
        question.disableNextButton();
        if (status) status.textContent = 'Move at least one value before continuing.';
      }}
    }}
  }}, false);
}});"""


def update_embedded_assignment(flow: Any) -> None:
    if isinstance(flow, dict):
        for item in flow.get("EmbeddedData", []):
            if item.get("Field") == "xaiType" and item.get("Value") == "counterfactuals":
                item["Value"] = "counterfactual"
        for value in flow.values():
            update_embedded_assignment(value)
    elif isinstance(flow, list):
        for value in flow:
            update_embedded_assignment(value)


def build_qsf(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8"))
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.5"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes recourse study with concrete two-attempt screening, discrete LIME, "
        "age-changing explanation clusters, delayed training review, and 20 test cases."
    )
    document["SurveyEntry"]["LastModified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    questions["QID20"]["QuestionText"] = re.sub(
        r'(<div data-domain="diabetes" hidden>\s*<h2>No Diabetes to Diabetes</h2>\s*<p>).*?(</p>)',
        r'\1Each of the following ten people currently receives a <b>No Diabetes</b> warning. '
        r'Describe the <b>minimal changes</b> that would make each profile receive a '
        r'<b>Diabetes</b> warning. These are the changes the person should avoid.\2',
        questions["QID20"]["QuestionText"],
        count=1,
        flags=re.DOTALL,
    )
    questions["QID379"]["QuestionText"] = re.sub(
        r'(<p class="cf-test-prompt" data-domain="diabetes" hidden>).*?(</p>)',
        r'\1\n  This person currently receives a <b>No Diabetes</b> warning. Describe the '
        r'<b>minimal changes</b> that would make this profile receive a <b>Diabetes</b> warning. '
        r'These are the changes the person should avoid.\n\2',
        questions["QID379"]["QuestionText"],
        count=1,
        flags=re.DOTALL,
    )
    update_embedded_assignment(
        next(element["Payload"] for element in document["SurveyElements"] if element.get("Element") == "FL")
    )

    tutorial = training[0]
    tutorial_id = int(tutorial["instance_id"])
    prediction_label = tutorial["prediction"]["label"]
    other_label = LABELS[1 - LABELS.index(prediction_label)]
    glucose = format_value(float(tutorial["feature_values"][0]))
    basic_choices = [prediction_label, other_label, "The interface does not show an output", "Both outputs"]
    glucose_choices = [
        glucose,
        format_value(float(tutorial["feature_values"][1])),
        format_value(float(tutorial["feature_values"][2])),
        format_value(float(tutorial["feature_values"][4])),
    ]

    influence_names = [
        DISPLAY_FEATURES[index] for index in tutorial["attribution"]["shown_feature_indices"]
    ]
    distractor_pairs = [
        ["Blood Pressure", "BMI"],
        ["Insulin", "BMI"],
        ["Blood Pressure", "Insulin"],
    ]
    influence_choices = [pair_text(influence_names)] + [pair_text(pair) for pair in distractor_pairs]
    red_influence_choices = [
        "The attribute's current value contributes toward a Diabetes warning",
        "The attribute's current value contributes toward a No Diabetes warning",
        "The attribute value decreased",
        "The attribute was ignored by the AI",
    ]

    change_entries = []
    reverse_entries = []
    for index, (before, after) in enumerate(
        zip(tutorial["feature_values"], tutorial["counterfactual"]["feature_values"])
    ):
        delta = float(after) - float(before)
        if abs(delta) <= 1e-9:
            continue
        change_entries.append(
            f"{DISPLAY_FEATURES[index]}: {format_value(float(before))} → {format_value(float(after))} ({signed(delta)})"
        )
        reverse_entries.append(
            f"{DISPLAY_FEATURES[index]}: {format_value(float(after))} → {format_value(float(before))} ({signed(-delta)})"
        )
    change_choices = [
        "; ".join(change_entries),
        "; ".join(reverse_entries),
        f"Only {influence_names[0]} changed",
        "No attribute values changed",
    ]
    red_change_choices = [
        "The attribute value decreased",
        "The attribute value increased",
        "The attribute caused a Diabetes warning",
        "The attribute is the most important feature",
    ]

    # Preserve the original numbered tutorial layout and wording. Only swap in
    # the v1.5 tutorial case and the values that the restored text describes.
    for qid in ("QID271", "QID34"):
        questions[qid]["QuestionJS"] = re.sub(
            r'("diabetes":\[)\d+',
            lambda match: match.group(1) + str(tutorial_id),
            questions[qid]["QuestionJS"],
            count=1,
        )

    attribution = tutorial["attribution"]
    shown_indices = attribution["shown_feature_indices"][:2]
    total_attribution = sum(abs(float(value)) for value in attribution["values"]) or 1.0
    influence_examples = []
    for index in shown_indices:
        value = float(attribution["values"][index])
        color = "tutorial-blue" if value >= 0 else "tutorial-red"
        sign = "+" if value >= 0 else "−"
        percent = round(abs(value) / total_attribution * 100)
        influence_examples.append(
            f'{DISPLAY_FEATURES[index]}, <span class="{color}">{sign}{percent}%</span>'
        )
    influence_example = " and ".join(influence_examples)

    changed_tutorial_items = []
    for index, (before, after) in enumerate(
        zip(tutorial["feature_values"], tutorial["counterfactual"]["feature_values"])
    ):
        delta = float(after) - float(before)
        if abs(delta) <= 1e-9:
            continue
        color = "tutorial-blue" if delta > 0 else "tutorial-red"
        direction = "increases" if delta > 0 else "decreases"
        changed_tutorial_items.append(
            f'<li><span class="{color}">{DISPLAY_FEATURES[index]}</span> {direction} '
            f'from {format_value(float(before))} to {format_value(float(after))}.</li>'
        )
    changed_tutorial_html = "".join(changed_tutorial_items)

    for qid in ("QID9", "QID35"):
        text = questions[qid]["QuestionText"]
        text = re.sub(
            r'(data-domain="diabetes" data-explanation="attribution" data-instance-id=")\d+(" hidden>)',
            lambda match: match.group(1) + str(tutorial_id) + match.group(2),
            text,
            count=1,
        )
        text = re.sub(
            r'(data-domain="diabetes" data-explanation="counterfactual" data-instance-id=")\d+(" hidden>)',
            lambda match: match.group(1) + str(tutorial_id) + match.group(2),
            text,
            count=1,
        )
        text = re.sub(
            r'(data-domain="diabetes" data-explanation="attribution"[\s\S]*?'
            r'<li>The influence of the two most important attributes \().*?(\)\. The higher)',
            lambda match: match.group(1) + influence_example + match.group(2),
            text,
            count=1,
        )
        text = re.sub(
            r'(data-domain="diabetes" data-explanation="counterfactual"[\s\S]*?'
            r'<li>The changes in the two attributes in the counter-example\.<ul>).*?(</ul></li>)',
            lambda match: match.group(1) + changed_tutorial_html + match.group(2),
            text,
            count=1,
        )
        questions[qid]["QuestionText"] = text
        questions[qid]["QuestionJS"] = re.sub(
            r'("diabetes":\{"attribution":)\d+(,"counterfactual":)\d+',
            lambda match: (
                match.group(1) + str(tutorial_id) + match.group(2) + str(tutorial_id)
            ),
            questions[qid]["QuestionJS"],
            count=1,
        )

    for qid in ("QID18", "QID27"):
        set_mcq(questions[qid], "What AI output is selected for the tutorial profile?", basic_choices)
        questions[qid]["QuestionJS"] = ""
    for qid in ("QID21", "QID28"):
        set_mcq(questions[qid], "What is the Glucose value in the tutorial profile?", glucose_choices)
    for qid in ("QID22", "QID29"):
        set_mcq(questions[qid], "For the current case, which are the two most influential attributes?", influence_choices)
    for qid in ("QID23", "QID30"):
        set_mcq(questions[qid], "What does a red influence bar mean?", red_influence_choices)
    for qid in ("QID24", "QID31"):
        set_mcq(questions[qid], "Which option gives both counter-example changes exactly?", change_choices)
    for qid in ("QID25", "QID32"):
        set_mcq(questions[qid], "What does a red change marker mean?", red_change_choices)

    frame_script = FRAME_SCRIPT.read_text(encoding="utf-8")
    for qid in ("QID12", "QID409", "QID376", "QID379"):
        questions[qid]["QuestionJS"] = frame_script

    preview = questions["QID26"]
    preview["QuestionText"] = re.sub(
        r"</div>\s*<style>",
        '<p id="cf-task-preview-status" style="font-weight:700;color:#34495e">Move at least one value before continuing.</p></div><style>',
        preview["QuestionText"],
        count=1,
    )
    preview["QuestionJS"] = preview_js(tutorial_id)

    blocks = next(
        element["Payload"] for element in document["SurveyElements"] if element.get("Element") == "BL"
    )
    blocks_by_id = {block["ID"]: block for block in blocks}
    blocks_by_id["BL_3fJJUKH6EXaeRIq"]["Description"] = "Training block A (hidden): Glucose + Age"
    blocks_by_id["BL_6Hb2Nq8Tx4Lm7Wp"]["Description"] = "Training block B (hidden): Glucose + Insulin"

    OUTPUT_QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def write_bundles(
    selection: dict[str, Any], training: list[dict[str, Any]], testing: list[dict[str, Any]]
) -> None:
    bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    diabetes = bundle["datasets"]["diabetes"]
    experiment_ids = set(TRAIN_IDS + TEST_IDS[0] + TEST_IDS[1])
    diabetes["training_pool"] = [
        case for case in diabetes["training_pool"] if int(case["instance_id"]) not in experiment_ids
    ] + copy.deepcopy(training)
    diabetes["test_pool"] = [
        case for case in diabetes["test_pool"] if int(case["instance_id"]) not in experiment_ids
    ] + copy.deepcopy(testing)
    model = joblib.load(MODEL_PATH)
    diabetes["browser_model"] = browser_model_payload(model)
    diabetes["metadata"].update(
        {
            "model": "regularized_mlp_alpha_0.01",
            "xai_methods": ["discrete_lime"],
            "qualtrics_v1_5_training_ids": TRAIN_IDS,
            "qualtrics_v1_5_testing_ids_by_prediction": {
                str(prediction): ids for prediction, ids in TEST_IDS.items()
            },
            "qualtrics_v1_5_training_blocks": {
                "glucose_age": TRAIN_IDS[:6],
                "glucose_insulin": TRAIN_IDS[6:],
            },
            "qualtrics_v1_5_selection": (
                "Discrete-LIME plus counterfactual clustering with an age-changing cluster; "
                "testing selected for same pair/direction and copy-edit transfer"
            ),
            "qualtrics_v1_5_copy_transfer": selection["copy_strategy_validation"],
            "static_training_pool_count": len(diabetes["training_pool"]),
            "static_test_pool_count": len(diabetes["test_pool"]),
        }
    )
    bundle["version"] = "static-experiment-v15-discrete-lime-age-copy"
    bundle["generated_at"] = date.today().isoformat()
    STATIC_JSON.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATIC_JS.write_text(
        "window.EXPERIMENT_DATA = "
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    analysis = {
        "version": "diabetes-experiment-v1.5-discrete-lime-age-copy",
        "generated_at": date.today().isoformat(),
        "default_model": "mlp",
        "datasets": {
            "diabetes": {
                "metadata": copy.deepcopy(diabetes["metadata"]),
                "browser_model": copy.deepcopy(diabetes["browser_model"]),
                "training_pool": copy.deepcopy(training),
                "test_pool": copy.deepcopy(testing),
            }
        },
    }
    ANALYSIS_BUNDLE.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_rows = []
    for case in training + testing:
        manifest_rows.append(
            {
                "domain": "diabetes",
                "experimental_phase": case["experimental_phase"],
                "qualtrics_instance_id": case["instance_id"],
                "source_split": case["source_split"],
                "source_instance_id": case["source_instance_id"],
                "source_row_id": case["source_row_id"],
                "prediction": int(case["prediction"]["value"]),
                "prediction_label": case["prediction"]["label"],
                "selection_cluster": case["selection_cluster"],
                "feature_pair_key": case["feature_pair_key"],
                "nearest_training_source": case.get("nearest_training_source"),
            }
        )
    MANIFEST.write_text(
        json.dumps({"version": "1.5-discrete-age", "cases": manifest_rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    selection, training, testing = build_cases()
    write_bundles(selection, training, testing)
    build_qsf(training, testing)
    print(
        json.dumps(
            {
                "qsf": str(OUTPUT_QSF),
                "training": len(training),
                "testing": len(testing),
                "clusters": Counter(case["feature_pair_key"] for case in training),
                "copy_transfer_success": selection["copy_strategy_validation"]["success_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
