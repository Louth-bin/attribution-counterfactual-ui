"""Fit the three perfect-memory forward models to v0.9 participant responses.

The fit is prequential: each response is predicted from earlier trials only.
The correct AI label and explanation on trial t are learned only after scoring
the participant's response on trial t.  The baseline has no fitted continuous
parameters, so BIC is simply -2 log likelihood for all three models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_models.minimal_mental_model import (  # noqa: E402
    FeatureSpace,
    MentalModel,
    PerfectMemoryCognitiveModel,
)


DEFAULT_INPUT = ROOT / "qualtrics" / "qualtrics_results_v0.9.for-plausibility.csv"
DEFAULT_BUNDLE = ROOT / "static" / "experiment-data.json"
DEFAULT_PARTICIPANTS = ROOT / "qualtrics" / "v0.9_participant_mental_model_fits.csv"
DEFAULT_TRIALS = ROOT / "qualtrics" / "v0.9_mental_model_trial_predictions.csv"
DEFAULT_SUMMARY = ROOT / "qualtrics" / "v0.9_mental_model_fit_summary.csv"
DEFAULT_REPORT = ROOT / "qualtrics" / "v0.9_mental_model_fit_report.md"

MODELS = tuple(MentalModel)
EPSILON = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cases(path: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        datasets = json.load(handle)["datasets"]
    case_maps = {
        domain: {str(case["instance_id"]): case for case in dataset["training_pool"]}
        for domain, dataset in datasets.items()
    }
    return case_maps, datasets


def profile_from_case(case: dict[str, Any], key: str = "raw_feature_values") -> dict[str, Any]:
    return dict(zip(case["raw_feature_names"], case[key]))


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def explanation_payload(case: dict[str, Any], xai: str) -> dict[str, Any]:
    if xai == "none":
        return {"explanation": "none"}
    if xai == "attribution":
        return {
            "explanation": "attribution",
            "attributions": dict(
                zip(case["raw_feature_names"], case["attribution"]["values"])
            ),
        }
    if xai == "counterfactual":
        counterfactual = case["counterfactual"]
        return {
            "explanation": "counterfactual",
            "counterfactual_profile": dict(
                zip(case["raw_feature_names"], counterfactual["raw_feature_values"])
            ),
            "counterfactual_target_label": counterfactual["prediction"]["label"],
        }
    raise ValueError(f"Unknown explanation condition {xai!r}")


def participant_fit(
    participant_rows: list[dict[str, str]],
    case_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    participant_rows = sorted(participant_rows, key=lambda row: int(row["case"]))
    participant = participant_rows[0]["participant"]
    domain = participant_rows[0]["domain"]
    xai = participant_rows[0]["xai"]
    if any(row["domain"] != domain or row["xai"] != xai for row in participant_rows):
        raise ValueError(f"Participant {participant} has inconsistent condition fields")

    first_case = case_map[participant_rows[0]["instance id"]]
    space = FeatureSpace.from_dataset_fields(
        first_case["raw_feature_names"],
        first_case["feature_types"],
        first_case["raw_feature_ranges"],
    )
    labels = tuple(first_case["prediction_labels"])
    model = PerfectMemoryCognitiveModel(space, labels)
    totals = {
        mental_model: {"nll": 0.0, "brier": 0.0, "accuracy": 0}
        for mental_model in MODELS
    }
    trial_rows: list[dict[str, Any]] = []

    for trial_index, row in enumerate(participant_rows, start=1):
        instance_id = row["instance id"]
        try:
            case = case_map[instance_id]
        except KeyError as error:
            raise ValueError(
                f"No static training case {domain}/{instance_id} for {participant}"
            ) from error
        correct_label = case["prediction"]["label"]
        if correct_label != row["original label"]:
            raise ValueError(
                f"Label mismatch for {domain}/{instance_id}: "
                f"bundle={correct_label!r}, CSV={row['original label']!r}"
            )
        response = row["training response"]
        if response not in labels:
            raise ValueError(f"Unknown response {response!r} for participant {participant}")
        profile = profile_from_case(case)
        relevance_before = model.global_importance()

        for mental_model in MODELS:
            prediction = model.predict(profile, mental_model)
            probability = max(float(prediction.probabilities[response]), EPSILON)
            positive_observed = float(response == labels[1])
            positive_probability = float(prediction.probabilities[labels[1]])
            nll = -math.log(probability)
            brier = (positive_probability - positive_observed) ** 2
            correct_prediction = int(prediction.label == response)
            totals[mental_model]["nll"] += nll
            totals[mental_model]["brier"] += brier
            totals[mental_model]["accuracy"] += correct_prediction
            trial_rows.append(
                {
                    "participant": participant,
                    "domain": domain,
                    "xai": xai,
                    "trial_order": trial_index,
                    "instance_id": instance_id,
                    "participant_response": response,
                    "ai_label": correct_label,
                    "participant_correct": int(response == correct_label),
                    "mental_model": mental_model.value,
                    "predicted_response": prediction.label,
                    "response_probability": probability,
                    "negative_log_likelihood": nll,
                    "brier_score": brier,
                    "predicted_participant_response": correct_prediction,
                    "evidence": prediction.evidence,
                    "memory_instances_before": len(model.instances),
                    "memory_changes_before": len(model.changes),
                    "feature_relevance_before": compact_json(relevance_before),
                }
            )

        # Feedback and XAI are displayed only after the participant responds.
        model.observe(profile, correct_label, **explanation_payload(case, xai))

    n_trials = len(participant_rows)
    fit_values: dict[str, dict[str, float]] = {}
    for mental_model in MODELS:
        name = mental_model.value
        values = totals[mental_model]
        fit_values[name] = {
            "nll": float(values["nll"]),
            "bic": 2.0 * float(values["nll"]),
            "mean_log_loss": float(values["nll"]) / n_trials,
            "brier": float(values["brier"]) / n_trials,
            "accuracy": float(values["accuracy"]) / n_trials,
        }

    ranked = sorted(MODELS, key=lambda item: (fit_values[item.value]["bic"], item.value))
    best = ranked[0].value
    best_bic = fit_values[best]["bic"]
    deltas = {
        mental_model.value: fit_values[mental_model.value]["bic"] - best_bic
        for mental_model in MODELS
    }
    supported = [mental_model.value for mental_model in MODELS if deltas[mental_model.value] < 2.0]
    second_gap = fit_values[ranked[1].value]["bic"] - best_bic
    if len(supported) == 1:
        conclusion = f"best supported: {best}"
    else:
        conclusion = "indistinguishable: " + " / ".join(supported)
    if second_gap < 2.0:
        strength = "indistinguishable"
    elif second_gap < 6.0:
        strength = "positive"
    elif second_gap < 10.0:
        strength = "strong"
    else:
        strength = "very strong"

    relevance = model.global_importance()
    polarity_vector = model.polarity()
    polarity = {}
    for feature in space.features:
        values = polarity_vector[space.feature_slice(feature.name)]
        polarity[feature.name] = float(values[0]) if feature.kind == "numerical" else values.tolist()

    participant_result: dict[str, Any] = {
        "participant": participant,
        "domain": domain,
        "xai": xai,
        "n_forward_trials": n_trials,
        "participant_training_accuracy": sum(
            int(row["training response"] == row["original label"])
            for row in participant_rows
        ) / n_trials,
        "fitted_continuous_parameters": 0,
        "best_forward_mental_model": best,
        "second_best_forward_mental_model": ranked[1].value,
        "delta_bic_to_second_best": second_gap,
        "supported_forward_mental_models_delta_bic_lt_2": " / ".join(supported),
        "forward_mental_model_fit_conclusion": conclusion,
        "forward_mental_model_evidence_strength": strength,
        "final_feature_relevance": compact_json(relevance),
        "final_feature_polarity": compact_json(polarity),
    }
    for mental_model in MODELS:
        name = mental_model.value
        participant_result[f"{name}_nll"] = fit_values[name]["nll"]
        participant_result[f"{name}_bic"] = fit_values[name]["bic"]
        participant_result[f"{name}_mean_log_loss"] = fit_values[name]["mean_log_loss"]
        participant_result[f"{name}_brier"] = fit_values[name]["brier"]
        participant_result[f"{name}_accuracy"] = fit_values[name]["accuracy"]
        participant_result[f"{name}_delta_bic"] = deltas[name]
    return participant_result, trial_rows


def make_summary(participants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", participants)]
    domains = sorted({row["domain"] for row in participants})
    conditions = sorted({row["xai"] for row in participants})
    groups.extend(
        (domain, "all", [row for row in participants if row["domain"] == domain])
        for domain in domains
    )
    groups.extend(
        ("all", xai, [row for row in participants if row["xai"] == xai])
        for xai in conditions
    )
    groups.extend(
        (
            domain,
            xai,
            [
                row
                for row in participants
                if row["domain"] == domain and row["xai"] == xai
            ],
        )
        for domain in domains
        for xai in conditions
        if any(row["domain"] == domain and row["xai"] == xai for row in participants)
    )

    summary: list[dict[str, Any]] = []
    for domain, xai, rows in groups:
        for mental_model in MODELS:
            name = mental_model.value
            summary.append(
                {
                    "domain": domain,
                    "xai": xai,
                    "mental_model": name,
                    "n_participants": len(rows),
                    "strict_best_count": sum(
                        row["best_forward_mental_model"] == name for row in rows
                    ),
                    "supported_delta_bic_lt_2_count": sum(
                        name
                        in row["supported_forward_mental_models_delta_bic_lt_2"].split(" / ")
                        for row in rows
                    ),
                    "mean_nll": sum(row[f"{name}_nll"] for row in rows) / len(rows),
                    "mean_log_loss": sum(
                        row[f"{name}_mean_log_loss"] for row in rows
                    )
                    / len(rows),
                    "mean_brier": sum(row[f"{name}_brier"] for row in rows) / len(rows),
                    "mean_accuracy": sum(row[f"{name}_accuracy"] for row in rows)
                    / len(rows),
                }
            )
    return summary


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def make_report(participants: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    best_counts = Counter(row["best_forward_mental_model"] for row in participants)
    support_sizes = Counter(
        len(row["supported_forward_mental_models_delta_bic_lt_2"].split(" / "))
        for row in participants
    )
    all_summary = {
        row["mental_model"]: row
        for row in summary
        if row["domain"] == "all" and row["xai"] == "all"
    }
    lines = [
        "# v0.9 perfect-memory mental-model fit",
        "",
        "## Method",
        "",
        (
            "Each participant's 10 forward-simulation responses was fitted "
            "prequentially. On each trial, exemplar, attribution, and prototype "
            "models predicted the participant's response using only earlier "
            "feedback. The current correct AI label and explanation were learned "
            "only after scoring that response."
        ),
        "",
        (
            "This is the reviewed perfect-memory baseline: all encountered "
            "instances are retained, there is no decay or retrieval threshold, "
            "and no continuous parameter is estimated per participant. All three "
            "models use the same range-normalized feature relevance vector, learned "
            "from class diagnosticity (no XAI), displayed absolute attribution "
            "strength (attribution XAI), or changed-feature frequency "
            "(counterfactual XAI)."
        ),
        "",
        (
            "Primary fit is negative log likelihood (NLL) of the participant's "
            "chosen labels. Because every model has zero fitted continuous "
            "parameters, BIC = 2 x NLL. Models within delta BIC < 2 are reported "
            "as empirically indistinguishable; the strict minimum is retained for "
            "sorting but should not be treated as decisive in those cases."
        ),
        "",
        "## Overall results",
        "",
        markdown_table(
            ["Mental model", "Strict best", "Supported (ΔBIC < 2)", "Mean NLL", "Mean accuracy"],
            [
                [
                    mental_model.value,
                    best_counts[mental_model.value],
                    all_summary[mental_model.value]["supported_delta_bic_lt_2_count"],
                    f"{all_summary[mental_model.value]['mean_nll']:.3f}",
                    f"{all_summary[mental_model.value]['mean_accuracy']:.3f}",
                ]
                for mental_model in MODELS
            ],
        ),
        "",
        (
            f"Fit identifiability: {support_sizes.get(1, 0)} participants have one "
            f"supported model, {support_sizes.get(2, 0)} have two, and "
            f"{support_sizes.get(3, 0)} have all three within ΔBIC < 2."
        ),
        "",
        "## Strict best model by domain and explanation",
        "",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in participants:
        grouped[(row["domain"], row["xai"])].append(row)
    group_rows = []
    for (domain, xai), rows in sorted(grouped.items()):
        counts = Counter(row["best_forward_mental_model"] for row in rows)
        decisive = sum(
            len(row["supported_forward_mental_models_delta_bic_lt_2"].split(" / ")) == 1
            for row in rows
        )
        group_rows.append(
            [
                domain,
                xai,
                len(rows),
                counts["exemplar"],
                counts["attribution"],
                counts["prototype"],
                decisive,
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["Domain", "XAI", "N", "Exemplar", "Attribution", "Prototype", "One supported"],
                group_rows,
            ),
            "",
            "## Participant-level interpretation",
            "",
            (
                "The participant CSV contains the strict best model, all models "
                "within ΔBIC < 2, fit strength, and the NLL/BIC, Brier score, and "
                "hard-choice accuracy for every candidate. The trial CSV contains "
                "the probability assigned to every observed response, allowing the "
                "fit to be audited trial by trial."
            ),
            "",
            "The first trial is necessarily an equal 0.5 prediction for all models because no feedback has yet been observed; this common term cancels in model comparison.",
            "",
            "This comparison concerns forward-simulation responses only. It does not infer which counterfactual editing strategy generated participants' later edits.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--participants", type=Path, default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for row in read_csv(args.input) if row["phase"] == "training"]
    case_maps, _ = load_cases(args.bundle)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["participant"]].append(row)

    participant_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for participant in sorted(grouped):
        participant_data = grouped[participant]
        domain = participant_data[0]["domain"]
        result, predictions = participant_fit(participant_data, case_maps[domain])
        participant_rows.append(result)
        trial_rows.extend(predictions)
    participant_rows.sort(key=lambda row: (row["domain"], row["xai"], row["participant"]))
    trial_rows.sort(
        key=lambda row: (
            row["domain"],
            row["xai"],
            row["participant"],
            int(row["trial_order"]),
            row["mental_model"],
        )
    )
    summary = make_summary(participant_rows)

    write_csv(args.participants, participant_rows, participant_rows[0].keys())
    write_csv(args.trials, trial_rows, trial_rows[0].keys())
    write_csv(args.summary, summary, summary[0].keys())
    args.report.write_text(make_report(participant_rows, summary), encoding="utf-8")
    print(f"participants={len(participant_rows)} trials={len(trial_rows)}")
    print(f"participant_fits={args.participants}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
