"""Fit cognitive model v0.2 to the latest participant counterfactual edits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from .model import (
    LAPSE_GRID,
    FEATURES,
    Trial,
    available_variants,
    build_memory,
    configure_selection_space,
    fit_global_lapse,
    fit_participant,
    label_sign,
    normalize_case,
    selection_space,
)


DEFAULT_REPOSITORY = Path(
    r"C:\Users\Louth\Desktop\counterfactual\attribution-counterfactual-ui"
)
DEFAULT_RESPONSES = DEFAULT_REPOSITORY / "qualtrics" / "qualtrics_results_v2.3.csv"
DEFAULT_CASES = DEFAULT_REPOSITORY / "analysis" / "diabetes_experiment_bundle_v1.5.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
EXCLUDED_PARTICIPANTS = {
    "R_31Mz4GTXDqENy8N",
    "R_5jKfnq8uYqeU1fO",
    "R_6uqevOJYlPhUkn9",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_data(response_path: Path, case_path: Path):
    rows = read_csv(response_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["participant"]].append(row)
    participants = {
        participant: participant_rows
        for participant, participant_rows in grouped.items()
        if sum(row["phase"] == "training" for row in participant_rows) == 12
        and sum(row["phase"] == "testing" for row in participant_rows) == 20
    }
    with case_path.open(encoding="utf-8") as source:
        dataset = json.load(source)["datasets"]["diabetes"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    output = []
    for participant, participant_rows in sorted(participants.items()):
        ordered_training = sorted(
            (row for row in participant_rows if row["phase"] == "training"),
            key=lambda row: int(float(row.get("case") or row["trial number"])),
        )
        ordered_testing = sorted(
            (row for row in participant_rows if row["phase"] == "testing"),
            key=lambda row: int(float(row.get("case") or row["trial number"])),
        )
        condition = ordered_training[0]["xai"]
        training_cases = [
            case_map[int(float(row["instance id"]))] for row in ordered_training
        ]
        memory = build_memory(training_cases, condition)
        trials = []
        for order, row in enumerate(ordered_testing, start=1):
            case = case_map[int(float(row["instance id"]))]
            trials.append(Trial(
                profile=normalize_case(case),
                target=label_sign(row["target label"]),
                observed=np.asarray([
                    float(row[f"x_{index}_change"]) for index in range(1, 6)
                ]),
                instance_id=int(case["instance_id"]),
                order=order,
            ))
        output.append((participant, memory, trials))
    return output


def safe_wilcoxon(left: list[float], right: list[float]) -> float:
    differences = np.asarray(left) - np.asarray(right)
    if np.allclose(differences, 0):
        return 1.0
    return float(wilcoxon(differences).pvalue)


def summarize(fits: list[dict], lapse_rows: list[dict], retained_only: bool) -> dict:
    eligible = [
        row for row in fits
        if not retained_only or row["participant"] not in EXCLUDED_PARTICIPANTS
    ]
    winners = [row for row in eligible if row["selected_by_cv"]]
    participant_count = len({row["participant"] for row in eligible})
    uniform_nll = math.log(len(selection_space()))
    result: dict[str, Any] = {
        "participants": participant_count,
        "retained_only": retained_only,
        "selection_space": [
            [FEATURES[index] for index in subset]
            for subset in selection_space()
        ],
        "global_lapse": winners[0]["lapse_global"],
        "global_lapse_search": lapse_rows,
        "uniform_nonempty_subset_nll": uniform_nll,
        "winner_counts": dict(Counter(row["variant"] for row in winners)),
        "winner_counts_by_xai": {},
        "winner_mean_metrics": {},
        "variant_metrics": {},
        "prototype_exemplar_comparisons": {},
        "condition_metrics": {},
        "baseline_comparisons": {},
    }
    for condition in sorted({row["xai"] for row in winners}):
        condition_rows = [row for row in winners if row["xai"] == condition]
        result["winner_counts_by_xai"][condition] = dict(
            Counter(row["variant"] for row in condition_rows)
        )
        result["condition_metrics"][condition] = {
            "participants": len(condition_rows),
            "mean_cv_selection_nll": float(np.mean([row["cv_selection_nll"] for row in condition_rows])),
            "mean_cv_expected_selection_f1": float(np.mean([row["cv_expected_selection_f1"] for row in condition_rows])),
            "mean_cv_map_selection_f1": float(np.mean([row["cv_map_selection_f1"] for row in condition_rows])),
            "mean_cv_exact_subset_accuracy": float(np.mean([row["cv_exact_subset_accuracy"] for row in condition_rows])),
            "mean_cv_amount_mae": float(np.mean([row["cv_amount_mae"] for row in condition_rows])),
            "mean_cv_direction_accuracy": float(np.mean([row["cv_direction_accuracy"] for row in condition_rows])),
        }
    for metric in (
        "cv_selection_nll",
        "cv_expected_selection_f1",
        "cv_map_selection_f1",
        "cv_exact_subset_accuracy",
        "cv_cardinality_mae",
        "cv_amount_mae",
        "cv_direction_accuracy",
        "cv_zero_amount_mae",
    ):
        result["winner_mean_metrics"][metric] = float(
            np.mean([row[metric] for row in winners])
        )
    winner_nll = [row["cv_selection_nll"] for row in winners]
    zero_amount = [row["cv_zero_amount_mae"] for row in winners]
    model_amount = [row["cv_amount_mae"] for row in winners]
    result["baseline_comparisons"] = {
        "mean_nll_improvement_over_uniform": float(uniform_nll - np.mean(winner_nll)),
        "participants_better_than_uniform_nll": sum(value < uniform_nll for value in winner_nll),
        "selection_nll_vs_uniform_wilcoxon_p": safe_wilcoxon(winner_nll, [uniform_nll] * len(winner_nll)),
        "mean_amount_mae_improvement_over_zero": float(np.mean(np.asarray(zero_amount) - model_amount)),
        "participants_better_than_zero_amount": sum(model < zero for model, zero in zip(model_amount, zero_amount)),
        "amount_mae_vs_zero_wilcoxon_p": safe_wilcoxon(model_amount, zero_amount),
    }
    for variant in sorted({row["variant"] for row in eligible}):
        rows = [row for row in eligible if row["variant"] == variant]
        result["variant_metrics"][variant] = {
            "available_participants": len(rows),
            "winner_count": sum(row["selected_by_cv"] for row in rows),
            "mean_cv_selection_nll": float(np.mean([row["cv_selection_nll"] for row in rows])),
            "mean_cv_expected_selection_f1": float(np.mean([row["cv_expected_selection_f1"] for row in rows])),
            "mean_cv_map_selection_f1": float(np.mean([row["cv_map_selection_f1"] for row in rows])),
            "mean_cv_amount_mae": float(np.mean([row["cv_amount_mae"] for row in rows])),
            "mean_cv_direction_accuracy": float(np.mean([row["cv_direction_accuracy"] for row in rows])),
        }
    by_key = {(row["participant"], row["variant"]): row for row in eligible}
    for representation in ("base", "additive", "counterfactual"):
        pairs = []
        for participant in sorted({row["participant"] for row in eligible}):
            prototype = by_key.get((participant, f"{representation}_prototype"))
            exemplar = by_key.get((participant, f"{representation}_exemplar"))
            if prototype and exemplar:
                pairs.append((prototype, exemplar))
        if not pairs:
            continue
        prototype_nll = [pair[0]["cv_selection_nll"] for pair in pairs]
        exemplar_nll = [pair[1]["cv_selection_nll"] for pair in pairs]
        prototype_amount = [pair[0]["cv_amount_mae"] for pair in pairs]
        exemplar_amount = [pair[1]["cv_amount_mae"] for pair in pairs]
        result["prototype_exemplar_comparisons"][representation] = {
            "participants": len(pairs),
            "mean_exemplar_minus_prototype_selection_nll": float(np.mean(np.asarray(exemplar_nll) - prototype_nll)),
            "selection_nll_wilcoxon_p": safe_wilcoxon(exemplar_nll, prototype_nll),
            "mean_exemplar_minus_prototype_amount_mae": float(np.mean(np.asarray(exemplar_amount) - prototype_amount)),
            "amount_mae_wilcoxon_p": safe_wilcoxon(exemplar_amount, prototype_amount),
        }
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def make_report(summary: dict, retained: dict) -> str:
    variant_rows = []
    for variant, metrics in summary["variant_metrics"].items():
        variant_rows.append([
            variant,
            metrics["available_participants"],
            metrics["winner_count"],
            f"{metrics['mean_cv_selection_nll']:.3f}",
            f"{metrics['mean_cv_expected_selection_f1']:.3f}",
            f"{metrics['mean_cv_map_selection_f1']:.3f}",
            f"{metrics['mean_cv_amount_mae']:.3f}",
            f"{metrics['mean_cv_direction_accuracy']:.3f}",
        ])
    comparison_rows = []
    for representation, metrics in summary["prototype_exemplar_comparisons"].items():
        comparison_rows.append([
            representation,
            metrics["participants"],
            f"{metrics['mean_exemplar_minus_prototype_selection_nll']:+.3f}",
            f"{metrics['selection_nll_wilcoxon_p']:.3f}",
            f"{metrics['mean_exemplar_minus_prototype_amount_mae']:+.3f}",
            f"{metrics['amount_mae_wilcoxon_p']:.3f}",
        ])
    condition_rows = []
    for condition, metrics in summary["condition_metrics"].items():
        winners = summary["winner_counts_by_xai"][condition]
        condition_rows.append([
            condition,
            metrics["participants"],
            ", ".join(f"{name}={count}" for name, count in sorted(winners.items())),
            f"{metrics['mean_cv_selection_nll']:.3f}",
            f"{metrics['mean_cv_expected_selection_f1']:.3f}",
            f"{metrics['mean_cv_amount_mae']:.3f}",
            f"{metrics['mean_cv_direction_accuracy']:.3f}",
        ])
    winner = summary["winner_mean_metrics"]
    baseline = summary["baseline_comparisons"]
    return "\n".join([
        "# Cognitive model v0.2 fit report",
        "",
        "## Analysis specification",
        "",
        f"The analysis includes {summary['participants']} complete participants (12 ordered training exposures and 20 testing edits each). The selection distribution is conditioned on the interface-permitted set: {summary['selection_space']}. The previously screened cohort contains {retained['participants']} participants. Base prototype and exemplar models are available in every condition; additive models are available after Attribution or Counterfactual XAI; counterfactual-edit models are available only after Counterfactual XAI.",
        "",
        "Feature probabilities are conditioned on the subsets actually permitted by the interface. Counterfactual memory is consolidated separately for each feature; edit frequency, conditional signed change, and conditional endpoint are retained rather than treating displayed feature pairs as atomic memories. The global feature-level lapse was selected by pooled five-fold cross-validated subset negative log likelihood. Each remaining parameter was fitted per participant and variant. Model selection uses cross-validated exact-subset NLL, with conditional amount MAE as a tie-breaker. Amount MAE is evaluated on the attributes the participant actually selected, keeping feature selection and edit magnitude diagnostically separate.",
        "",
        "## Global lapse",
        "",
        f"Selected lapse: **{summary['global_lapse']:.3f}**. For each feature, lapse transforms its core inclusion probability as p'=(1-lapse)p+lapse/2 before the independent feature probabilities are multiplied and conditioned on the permitted selection space. Its uniform baseline has NLL {summary['uniform_nonempty_subset_nll']:.3f}.",
        "",
        "## Variant-level cross-validated fit",
        "",
        markdown_table(
            ["Variant", "Available N", "Winners", "Subset NLL", "Expected F1", "MAP F1", "Amount MAE", "Direction accuracy"],
            variant_rows,
        ),
        "",
        "## Selected-model performance",
        "",
        f"Across participants, selected models achieved mean subset NLL {winner['cv_selection_nll']:.3f}, expected selection F1 {winner['cv_expected_selection_f1']:.3f}, MAP selection F1 {winner['cv_map_selection_f1']:.3f}, exact-subset accuracy {winner['cv_exact_subset_accuracy']:.3f}, conditional amount MAE {winner['cv_amount_mae']:.3f}, and conditional direction accuracy {winner['cv_direction_accuracy']:.3f}.",
        "",
        "Winner counts: " + ", ".join(f"{name}={count}" for name, count in sorted(summary["winner_counts"].items())),
        "",
        f"The selected models improve mean subset NLL over the uniform baseline by {baseline['mean_nll_improvement_over_uniform']:.3f}; {baseline['participants_better_than_uniform_nll']}/{summary['participants']} participants improve (Wilcoxon p={baseline['selection_nll_vs_uniform_wilcoxon_p']:.3f}). Conditional amount MAE improves over predicting zero movement by {baseline['mean_amount_mae_improvement_over_zero']:.3f}; {baseline['participants_better_than_zero_amount']}/{summary['participants']} participants improve (Wilcoxon p={baseline['amount_mae_vs_zero_wilcoxon_p']:.3f}).",
        "",
        "## Fit by explanation condition",
        "",
        markdown_table(
            ["XAI", "N", "Selected variants", "Subset NLL", "Expected F1", "Amount MAE", "Direction accuracy"],
            condition_rows,
        ),
        "",
        "## Prototype versus exemplar",
        "",
        "Negative differences favor exemplars; positive differences favor prototypes.",
        "",
        markdown_table(
            ["Representation", "Paired N", "Exemplar - prototype NLL", "p", "Exemplar - prototype amount MAE", "p"],
            comparison_rows,
        ),
        "",
        "## Interpretation limits",
        "",
        "This is an exploratory model comparison. The global lapse is cross-validated over trials but selected using this cohort, so it should be frozen before confirmatory testing on a new cohort. Aggregate selected-model performance has some model-selection optimism because each participant's winning strategy is chosen from the same cross-validation estimates being summarized; the fixed-variant results are the cleaner descriptive comparison. Strategy assignments indicate which implemented process predicts held-out edits most closely; they do not establish that a participant exclusively used that strategy. Prototype and exemplar models overlap when distance sensitivity is zero, and the counterfactual-derived additive representation is an inverse-change approximation rather than a displayed attribution.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-lapse-search", action="store_true")
    parser.add_argument("--lapse", type=float)
    parser.add_argument("--participant")
    parser.add_argument(
        "--locked-feature",
        choices=FEATURES,
        help="Exclude this interface-locked feature from every candidate subset.",
    )
    parser.add_argument(
        "--fixed-cardinality",
        type=int,
        help="Condition selection on the number of edits permitted by the interface.",
    )
    arguments = parser.parse_args()

    configure_selection_space(
        locked_indices=(FEATURES.index(arguments.locked_feature),)
        if arguments.locked_feature else (),
        cardinality=arguments.fixed_cardinality,
    )

    participant_data = load_data(arguments.responses, arguments.cases)
    if arguments.participant:
        participant_data = [
            item for item in participant_data if item[0] == arguments.participant
        ]
    if arguments.lapse is not None:
        lapse = arguments.lapse
        lapse_rows = [{"lapse": lapse, "pooled_cv_selection_nll": None}]
    elif arguments.skip_lapse_search:
        raise ValueError("--skip-lapse-search requires --lapse")
    else:
        lapse, lapse_rows = fit_global_lapse(participant_data)
    print(f"global lapse={lapse}", flush=True)

    fits, trial_rows = [], []
    for participant, memory, trials in participant_data:
        participant_fits, participant_trials = fit_participant(
            participant, memory, trials, lapse
        )
        for row in participant_fits:
            row["locked_feature"] = arguments.locked_feature or ""
            row["fixed_cardinality"] = arguments.fixed_cardinality or ""
        for row in participant_trials:
            row["locked_feature"] = arguments.locked_feature or ""
            row["fixed_cardinality"] = arguments.fixed_cardinality or ""
        fits.extend(participant_fits)
        trial_rows.extend(participant_trials)
        print(f"fitted {participant}", flush=True)
    fits.sort(key=lambda row: (row["participant"], row["variant"]))
    trial_rows.sort(key=lambda row: (row["participant"], row["variant"], row["trial_order"]))
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output / "fits.csv", fits)
    write_csv(arguments.output / "trial_predictions.csv", trial_rows)
    write_csv(arguments.output / "global_lapse.csv", lapse_rows)
    summary = summarize(fits, lapse_rows, retained_only=False)
    retained = summarize(fits, lapse_rows, retained_only=True)
    with (arguments.output / "summary.json").open("w", encoding="utf-8") as destination:
        json.dump(
            {"all_complete": summary, "retained_cohort": retained},
            destination,
            indent=2,
            allow_nan=False,
        )
    (arguments.output / "REPORT.md").write_text(
        make_report(summary, retained), encoding="utf-8"
    )
    print(f"fits={len(fits)} trials={len(trial_rows)} output={arguments.output}")


if __name__ == "__main__":
    main()
