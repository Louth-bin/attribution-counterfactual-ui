"""Relate latest-instance cognitive-model fits to participant outcomes.

The unit of analysis is the participant.  Testing outcomes are averaged over
the participant's 20 testing trials before correlations are calculated, which
avoids treating repeated trials as independent observations.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "qualtrics" / "v1.9_latest_processed.csv"
FORWARD = ROOT / "qualtrics" / "v1.9_latest_forward_fits.csv"
STRATEGY = ROOT / "qualtrics" / "v1.9_latest_strategy_components.csv"
OUTPUT_JSON = ROOT / "qualtrics" / "v1.9_latest_cognitive_correlations.json"
OUTPUT_REPORT = ROOT / "qualtrics" / "v1.9_latest_cognitive_model_report.md"


FIT_LABELS = {
    "forward_fit_quality": "Forward fit quality (-minimum NLL)",
    "forward_response_accuracy": "Best forward-model response accuracy",
    "forward_separation": "Forward-model separation (delta BIC)",
    "selection_ranking_score": "Attribute-selection ranking score",
    "selection_average_precision": "Attribute-selection average precision",
    "amount_fit_quality": "Change-amount fit quality (-MAE)",
}

OUTCOME_LABELS = {
    "testing_success_rate": "Counterfactual success rate",
    "testing_continuous_success": "Continuous counterfactual success",
    "testing_move_towards_rate": "Moved toward target rate",
    "testing_boundary_improvement": "Boundary-distance improvement",
    "testing_plausibility": "Plausibility",
    "testing_delta_target_confidence": "Increase in target confidence",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def number(row: dict[str, str], name: str) -> float:
    value = row[name].strip()
    return float(value) if value else math.nan


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def partial_rank_correlation(
    x: np.ndarray, y: np.ndarray, conditions: list[str]
) -> tuple[float, float]:
    ranked_x = stats.rankdata(x)
    ranked_y = stats.rankdata(y)
    design = np.column_stack(
        [
            np.ones(len(x)),
            np.asarray([condition == "attribution" for condition in conditions], dtype=float),
            np.asarray([condition == "counterfactual" for condition in conditions], dtype=float),
        ]
    )
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    correlation = float(np.corrcoef(residual_x, residual_y)[0, 1])
    degrees_freedom = len(x) - design.shape[1] - 1
    if abs(correlation) >= 1:
        p_value = 0.0
    else:
        statistic = correlation * math.sqrt(
            degrees_freedom / max(1e-15, 1.0 - correlation**2)
        )
        p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_freedom))
    return correlation, p_value


def main() -> None:
    forward_rows = read_csv(FORWARD)
    strategy_rows = read_csv(STRATEGY)
    forward = {row["participant"]: row for row in forward_rows}
    strategy = {row["participant"]: row for row in strategy_rows}

    testing = defaultdict(list)
    for row in read_csv(PROCESSED):
        if row["phase"] == "testing":
            testing[row["participant"]].append(row)

    participants = sorted(set(forward) & set(strategy) & set(testing))
    if len(participants) != 25:
        raise RuntimeError(f"Expected 25 participants, found {len(participants)}")

    participant_rows = []
    for participant in participants:
        frow = forward[participant]
        srow = strategy[participant]
        trials = testing[participant]
        if len(trials) != 20:
            raise RuntimeError(f"Expected 20 testing trials for {participant}")
        best_model = frow["best_forward_mental_model"]
        boundary_change = np.asarray(
            [number(row, "boundary distance change (new - original)") for row in trials]
        )
        participant_rows.append(
            {
                "participant": participant,
                "xai": frow["xai"],
                "forward_fit_quality": -number(frow, f"{best_model}_nll"),
                "forward_response_accuracy": number(frow, f"{best_model}_accuracy"),
                "forward_separation": number(frow, "delta_bic_to_second_best"),
                "selection_ranking_score": number(srow, "attribute selection ranking score"),
                "selection_average_precision": number(srow, "attribute selection average precision"),
                "amount_fit_quality": -number(srow, "change amount normalized MAE"),
                "testing_success_rate": float(
                    np.mean([number(row, "successful counterfactual (0/1)") for row in trials])
                ),
                "testing_continuous_success": float(
                    np.mean([number(row, "successful counterfactual (continuous)") for row in trials])
                ),
                "testing_move_towards_rate": float(
                    np.mean([number(row, "move towards target (0/1)") for row in trials])
                ),
                "testing_boundary_improvement": float(-np.mean(boundary_change)),
                "testing_plausibility": float(
                    np.mean([number(row, "plausibility") for row in trials])
                ),
                "testing_delta_target_confidence": float(
                    np.mean([number(row, "delta confidence of target label") for row in trials])
                ),
            }
        )

    correlation_rows = []
    conditions = [row["xai"] for row in participant_rows]
    for fit_name in FIT_LABELS:
        for outcome_name in OUTCOME_LABELS:
            x = np.asarray([row[fit_name] for row in participant_rows], dtype=float)
            y = np.asarray([row[outcome_name] for row in participant_rows], dtype=float)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise RuntimeError(f"Non-finite correlation input: {fit_name}, {outcome_name}")
            spearman = stats.spearmanr(x, y)
            pearson = stats.pearsonr(x, y)
            partial_rho, partial_p = partial_rank_correlation(x, y, conditions)
            correlation_rows.append(
                {
                    "fit_metric": fit_name,
                    "outcome": outcome_name,
                    "n": len(x),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "partial_spearman_rho_controlling_xai": partial_rho,
                    "partial_spearman_p_controlling_xai": partial_p,
                }
            )

    for p_name, adjusted_name in (
        ("spearman_p", "spearman_holm_p"),
        ("partial_spearman_p_controlling_xai", "partial_spearman_holm_p_controlling_xai"),
    ):
        adjusted = holm_adjust([row[p_name] for row in correlation_rows])
        for row, value in zip(correlation_rows, adjusted):
            row[adjusted_name] = value

    forward_summary = {}
    for model in ("exemplar", "attribution", "prototype"):
        forward_summary[model] = {
            "strict_best_count": sum(
                row["best_forward_mental_model"] == model for row in forward_rows
            ),
            "supported_delta_bic_lt_2_count": sum(
                model in row["supported_forward_mental_models_delta_bic_lt_2"].split(" / ")
                for row in forward_rows
            ),
            "mean_nll": float(np.mean([number(row, f"{model}_nll") for row in forward_rows])),
            "mean_accuracy": float(
                np.mean([number(row, f"{model}_accuracy") for row in forward_rows])
            ),
        }

    selection_counts = defaultdict(int)
    amount_counts = defaultdict(int)
    for row in strategy_rows:
        selection_counts[row["attribute selection method"]] += 1
        amount_counts[row["change amount method"]] += 1

    result = {
        "scope": "25 participants who saw the latest v1.6/160xxx instances",
        "unit_of_analysis": "participant; testing metrics averaged across 20 trials",
        "correlation_interpretation": "All fit-quality metrics and outcomes are oriented so higher is better. Primary correlations are Spearman; partial rank correlations control for XAI condition. Holm p-values adjust across the 36 fit-outcome pairs.",
        "forward_models": forward_summary,
        "forward_identifiability": {
            "one_supported_model": sum(
                len(row["supported_forward_mental_models_delta_bic_lt_2"].split(" / ")) == 1
                for row in forward_rows
            ),
            "two_supported_models": sum(
                len(row["supported_forward_mental_models_delta_bic_lt_2"].split(" / ")) == 2
                for row in forward_rows
            ),
            "three_supported_models": sum(
                len(row["supported_forward_mental_models_delta_bic_lt_2"].split(" / ")) == 3
                for row in forward_rows
            ),
        },
        "counterfactual_components": {
            "selection_method_counts": dict(sorted(selection_counts.items())),
            "amount_method_counts": dict(sorted(amount_counts.items())),
            "mean_selection_ranking_score": float(
                np.mean([number(row, "attribute selection ranking score") for row in strategy_rows])
            ),
            "mean_selection_average_precision": float(
                np.mean([number(row, "attribute selection average precision") for row in strategy_rows])
            ),
            "mean_change_amount_normalized_mae": float(
                np.mean([number(row, "change amount normalized MAE") for row in strategy_rows])
            ),
        },
        "correlations": correlation_rows,
        "participant_metrics": participant_rows,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    strongest = sorted(
        correlation_rows,
        key=lambda row: abs(row["partial_spearman_rho_controlling_xai"]),
        reverse=True,
    )[:10]
    lines = [
        "# Latest-instance cognitive-model fits and correlations",
        "",
        "The analysis includes all 25 participants who saw the v1.6/160xxx instances. Correlations use one row per participant; each testing outcome is averaged across that participant's 20 trials.",
        "",
        "## Forward-model fits",
        "",
        "| Model | Strict best | Supported (delta BIC < 2) | Mean NLL | Mean response accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model, values in forward_summary.items():
        lines.append(
            f"| {model.title()} | {values['strict_best_count']} | {values['supported_delta_bic_lt_2_count']} | {values['mean_nll']:.3f} | {values['mean_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All three forward models were within delta BIC < 2 for every participant, so strict-best counts are descriptive rather than identifiable classifications.",
            "",
            "## Counterfactual-component fits",
            "",
            f"Mean attribute-selection ranking score: {result['counterfactual_components']['mean_selection_ranking_score']:.3f}.",
            f"Mean attribute-selection average precision: {result['counterfactual_components']['mean_selection_average_precision']:.3f}.",
            f"Mean normalized change-amount MAE: {result['counterfactual_components']['mean_change_amount_normalized_mae']:.3f}.",
            "",
            "## Strongest condition-adjusted correlations",
            "",
            "| Fit measure | Outcome | Partial Spearman rho | p | Holm p |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in strongest:
        lines.append(
            "| "
            + FIT_LABELS[row["fit_metric"]]
            + " | "
            + OUTCOME_LABELS[row["outcome"]]
            + f" | {row['partial_spearman_rho_controlling_xai']:.3f}"
            + f" | {row['partial_spearman_p_controlling_xai']:.4f}"
            + f" | {row['partial_spearman_holm_p_controlling_xai']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Higher values mean better fit and better outcomes throughout this table. The forward and amount metrics were sign-reversed from NLL and MAE respectively. Holm correction covers all 36 fit-outcome comparisons.",
            "",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"correlations", "participant_metrics"}}, indent=2))


if __name__ == "__main__":
    main()
