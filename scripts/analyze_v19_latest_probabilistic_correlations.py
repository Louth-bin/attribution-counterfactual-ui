"""Correlate latest additive-rho model fits with latest-cohort outcomes."""

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
FITS = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_fits.csv"
OUTPUT = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_correlations.json"
REPORT = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_report.md"

FIT_LABELS = {
    "selection_log_fit": "Selection log-fit (-CV NLL)",
    "selection_f1": "Expected selection F1",
    "amount_fit": "Conditional amount fit (-CV MAE)",
    "direction_accuracy": "Direction accuracy",
    "combined_fit": "Combined fit (-CV loss)",
}
OUTCOME_LABELS = {
    "success_rate": "Counterfactual success rate",
    "move_towards_rate": "Moved-toward-target rate",
    "boundary_improvement": "Boundary-distance improvement",
    "plausibility": "Plausibility",
    "target_confidence_gain": "Target-confidence gain",
    "actionability": "Actionability rate",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def value(row: dict[str, str], name: str) -> float:
    return float(row[name])


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def partial_rank(x: np.ndarray, y: np.ndarray, conditions: list[str]) -> tuple[float, float]:
    design = np.column_stack(
        [
            np.ones(len(x)),
            [condition == "attribution" for condition in conditions],
            [condition == "counterfactual" for condition in conditions],
        ]
    ).astype(float)
    ranked_x = stats.rankdata(x)
    ranked_y = stats.rankdata(y)
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    rho = float(np.corrcoef(residual_x, residual_y)[0, 1])
    df = len(x) - design.shape[1] - 1
    statistic = rho * math.sqrt(df / max(1e-15, 1.0 - rho**2))
    return rho, float(2.0 * stats.t.sf(abs(statistic), df))


def main() -> None:
    all_fits = read_csv(FITS)
    winners = {
        row["participant"]: row for row in all_fits if row["selected family by CV"] == "1"
    }
    testing: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(PROCESSED):
        if row["phase"] == "testing":
            testing[row["participant"]].append(row)
    participants = sorted(set(winners) & set(testing))
    if len(participants) != 25:
        raise RuntimeError(f"Expected 25 participants, found {len(participants)}")

    participant_rows = []
    for participant in participants:
        fit = winners[participant]
        trials = testing[participant]
        if len(trials) != 20:
            raise RuntimeError(f"Expected 20 testing trials for {participant}")
        participant_rows.append(
            {
                "participant": participant,
                "xai": fit["xai"],
                "model_family": fit["model family"],
                "selection_log_fit": -value(fit, "5-fold CV selection_nll"),
                "selection_f1": value(fit, "5-fold CV selection_f1"),
                "amount_fit": -value(fit, "5-fold CV amount_mae"),
                "direction_accuracy": value(fit, "5-fold CV direction_accuracy"),
                "combined_fit": -value(fit, "5-fold CV loss"),
                "success_rate": float(
                    np.mean([value(row, "successful counterfactual (0/1)") for row in trials])
                ),
                "move_towards_rate": float(
                    np.mean([value(row, "move towards target (0/1)") for row in trials])
                ),
                "boundary_improvement": float(
                    -np.mean([value(row, "boundary distance change (new - original)") for row in trials])
                ),
                "plausibility": float(np.mean([value(row, "plausibility") for row in trials])),
                "target_confidence_gain": float(
                    np.mean([value(row, "delta confidence of target label") for row in trials])
                ),
                "actionability": float(
                    np.mean([value(row, "actionability (0/1)") for row in trials])
                ),
            }
        )

    conditions = [row["xai"] for row in participant_rows]
    correlations = []
    for fit_name in FIT_LABELS:
        for outcome_name in OUTCOME_LABELS:
            x = np.asarray([row[fit_name] for row in participant_rows], dtype=float)
            y = np.asarray([row[outcome_name] for row in participant_rows], dtype=float)
            rank = stats.spearmanr(x, y)
            partial_rho, partial_p = partial_rank(x, y, conditions)
            correlations.append(
                {
                    "fit_metric": fit_name,
                    "outcome": outcome_name,
                    "n": len(x),
                    "spearman_rho": float(rank.statistic),
                    "spearman_p": float(rank.pvalue),
                    "partial_spearman_rho_controlling_xai": partial_rho,
                    "partial_spearman_p_controlling_xai": partial_p,
                }
            )
    for source, target in (
        ("spearman_p", "spearman_holm_p"),
        ("partial_spearman_p_controlling_xai", "partial_spearman_holm_p_controlling_xai"),
    ):
        adjusted = holm([row[source] for row in correlations])
        for row, adjusted_p in zip(correlations, adjusted):
            row[target] = adjusted_p

    family_counts = dict(
        sorted(
            (family, sum(row["model_family"] == family for row in participant_rows))
            for family in {row["model_family"] for row in participant_rows}
        )
    )
    result = {
        "model_version": "latest additive-rho probabilistic attribute-selection model",
        "participants": 25,
        "unit_of_analysis": "participant; outcomes averaged across 20 testing trials",
        "winner_counts": family_counts,
        "fit_metric_orientation": "all fit metrics are oriented higher-is-better",
        "correlation_method": "Spearman correlations and partial rank correlations controlling XAI condition; Holm adjustment across 30 pairs",
        "correlations": correlations,
        "participant_metrics": participant_rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    strongest = sorted(
        correlations,
        key=lambda row: abs(row["partial_spearman_rho_controlling_xai"]),
        reverse=True,
    )[:10]
    lines = [
        "# Latest additive-rho cognitive-model results",
        "",
        "This report supersedes the earlier perfect-memory exemplar/attribution/prototype analysis for the v1.9 latest-instance cohort.",
        "",
        "## Model-family selection",
        "",
        f"Feature contribution was selected for {family_counts.get('feature contribution', 0)} participants and weighted examples for {family_counts.get('weighted examples', 0)} participants.",
        "",
        "## Strongest condition-adjusted fit/outcome correlations",
        "",
        "| Fit metric | Outcome | Partial Spearman rho | p | Holm p |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in strongest:
        lines.append(
            f"| {FIT_LABELS[row['fit_metric']]} | {OUTCOME_LABELS[row['outcome']]} | "
            f"{row['partial_spearman_rho_controlling_xai']:.3f} | "
            f"{row['partial_spearman_p_controlling_xai']:.4f} | "
            f"{row['partial_spearman_holm_p_controlling_xai']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All fit metrics are oriented so that higher means a better fit. Testing outcomes are participant averages, preventing repeated trials from being treated as independent observations.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"correlations", "participant_metrics"}}, indent=2))


if __name__ == "__main__":
    main()
