"""Cross-validated parameter ablations for cognitive model v0.1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_models.v0_1.model import (  # noqa: E402
    RHO_GRID,
    evaluate,
    observed_trials,
    selection_parameter_grid,
)


RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
FITS = ROOT / "qualtrics" / "v20_cognitive_model_v01_fits.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTPUT_DIR = ROOT / "outputs" / "v20-v01-parameter-ablation"

PARAMETERS = {
    "eta": {
        "label": "Explanation reliance (η)",
        "null": 0.0,
        "primary metric": "selection_nll",
        "applies": lambda family, condition: condition != "none",
    },
    "alpha": {
        "label": "Global relevance reliance (α)",
        "null": 0.0,
        "primary metric": "selection_nll",
        "applies": lambda family, condition: family == "feature contribution",
    },
    "rho": {
        "label": "Additive change margin (ρ)",
        "null": 0.0,
        "primary metric": "amount_mae",
        "applies": lambda family, condition: True,
    },
    "lambda": {
        "label": "Exemplar locality (λ)",
        "null": 0.0,
        "primary metric": "selection_nll",
        "applies": lambda family, condition: family == "weighted examples",
    },
    "beta": {
        "label": "Remembered-change reliance (β)",
        "null": 0.0,
        "primary metric": "selection_nll",
        "applies": lambda family, condition: (
            family == "weighted examples" and condition == "counterfactual"
        ),
    },
    "age actionable": {
        "label": "Age-actionability flexibility",
        "null": 1,
        "primary metric": "selection_nll",
        "applies": lambda family, condition: True,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def fit_indices_restricted(
    family: str,
    condition: str,
    training: list[dict],
    trials: list[dict],
    candidates: list[dict],
    rho_values: tuple[float, ...],
    indices: list[int] | None = None,
) -> tuple[dict[str, float], dict]:
    selection_scores = [
        (
            evaluate(family, condition, candidate, training, trials, indices),
            candidate,
        )
        for candidate in candidates
    ]
    best_nll = min(score["selection_nll"] for score, _ in selection_scores)
    selection_winners = [
        candidate
        for score, candidate in selection_scores
        if math.isclose(
            score["selection_nll"], best_nll, rel_tol=1e-12, abs_tol=1e-12
        )
    ]
    amount_scores = []
    for selection_parameters in selection_winners:
        for rho in rho_values:
            parameters = {**selection_parameters, "rho": rho}
            amount_scores.append(
                (
                    evaluate(
                        family, condition, parameters, training, trials, indices
                    ),
                    parameters,
                )
            )
    return min(amount_scores, key=lambda item: item[0]["amount_mae"])


def cross_validated_ablation(
    family: str,
    condition: str,
    training: list[dict],
    trials: list[dict],
    parameter: str,
    fixed_value: float | int | None = None,
) -> tuple[dict[str, float], dict]:
    definition = PARAMETERS[parameter]
    null_value = definition["null"] if fixed_value is None else fixed_value
    candidates = list(selection_parameter_grid(condition, family))
    if parameter != "rho":
        candidates = [
            candidate
            for candidate in candidates
            if candidate[parameter] == null_value
        ]
    rho_values = (float(null_value),) if parameter == "rho" else RHO_GRID
    if not candidates:
        raise RuntimeError(
            f"Ablation {parameter} produced no candidates for {family}/{condition}"
        )

    in_sample, fitted_parameters = fit_indices_restricted(
        family, condition, training, trials, candidates, rho_values
    )
    fold_count = min(5, len(trials))
    folds = []
    for fold in range(fold_count):
        train_indices = [
            index for index in range(len(trials)) if index % fold_count != fold
        ]
        test_indices = [
            index for index in range(len(trials)) if index % fold_count == fold
        ]
        _, fold_parameters = fit_indices_restricted(
            family,
            condition,
            training,
            trials,
            candidates,
            rho_values,
            train_indices,
        )
        folds.append(
            evaluate(
                family,
                condition,
                fold_parameters,
                training,
                trials,
                test_indices,
            )
        )
    cross_validated = {
        name: float(np.nanmean([fold[name] for fold in folds]))
        for name in in_sample
    }
    return cross_validated, fitted_parameters


def ablate_participant(arguments: tuple) -> list[dict[str, object]]:
    participant, rows, fit, training, case_map, qualified = arguments
    family = fit["model family"]
    condition = fit["xai"]
    trials = observed_trials(rows, case_map)
    output = []
    for parameter, definition in PARAMETERS.items():
        if not definition["applies"](family, condition):
            continue
        ablated, fitted_parameters = cross_validated_ablation(
            family, condition, training, trials, parameter
        )
        full_metrics = {
            name: float(fit[f"5-fold CV {name}"])
            for name in (
                "loss",
                "selection_f1",
                "amount_mae",
                "direction_accuracy",
                "selection_nll",
            )
        }
        full_value = fit[parameter]
        full_value_number = (
            math.nan if full_value in ("", "nan") else float(full_value)
        )
        output.append(
            {
                "participant": participant,
                "direction-qualified cohort (0/1)": int(qualified),
                "xai": condition,
                "model family held fixed": family,
                "parameter": parameter,
                "parameter label": definition["label"],
                "null value": definition["null"],
                "full fitted value": full_value_number,
                "full value differs from null (0/1)": int(
                    math.isfinite(full_value_number)
                    and full_value_number != definition["null"]
                ),
                "primary metric": definition["primary metric"],
                **{
                    f"full CV {name}": value
                    for name, value in full_metrics.items()
                },
                **{
                    f"ablated CV {name}": value
                    for name, value in ablated.items()
                },
                **{
                    f"delta CV {name} (ablated - full)": ablated[name]
                    - full_metrics[name]
                    for name in full_metrics
                },
                "ablated fitted eta": fitted_parameters["eta"],
                "ablated fitted alpha": fitted_parameters["alpha"],
                "ablated fitted rho": fitted_parameters["rho"],
                "ablated fitted lambda": fitted_parameters["lambda"],
                "ablated fitted beta": fitted_parameters["beta"],
                "ablated fitted age actionable": fitted_parameters[
                    "age actionable"
                ],
            }
        )
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def bootstrap_mean_interval(
    values: np.ndarray, seed: int, repetitions: int = 20_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(values, size=(repetitions, len(values)), replace=True), axis=1
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    detail_rows: list[dict[str, object]], cohort: str
) -> list[dict[str, object]]:
    eligible = [
        row
        for row in detail_rows
        if cohort == "all 62" or row["direction-qualified cohort (0/1)"] == 1
    ]
    by_parameter: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        by_parameter[str(row["parameter"])].append(row)
    summary = []
    for parameter in PARAMETERS:
        rows = by_parameter.get(parameter, [])
        if not rows:
            continue
        primary_metric = str(rows[0]["primary metric"])
        delta_column = f"delta CV {primary_metric} (ablated - full)"
        deltas = np.asarray([float(row[delta_column]) for row in rows])
        nonzero = deltas[np.abs(deltas) > 1e-12]
        if len(nonzero):
            test = stats.wilcoxon(
                deltas,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            )
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        else:
            p_value = 1.0
            statistic = 0.0
        low, high = bootstrap_mean_interval(
            deltas, 20260831 + list(PARAMETERS).index(parameter) + 100 * len(cohort)
        )
        summary.append(
            {
                "cohort": cohort,
                "parameter": parameter,
                "parameter label": PARAMETERS[parameter]["label"],
                "null value": PARAMETERS[parameter]["null"],
                "primary metric": primary_metric,
                "n applicable participants": len(rows),
                "n full fitted values differing from null": sum(
                    int(row["full value differs from null (0/1)"]) for row in rows
                ),
                "mean delta (ablated - full; positive supports parameter)": float(
                    np.mean(deltas)
                ),
                "median delta": float(np.median(deltas)),
                "bootstrap mean delta 95% CI low": low,
                "bootstrap mean delta 95% CI high": high,
                "participants improved by full parameter": int(np.sum(deltas > 1e-12)),
                "participants tied": int(np.sum(np.abs(deltas) <= 1e-12)),
                "participants worse with full parameter": int(np.sum(deltas < -1e-12)),
                "one-sided Wilcoxon statistic": statistic,
                "one-sided Wilcoxon p": p_value,
            }
        )
    adjusted = holm_adjust([float(row["one-sided Wilcoxon p"]) for row in summary])
    for row, adjusted_p in zip(summary, adjusted):
        row["Holm-adjusted p"] = adjusted_p
        mean_delta = float(
            row["mean delta (ablated - full; positive supports parameter)"]
        )
        ci_low = float(row["bootstrap mean delta 95% CI low"])
        if mean_delta > 0 and ci_low > 0 and adjusted_p < 0.05:
            conclusion = "useful"
        elif mean_delta > 0 and float(row["one-sided Wilcoxon p"]) < 0.05:
            conclusion = "limited evidence"
        elif mean_delta <= 0:
            conclusion = "not useful on held-out performance"
        else:
            conclusion = "inconclusive"
        row["conclusion"] = conclusion
    return summary


def make_plot(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    primary = {
        str(row["parameter"]): row
        for row in summary_rows
        if row["cohort"] == "direction-qualified 46"
    }
    sensitivity = {
        str(row["parameter"]): row
        for row in summary_rows
        if row["cohort"] == "all 62"
    }
    selection_parameters = [
        parameter
        for parameter in PARAMETERS
        if PARAMETERS[parameter]["primary metric"] == "selection_nll"
    ]
    amount_parameters = ["rho"]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    cohorts = (
        (primary, "Direction-qualified 46", "o", "#0072B2", -0.10),
        (sensitivity, "All 62", "s", "#D55E00", 0.10),
    )
    for axis, parameters, title, xlabel in (
        (
            axes[0],
            selection_parameters,
            "Selection-parameter ablations",
            "Increase in held-out selection NLL when ablated",
        ),
        (
            axes[1],
            amount_parameters,
            "Change-amount ablation",
            "Increase in held-out amount MAE when ablated",
        ),
    ):
        positions = np.arange(len(parameters))
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        for rows, label, marker, color, offset in cohorts:
            means = np.asarray(
                [
                    rows[p][
                        "mean delta (ablated - full; positive supports parameter)"
                    ]
                    for p in parameters
                ],
                dtype=float,
            )
            lows = np.asarray(
                [rows[p]["bootstrap mean delta 95% CI low"] for p in parameters],
                dtype=float,
            )
            highs = np.asarray(
                [rows[p]["bootstrap mean delta 95% CI high"] for p in parameters],
                dtype=float,
            )
            axis.errorbar(
                means,
                positions + offset,
                xerr=np.vstack([means - lows, highs - means]),
                fmt=marker,
                color=color,
                capsize=3,
                markersize=6,
                linewidth=1.2,
                label=label,
            )
        axis.set_yticks(
            positions,
            [str(PARAMETERS[p]["label"]) for p in parameters],
        )
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        "Cognitive model v0.1 parameter ablation\nPositive values mean the fitted parameter improves held-out performance",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_dir / "v01_parameter_ablation.png", dpi=300)
    figure.savefig(output_dir / "v01_parameter_ablation.pdf")
    plt.close(figure)


def write_report(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    primary = [
        row
        for row in summary_rows
        if row["cohort"] == "direction-qualified 46"
    ]
    lines = [
        "# Cognitive model v0.1 parameter ablation",
        "",
        "Primary analysis uses the 46 participants with at least 70% of edits moving toward the target. The all-62 analysis is a sensitivity check. Each participant's winning model family is held fixed. Positive deltas mean held-out performance worsened when the parameter was fixed to its null value.",
        "",
        "| Parameter | Primary metric | Applicable n | Mean delta [95% bootstrap CI] | Holm p | Conclusion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in primary:
        lines.append(
            "| {label} | {metric} | {n} | {mean:.4f} [{low:.4f}, {high:.4f}] | {p:.4g} | {conclusion} |".format(
                label=row["parameter label"],
                metric=row["primary metric"],
                n=row["n applicable participants"],
                mean=row[
                    "mean delta (ablated - full; positive supports parameter)"
                ],
                low=row["bootstrap mean delta 95% CI low"],
                high=row["bootstrap mean delta 95% CI high"],
                p=row["Holm-adjusted p"],
                conclusion=row["conclusion"],
            )
        )
    lines.extend(
        [
            "",
            "Selection parameters are evaluated with five-fold cross-validated selection NLL. The additive margin ρ is evaluated with five-fold cross-validated conditional amount MAE. One-sided paired Wilcoxon tests ask whether ablation worsens performance; Holm adjustment covers the six parameter tests within each cohort.",
        ]
    )
    (output_dir / "v01_parameter_ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    parser.add_argument("--participant", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_rows = read_csv(RESULTS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    direction_values: dict[str, list[float]] = defaultdict(list)
    for row in result_rows:
        grouped[row["participant"]].append(row)
        if row["phase"] == "testing" and row["move towards target (0/1)"] != "":
            direction_values[row["participant"]].append(
                float(row["move towards target (0/1)"])
            )
    qualified = {
        participant
        for participant, values in direction_values.items()
        if len(values) == 20 and float(np.mean(values)) >= 0.70
    }
    fits = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["selected family by CV"] == "1"
    }
    if set(grouped) != set(fits):
        raise RuntimeError("Results and selected-fit participant sets differ")
    selected_participants = sorted(grouped)
    if args.participant:
        selected_participants = [
            participant
            for participant in selected_participants
            if participant in set(args.participant)
        ]

    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training = dataset["training_pool"]
    jobs = [
        (
            participant,
            grouped[participant],
            fits[participant],
            training,
            case_map,
            participant in qualified,
        )
        for participant in selected_participants
    ]

    detail_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {
            executor.submit(ablate_participant, job): job[0] for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            participant = futures[future]
            detail_rows.extend(future.result())
            print(
                f"ablated {participant} ({completed}/{len(futures)})", flush=True
            )
    detail_rows.sort(key=lambda row: (str(row["participant"]), str(row["parameter"])))
    detail_output = OUTPUT_DIR / "v01_parameter_ablation_participants.csv"
    write_csv(detail_output, detail_rows)

    summary_rows = summarize(detail_rows, "direction-qualified 46")
    summary_rows.extend(summarize(detail_rows, "all 62"))
    summary_output = OUTPUT_DIR / "v01_parameter_ablation_summary.csv"
    write_csv(summary_output, summary_rows)
    summary_json = {
        "model": "cognitive model v0.1",
        "participants fitted": len(selected_participants),
        "direction-qualified participants": len(qualified),
        "method": (
            "Winning family held fixed; each parameter fixed to its null value; "
            "five-fold CV refits all remaining parameters within each fold."
        ),
        "null values": {
            parameter: definition["null"]
            for parameter, definition in PARAMETERS.items()
        },
        "positive delta interpretation": (
            "Ablated held-out error minus full-model held-out error; positive "
            "values support parameter usefulness."
        ),
        "summaries": summary_rows,
    }
    (OUTPUT_DIR / "v01_parameter_ablation_summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary_rows, OUTPUT_DIR)
    make_plot(summary_rows, OUTPUT_DIR)
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
