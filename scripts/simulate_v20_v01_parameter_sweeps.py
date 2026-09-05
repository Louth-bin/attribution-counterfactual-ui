"""Simulate v0.1 desiderata across parameter grids and XAI conditions.

The analysis changes one parameter at a time around each participant's fitted
winning-family specification. Each prediction uses the maximum-probability
feature subset conditional on that trial's observed edit count K, matching the
earlier v0.1 participant-versus-model analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_v19_latest_model_simulations import (  # noqa: E402
    exact_boundaries,
    prediction_label,
    target_probability,
)
from scripts.run_model_only_strategy_study import (  # noqa: E402
    reference_matrices,
    similarity,
)
from src.cognitive_models.v0_1.model import (  # noqa: E402
    ALPHA_GRID,
    BETA_GRID,
    ETA_GRID,
    LAMBDA_GRID,
    RHO_GRID,
    build_memory,
    contribution_distribution,
    exemplar_distribution,
    observed_trials,
)


RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
FITS = ROOT / "qualtrics" / "v20_cognitive_model_v01_fits.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTPUT_DIR = ROOT / "outputs" / "v20-v01-parameter-sweeps"

PARAMETERS = {
    "eta": {
        "label": "Explanation reliance (η)",
        "short": "eta",
        "values": ETA_GRID,
        "applies": lambda family, xai: xai != "none",
    },
    "alpha": {
        "label": "Global relevance reliance (α)",
        "short": "alpha",
        "values": ALPHA_GRID,
        "applies": lambda family, xai: family == "feature contribution",
    },
    "rho": {
        "label": "Additive change margin (ρ)",
        "short": "rho",
        "values": RHO_GRID,
        "applies": lambda family, xai: True,
    },
    "lambda": {
        "label": "Exemplar locality (λ)",
        "short": "lambda",
        "values": LAMBDA_GRID,
        "applies": lambda family, xai: family == "weighted examples",
    },
    "beta": {
        "label": "Remembered-change reliance (β)",
        "short": "beta",
        "values": BETA_GRID,
        "applies": lambda family, xai: (
            family == "weighted examples" and xai == "counterfactual"
        ),
    },
    "age actionable": {
        "label": "Age actionable",
        "short": "age-actionable",
        "values": (0, 1),
        "applies": lambda family, xai: True,
    },
}

METRICS = {
    "success": "Counterfactual success rate",
    "target confidence gain": "Target-confidence gain",
    "boundary improvement": "Boundary-distance improvement",
    "plausibility": "Plausibility",
    "actionability": "Actionability rate",
    "edit L1": "Normalized edit distance (L1)",
}

XAI_LABELS = {
    "none": "No XAI",
    "attribution": "Feature attribution",
    "counterfactual": "Counterfactual example",
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


def fitted_parameters(fit: dict[str, str]) -> dict[str, float | int]:
    parameters: dict[str, float | int] = {
        name: float(fit[name]) if fit[name] not in ("", "nan") else math.nan
        for name in ("eta", "alpha", "rho", "lambda", "beta")
    }
    parameters["age actionable"] = int(float(fit["age actionable"]))
    parameters["immutable index"] = int(float(fit["immutable index"]))
    return parameters


def expected(samples: list[dict], function) -> float:
    return float(sum(sample["weight"] * function(sample) for sample in samples))


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    generator = np.random.default_rng(seed)
    means = np.mean(
        generator.choice(values, size=(5_000, len(values)), replace=True), axis=1
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def prepare_simulations(
    result_rows: list[dict[str, str]],
    fits: dict[str, dict[str, str]],
    dataset: dict,
) -> tuple[list[dict], dict[str, np.ndarray], set[str]]:
    browser_model = dataset["browser_model"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training = dataset["training_pool"]
    reference = reference_matrices("diabetes", dataset)[0]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    directions: dict[str, list[float]] = defaultdict(list)
    for row in result_rows:
        grouped[row["participant"]].append(row)
        if row["phase"] == "testing" and row["move towards target (0/1)"] != "":
            directions[row["participant"]].append(
                float(row["move towards target (0/1)"])
            )
    qualified = {
        participant
        for participant, values in directions.items()
        if len(values) == 20 and float(np.mean(values)) >= 0.70
    }

    pending: list[dict] = []
    boundary_profiles: dict[str, np.ndarray] = {}
    for participant, participant_rows in sorted(grouped.items()):
        fit = fits[participant]
        family = fit["model family"]
        xai = fit["xai"]
        base_parameters = fitted_parameters(fit)
        trials = observed_trials(participant_rows, case_map)
        sources = sorted(
            (row for row in participant_rows if row["phase"] == "testing"),
            key=lambda row: int(float(row["case"])),
        )
        for parameter, definition in PARAMETERS.items():
            if not definition["applies"](family, xai):
                continue
            for value in definition["values"]:
                parameters = dict(base_parameters)
                parameters[parameter] = value
                memory = build_memory(training, xai, float(parameters["eta"]))
                for trial_index, (trial, source) in enumerate(zip(trials, sources)):
                    case_id = int(float(source["instance id"]))
                    case = case_map[case_id]
                    if family == "feature contribution":
                        distribution = contribution_distribution(
                            trial["profile"],
                            trial["target"],
                            memory,
                            parameters,
                            trial["observed_k"],
                        )
                    else:
                        distribution = exemplar_distribution(
                            trial["profile"],
                            trial["target"],
                            memory,
                            xai,
                            parameters,
                            trial["observed_k"],
                        )
                    if not distribution:
                        continue
                    target_label = source["target label"]
                    original_probability = target_probability(
                        trial["profile"], target_label, case, browser_model
                    )
                    _, _, delta = max(distribution, key=lambda proposal: proposal[1])
                    profile = np.clip(trial["profile"] + delta, 0.0, 1.0)
                    profile_key = f"{participant}|{parameter}|{value}|{trial_index}"
                    boundary_profiles[profile_key] = profile
                    samples = [
                        {
                            "weight": 1.0,
                            "delta": delta,
                            "profile": profile,
                            "profile key": profile_key,
                        }
                    ]
                    pending.append(
                        {
                            "participant": participant,
                            "qualified": participant in qualified,
                            "xai": xai,
                            "model family": family,
                            "parameter": parameter,
                            "parameter value": float(value),
                            "case": source["case"],
                            "instance id": case_id,
                            "case data": case,
                            "source": source,
                            "target label": target_label,
                            "original probability": original_probability,
                            "reference": reference,
                            "samples": samples,
                        }
                    )
    return pending, boundary_profiles, qualified


def score_trials(
    pending: list[dict],
    boundaries: dict[str, float],
    browser_model: dict,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in pending:
        samples = item["samples"]
        source = item["source"]
        case = item["case data"]
        target_label = item["target label"]
        original_boundary = float(source["boundary distance original"])
        rows.append(
            {
                "participant": item["participant"],
                "direction-qualified cohort (0/1)": int(item["qualified"]),
                "xai": item["xai"],
                "model family": item["model family"],
                "parameter": item["parameter"],
                "parameter label": PARAMETERS[item["parameter"]]["label"],
                "parameter value": item["parameter value"],
                "case": item["case"],
                "instance id": item["instance id"],
                "success": expected(
                    samples,
                    lambda sample: float(
                        prediction_label(sample["profile"], case, browser_model)
                        == target_label
                    ),
                ),
                "target confidence gain": expected(
                    samples,
                    lambda sample: target_probability(
                        sample["profile"], target_label, case, browser_model
                    )
                    - item["original probability"],
                ),
                "boundary improvement": expected(
                    samples,
                    lambda sample: original_boundary
                    - boundaries[sample["profile key"]],
                ),
                "plausibility": expected(
                    samples,
                    lambda sample: similarity(sample["profile"], item["reference"]),
                ),
                "actionability": expected(
                    samples,
                    lambda sample: float(abs(sample["delta"][4]) <= 1e-9),
                ),
                "edit L1": expected(
                    samples, lambda sample: float(np.abs(sample["delta"]).sum())
                ),
            }
        )
    return rows


def participant_means(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        key = (
            row["participant"],
            row["direction-qualified cohort (0/1)"],
            row["xai"],
            row["model family"],
            row["parameter"],
            row["parameter label"],
            row["parameter value"],
        )
        grouped[key].append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append(
            {
                "participant": key[0],
                "direction-qualified cohort (0/1)": key[1],
                "xai": key[2],
                "model family": key[3],
                "parameter": key[4],
                "parameter label": key[5],
                "parameter value": key[6],
                "testing trials": len(rows),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in METRICS
                },
            }
        )
    return output


def aggregate(
    participant_rows: list[dict[str, object]], cohort: str
) -> list[dict[str, object]]:
    selected = (
        [row for row in participant_rows if row["direction-qualified cohort (0/1)"] == 1]
        if cohort == "direction-qualified 46"
        else participant_rows
    )
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[(row["xai"], row["parameter"], row["parameter value"])].append(row)
    output = []
    for group_index, (key, rows) in enumerate(sorted(grouped.items())):
        xai, parameter, value = key
        for metric, metric_label in METRICS.items():
            values = np.asarray([float(row[metric]) for row in rows])
            low, high = bootstrap_mean_ci(
                values, 20260831 + group_index * len(METRICS) + list(METRICS).index(metric)
            )
            output.append(
                {
                    "cohort": cohort,
                    "xai": xai,
                    "xai label": XAI_LABELS[xai],
                    "parameter": parameter,
                    "parameter label": PARAMETERS[parameter]["label"],
                    "parameter value": float(value),
                    "metric": metric,
                    "metric label": metric_label,
                    "applicable participants": len(rows),
                    "mean": float(values.mean()),
                    "95% bootstrap CI low": low,
                    "95% bootstrap CI high": high,
                }
            )
    return output


def endpoint_effects(
    participant_rows: list[dict[str, object]], cohort: str
) -> list[dict[str, object]]:
    selected = (
        [row for row in participant_rows if row["direction-qualified cohort (0/1)"] == 1]
        if cohort == "direction-qualified 46"
        else participant_rows
    )
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[(row["xai"], row["parameter"], row["participant"])].append(row)
    deltas: dict[tuple, list[float]] = defaultdict(list)
    for (xai, parameter, _), rows in grouped.items():
        low_row = min(rows, key=lambda row: float(row["parameter value"]))
        high_row = max(rows, key=lambda row: float(row["parameter value"]))
        for metric in METRICS:
            deltas[(xai, parameter, metric)].append(
                float(high_row[metric]) - float(low_row[metric])
            )
    output = []
    for group_index, (key, values_list) in enumerate(sorted(deltas.items())):
        xai, parameter, metric = key
        values = np.asarray(values_list)
        low, high = bootstrap_mean_ci(values, 20260901 + group_index)
        output.append(
            {
                "cohort": cohort,
                "xai": xai,
                "xai label": XAI_LABELS[xai],
                "parameter": parameter,
                "parameter label": PARAMETERS[parameter]["label"],
                "low endpoint": float(min(PARAMETERS[parameter]["values"])),
                "high endpoint": float(max(PARAMETERS[parameter]["values"])),
                "metric": metric,
                "metric label": METRICS[metric],
                "applicable participants": len(values),
                "mean change (high - low)": float(values.mean()),
                "95% bootstrap CI low": low,
                "95% bootstrap CI high": high,
            }
        )
    return output


def make_figure(
    rows: list[dict[str, object]], parameter: str, cohort: str
) -> plt.Figure:
    parameter_rows = [
        row
        for row in rows
        if row["cohort"] == cohort and row["parameter"] == parameter
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=False)
    fig.subplots_adjust(top=0.80, bottom=0.08, hspace=0.42, wspace=0.30)
    colors = {"none": "#6b7280", "attribution": "#0072b2", "counterfactual": "#d55e00"}
    markers = {"none": "o", "attribution": "s", "counterfactual": "^"}
    for axis, (metric, metric_label) in zip(axes.flat, METRICS.items()):
        metric_rows = [row for row in parameter_rows if row["metric"] == metric]
        for xai in ("none", "attribution", "counterfactual"):
            series = sorted(
                (row for row in metric_rows if row["xai"] == xai),
                key=lambda row: float(row["parameter value"]),
            )
            if not series:
                continue
            x = np.asarray([float(row["parameter value"]) for row in series])
            y = np.asarray([float(row["mean"]) for row in series])
            low = np.asarray([float(row["95% bootstrap CI low"]) for row in series])
            high = np.asarray([float(row["95% bootstrap CI high"]) for row in series])
            label = f"{XAI_LABELS[xai]} (n={series[0]['applicable participants']})"
            axis.plot(x, y, marker=markers[xai], color=colors[xai], label=label, linewidth=2)
            axis.fill_between(x, low, high, color=colors[xai], alpha=0.14)
        axis.set_title(metric_label)
        axis.set_xlabel(PARAMETERS[parameter]["label"])
        axis.set_ylabel("Mean simulated value")
        axis.grid(axis="y", alpha=0.25)
        if metric in ("success", "actionability"):
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:
        for axis in axes.flat[1:]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=min(3, len(labels)),
    )
    cohort_label = (
        "Direction-qualified participants (≥70% target-directed edits)"
        if cohort == "direction-qualified 46"
        else "All 62 participants"
    )
    fig.suptitle(
        f"v0.1 one-parameter simulation: {PARAMETERS[parameter]['label']}\n{cohort_label}",
        fontsize=16,
        y=0.985,
    )
    return fig


def write_figures(summary_rows: list[dict[str, object]]) -> None:
    for cohort, stem in (
        ("direction-qualified 46", "direction46"),
        ("all 62", "all62"),
    ):
        pdf_path = OUTPUT_DIR / f"v01_parameter_sweeps_{stem}.pdf"
        with PdfPages(pdf_path) as pdf:
            for parameter in PARAMETERS:
                fig = make_figure(summary_rows, parameter, cohort)
                fig.savefig(
                    OUTPUT_DIR
                    / f"v01_parameter_sweep_{PARAMETERS[parameter]['short']}_{stem}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def write_endpoint_overview(effects: list[dict[str, object]]) -> None:
    parameter_order = list(PARAMETERS)
    xai_order = ["none", "attribution", "counterfactual"]
    for cohort, stem in (
        ("direction-qualified 46", "direction46"),
        ("all 62", "all62"),
    ):
        cohort_rows = [row for row in effects if row["cohort"] == cohort]
        lookup = {
            (row["parameter"], row["xai"], row["metric"]): row
            for row in cohort_rows
        }
        row_keys = [
            (parameter, xai)
            for parameter in parameter_order
            for xai in xai_order
            if any(
                (parameter, xai, metric) in lookup for metric in METRICS
            )
        ]
        fig, axes = plt.subplots(
            1,
            len(METRICS),
            figsize=(17, 9.5),
            sharey=True,
            gridspec_kw={"wspace": 0.08},
        )
        for axis, (metric, metric_label) in zip(axes, METRICS.items()):
            values = np.asarray(
                [
                    float(lookup[(parameter, xai, metric)]["mean change (high - low)"])
                    if (parameter, xai, metric) in lookup
                    else math.nan
                    for parameter, xai in row_keys
                ]
            )
            maximum = max(float(np.nanmax(np.abs(values))), 1e-12)
            shown = values[:, None]
            axis.imshow(
                shown,
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
                aspect="auto",
            )
            axis.set_title(metric_label, fontsize=11)
            axis.set_xticks([])
            axis.set_yticks(range(len(row_keys)))
            if axis is axes[0]:
                labels = []
                for parameter, xai in row_keys:
                    row = lookup[(parameter, xai, metric)]
                    labels.append(
                        f"{PARAMETERS[parameter]['label']} · {XAI_LABELS[xai]} "
                        f"(n={row['applicable participants']})"
                    )
                axis.set_yticklabels(labels, fontsize=9)
            axis.tick_params(length=0)
            for row_index, value in enumerate(values):
                if math.isnan(float(value)):
                    continue
                color = "white" if abs(float(value)) / maximum > 0.58 else "black"
                axis.text(
                    0,
                    row_index,
                    f"{float(value):+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
        cohort_label = (
            "Direction-qualified participants (≥70% target-directed edits)"
            if cohort == "direction-qualified 46"
            else "All 62 participants"
        )
        fig.suptitle(
            "Change in simulated desiderata from lowest to highest parameter value\n"
            f"{cohort_label} · Color scaled separately within each desideratum",
            fontsize=15,
            y=0.985,
        )
        fig.subplots_adjust(left=0.34, right=0.99, top=0.88, bottom=0.04)
        fig.savefig(
            OUTPUT_DIR / f"v01_parameter_sweep_overview_{stem}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def write_report(
    summary_rows: list[dict[str, object]],
    effects: list[dict[str, object]],
    qualified_count: int,
) -> None:
    primary = [row for row in effects if row["cohort"] == "direction-qualified 46"]
    lines = [
        "# Cognitive model v0.1 parameter-sweep simulation",
        "",
        f"Primary cohort: {qualified_count} participants with at least 70% of testing edits moving toward the target. All 62 participants are included as a sensitivity analysis.",
        "",
        "Each sweep changes one parameter while preserving the participant's winning model family and fitted values for every other parameter. Predictions use the maximum-probability feature subset conditional on the observed number of edited features, matching the earlier v0.1 participant-versus-model analysis.",
        "",
        "## Largest primary-cohort endpoint changes",
        "",
        "The table reports the change from each parameter's lowest to highest grid value. Positive means the simulated outcome increases.",
        "",
        "| XAI | Parameter | Desideratum | n | Mean change [95% CI] |",
        "|---|---|---|---:|---:|",
    ]
    ranked = sorted(primary, key=lambda row: abs(float(row["mean change (high - low)"])), reverse=True)
    for row in ranked[:24]:
        lines.append(
            f"| {row['xai label']} | {row['parameter label']} | {row['metric label']} | "
            f"{row['applicable participants']} | {float(row['mean change (high - low)']):.4f} "
            f"[{float(row['95% bootstrap CI low']):.4f}, {float(row['95% bootstrap CI high']):.4f}] |"
        )
    lines.extend(
        [
            "",
            "Success and actionability are rates. Confidence gain is a probability change. Boundary improvement and edit L1 use summed range-normalized L1 units. Plausibility is nearest-reference Gower similarity on a 0–1 scale.",
            "",
        ]
    )
    (OUTPUT_DIR / "v01_parameter_sweep_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-boundaries",
        action="store_true",
        help="Reuse the saved boundary-distance JSON if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    dataset = experiment["datasets"]["diabetes"]
    result_rows = read_csv(RESULTS)
    fits = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["selected family by CV"] == "1"
    }
    pending, boundary_profiles, qualified = prepare_simulations(
        result_rows, fits, dataset
    )
    boundary_path = OUTPUT_DIR / "v01_parameter_sweep_boundaries.json"
    if args.reuse_boundaries and boundary_path.exists():
        boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    else:
        print(
            f"Solving exact boundaries for {len(boundary_profiles)} proposals...",
            flush=True,
        )
        boundaries = exact_boundaries(
            dataset["browser_model"], dataset["test_pool"][0], boundary_profiles
        )
        boundary_path.write_text(
            json.dumps(boundaries, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    trial_rows = score_trials(pending, boundaries, dataset["browser_model"])
    participant_rows = participant_means(trial_rows)
    summaries = aggregate(participant_rows, "direction-qualified 46")
    summaries.extend(aggregate(participant_rows, "all 62"))
    effects = endpoint_effects(participant_rows, "direction-qualified 46")
    effects.extend(endpoint_effects(participant_rows, "all 62"))

    write_csv(OUTPUT_DIR / "v01_parameter_sweep_trials.csv", trial_rows)
    write_csv(OUTPUT_DIR / "v01_parameter_sweep_participants.csv", participant_rows)
    write_csv(OUTPUT_DIR / "v01_parameter_sweep_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "v01_parameter_sweep_endpoint_effects.csv", effects)
    write_figures(summaries)
    write_endpoint_overview(effects)
    write_report(summaries, effects, len(qualified))
    metadata = {
        "model": "cognitive model v0.1",
        "participants": len(fits),
        "direction-qualified participants": len(qualified),
        "trials per participant": 20,
        "method": "One-at-a-time parameter sweeps around each participant's fitted winning-family model; maximum-probability feature subset conditional on observed K, matching the earlier v0.1 participant-versus-model analysis.",
        "parameter grids": {
            parameter: list(definition["values"])
            for parameter, definition in PARAMETERS.items()
        },
        "desiderata": METRICS,
        "trial simulation rows": len(trial_rows),
        "participant simulation rows": len(participant_rows),
        "boundary proposals": len(boundary_profiles),
        "aggregate rows": len(summaries),
    }
    (OUTPUT_DIR / "v01_parameter_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
