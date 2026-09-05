"""Quick theory-only Monte Carlo simulation for cognitive model v0.1."""

from __future__ import annotations

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
from scripts.simulate_v20_v01_theoretical_parameter_sweeps import (  # noqa: E402
    BUNDLE,
    FAMILY_LABELS,
    K,
    METRICS,
    PARAMETERS,
    XAI_LABELS,
    applicable_parameters,
    parameter_grid,
    signed_distribution,
)
from src.cognitive_models.v0_1.model import (  # noqa: E402
    build_memory,
    label_sign,
    normalize_case,
)


OUTPUT_DIR = ROOT / "outputs" / "v20-v01-theoretical-monte-carlo-quick"
DRAWS_PER_CELL = 50
SEED = 20260831


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare(dataset: dict):
    generator = np.random.default_rng(SEED)
    training, cases = dataset["training_pool"], dataset["test_pool"]
    memories = {
        (xai, eta): build_memory(training, xai, eta)
        for xai in XAI_LABELS
        for eta in ((0.0,) if xai == "none" else PARAMETERS["eta"]["values"])
    }
    profiles = {
        f"original|{index}": normalize_case(case)
        for index, case in enumerate(cases)
    }
    pending = []
    cell_count = 0
    for family in FAMILY_LABELS:
        for xai in XAI_LABELS:
            configurations = list(parameter_grid(family, xai))
            for parameter in applicable_parameters(family, xai):
                for value in PARAMETERS[parameter]["values"]:
                    candidates = [
                        item
                        for item in configurations
                        if math.isclose(float(item[parameter]), float(value))
                    ]
                    cell_count += 1
                    for draw in range(DRAWS_PER_CELL):
                        parameters = dict(candidates[int(generator.integers(len(candidates)))])
                        case_index = int(generator.integers(len(cases)))
                        case = cases[case_index]
                        original = normalize_case(case)
                        target = -label_sign(case["prediction"]["label"])
                        distribution = signed_distribution(
                            family,
                            xai,
                            original,
                            target,
                            memories[(xai, float(parameters["eta"]))],
                            parameters,
                        )
                        probabilities = np.asarray(
                            [proposal[1] for proposal in distribution], dtype=float
                        )
                        proposal = distribution[
                            int(generator.choice(len(distribution), p=probabilities))
                        ]
                        subset, subset_probability, delta = proposal
                        edited = np.clip(original + delta, 0.0, 1.0)
                        key = f"{family}|{xai}|{parameter}|{value}|{draw}"
                        profiles[key] = edited
                        pending.append(
                            {
                                "model family": family,
                                "xai": xai,
                                "parameter": parameter,
                                "parameter value": float(value),
                                "draw": draw,
                                "case index": case_index,
                                "case": case,
                                "original": original,
                                "edited": edited,
                                "delta": delta,
                                "subset": subset,
                                "subset probability": float(subset_probability),
                                "profile key": key,
                            }
                        )
    return pending, profiles, cell_count


def score(pending, boundaries, dataset):
    browser_model = dataset["browser_model"]
    reference = reference_matrices("diabetes", dataset)[0]
    rows = []
    for item in pending:
        case = item["case"]
        target_label = case["counterfactual"]["prediction"]["label"]
        original_probability = target_probability(
            item["original"], target_label, case, browser_model
        )
        rows.append(
            {
                "model family": item["model family"],
                "xai": item["xai"],
                "parameter": item["parameter"],
                "parameter label": PARAMETERS[item["parameter"]]["label"],
                "parameter value": item["parameter value"],
                "draw": item["draw"],
                "instance id": case["instance_id"],
                "sampled subset": "|".join(str(index) for index in item["subset"]),
                "sampled subset probability": item["subset probability"],
                "success": float(
                    prediction_label(item["edited"], case, browser_model)
                    == target_label
                ),
                "target confidence gain": target_probability(
                    item["edited"], target_label, case, browser_model
                )
                - original_probability,
                "boundary improvement": boundaries[
                    f"original|{item['case index']}"
                ]
                - boundaries[item["profile key"]],
                "plausibility": similarity(item["edited"], reference),
                "actionability": float(abs(item["delta"][4]) <= 1e-9),
                "edit L1": float(np.abs(item["delta"]).sum()),
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        for metric in METRICS:
            grouped[
                (
                    row["model family"],
                    row["xai"],
                    row["parameter"],
                    row["parameter value"],
                    metric,
                )
            ].append(float(row[metric]))
    output = []
    generator = np.random.default_rng(SEED + 1)
    for key, values in sorted(grouped.items()):
        array = np.asarray(values)
        bootstrap = np.mean(
            generator.choice(
                array, size=(5_000, len(array)), replace=True
            ),
            axis=1,
        )
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        family, xai, parameter, value, metric = key
        output.append(
            {
                "model family": family,
                "xai": xai,
                "xai label": XAI_LABELS[xai],
                "parameter": parameter,
                "parameter label": PARAMETERS[parameter]["label"],
                "parameter value": value,
                "metric": metric,
                "metric label": METRICS[metric],
                "Monte Carlo draws": len(array),
                "mean": float(array.mean()),
                "standard deviation": float(array.std(ddof=1)),
                "95% bootstrap mean CI low": float(low),
                "95% bootstrap mean CI high": float(high),
                "10th percentile": float(np.quantile(array, 0.10)),
                "90th percentile": float(np.quantile(array, 0.90)),
            }
        )
    return output


def make_figure(rows, family, parameter):
    selected = [
        row
        for row in rows
        if row["model family"] == family and row["parameter"] == parameter
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    fig.subplots_adjust(top=0.80, bottom=0.08, hspace=0.42, wspace=0.30)
    colors = {"none": "#6b7280", "attribution": "#0072b2", "counterfactual": "#d55e00"}
    markers = {"none": "o", "attribution": "s", "counterfactual": "^"}
    for axis, (metric, metric_label) in zip(axes.flat, METRICS.items()):
        for xai in XAI_LABELS:
            series = sorted(
                (
                    row
                    for row in selected
                    if row["metric"] == metric and row["xai"] == xai
                ),
                key=lambda row: float(row["parameter value"]),
            )
            if not series:
                continue
            x = np.asarray([float(row["parameter value"]) for row in series])
            y = np.asarray([float(row["mean"]) for row in series])
            low = np.asarray(
                [float(row["95% bootstrap mean CI low"]) for row in series]
            )
            high = np.asarray(
                [float(row["95% bootstrap mean CI high"]) for row in series]
            )
            axis.plot(
                x,
                y,
                color=colors[xai],
                marker=markers[xai],
                linewidth=2,
                label=XAI_LABELS[xai],
            )
            axis.fill_between(x, low, high, color=colors[xai], alpha=0.16)
        axis.set_title(metric_label)
        axis.set_xlabel(PARAMETERS[parameter]["label"])
        axis.set_ylabel("Monte Carlo mean")
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
    fig.suptitle(
        f"Quick Monte Carlo sweep: {PARAMETERS[parameter]['label']}\n"
        f"{FAMILY_LABELS[family]} · K={K} · {DRAWS_PER_CELL} random draws per point",
        fontsize=16,
        y=0.985,
    )
    return fig


def write_figures(rows):
    for family in FAMILY_LABELS:
        stem = family.replace(" ", "-")
        parameters = [
            parameter
            for parameter in PARAMETERS
            if any(
                row["model family"] == family and row["parameter"] == parameter
                for row in rows
            )
        ]
        with PdfPages(OUTPUT_DIR / f"quick_mc_{stem}.pdf") as pdf:
            for parameter in parameters:
                fig = make_figure(rows, family, parameter)
                fig.savefig(
                    OUTPUT_DIR / f"quick_mc_{stem}_{parameter.replace(' ', '-')}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    pending, profiles, cells = prepare(dataset)
    print(f"Solving exact boundaries for {len(profiles)} sampled profiles...", flush=True)
    boundaries = exact_boundaries(
        dataset["browser_model"], dataset["test_pool"][0], profiles
    )
    rows = score(pending, boundaries, dataset)
    summary = summarize(rows)
    write_csv(OUTPUT_DIR / "quick_mc_draws.csv", rows)
    write_csv(OUTPUT_DIR / "quick_mc_summary.csv", summary)
    write_figures(summary)
    metadata = {
        "analysis": "quick theory-only Monte Carlo parameter sweep",
        "participant data used": False,
        "seed": SEED,
        "draws per parameter/XAI/family point": DRAWS_PER_CELL,
        "parameter cells": cells,
        "total random draws": len(rows),
        "K": K,
        "random variables": [
            "other parameter values sampled uniformly from their grids",
            "test stimulus sampled uniformly from 20 cases",
            "feature subset sampled from P(S|x,K)",
        ],
        "intervals": "95% nonparametric bootstrap intervals for the Monte Carlo mean",
    }
    (OUTPUT_DIR / "quick_mc_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
