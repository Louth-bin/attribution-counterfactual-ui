"""Theory-only factorial parameter sweeps for cognitive model v0.1.

No participant fits, choices, or edit counts are used. The design crosses the
model's parameter grids within each family and XAI condition, fixes K=2, and
evaluates the 20 v1.6 testing stimuli. Rho uses a symmetric signed grid and a
no-direction-reversal magnitude correction.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    build_memory,
    contribution_components,
    contribution_delta,
    exemplar_delta,
    exemplar_vector,
    label_sign,
    normalize_case,
    subset_distribution,
)


BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTPUT_DIR = ROOT / "outputs" / "v20-v01-theoretical-parameter-sweeps"
SIGNED_RHO_GRID = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
K = 2

PARAMETERS = {
    "eta": {"label": "Explanation reliance (η)", "values": ETA_GRID},
    "alpha": {"label": "Global relevance reliance (α)", "values": ALPHA_GRID},
    "rho": {"label": "Signed change margin (ρ)", "values": SIGNED_RHO_GRID},
    "lambda": {"label": "Exemplar locality (λ)", "values": LAMBDA_GRID},
    "beta": {"label": "Remembered-change reliance (β)", "values": BETA_GRID},
    "age actionable": {"label": "Age actionable", "values": (0, 1)},
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

FAMILY_LABELS = {
    "feature contribution": "Feature-contribution family",
    "weighted examples": "Weighted-example family",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def applicable_parameters(family: str, xai: str) -> tuple[str, ...]:
    if family == "feature contribution":
        values = ["alpha", "rho", "age actionable"]
    else:
        values = ["rho", "lambda", "age actionable"]
        if xai == "counterfactual":
            values.append("beta")
    if xai != "none":
        values.insert(0, "eta")
    return tuple(values)


def parameter_grid(family: str, xai: str):
    eta_values = (0.0,) if xai == "none" else ETA_GRID
    age_values = (0, 1)
    if family == "feature contribution":
        for eta, alpha, rho, age_actionable in itertools.product(
            eta_values, ALPHA_GRID, SIGNED_RHO_GRID, age_values
        ):
            yield {
                "eta": eta,
                "alpha": alpha,
                "rho": rho,
                "lambda": math.nan,
                "beta": math.nan,
                "age actionable": age_actionable,
                "immutable index": 4,
            }
    else:
        beta_values = BETA_GRID if xai == "counterfactual" else (0.0,)
        for eta, locality, beta, rho, age_actionable in itertools.product(
            eta_values,
            LAMBDA_GRID,
            beta_values,
            SIGNED_RHO_GRID,
            age_values,
        ):
            yield {
                "eta": eta,
                "alpha": math.nan,
                "rho": rho,
                "lambda": locality,
                "beta": beta,
                "age actionable": age_actionable,
                "immutable index": 4,
            }


def apply_signed_margin(
    profile: np.ndarray,
    base_delta: np.ndarray,
    direction: np.ndarray,
    subset: tuple[int, ...],
    rho: float,
) -> np.ndarray:
    result = np.zeros_like(base_delta)
    selected = np.asarray(subset, dtype=int)
    signs = np.sign(direction[selected])
    magnitudes = np.maximum(0.0, np.abs(base_delta[selected]) + rho)
    result[selected] = signs * magnitudes
    return np.clip(profile + result, 0.0, 1.0) - profile


def signed_distribution(
    family: str,
    xai: str,
    profile: np.ndarray,
    target: int,
    memory: dict,
    parameters: dict,
) -> list[tuple[tuple[int, ...], float, np.ndarray]]:
    rho = float(parameters["rho"])
    zero_parameters = {**parameters, "rho": 0.0}
    if family == "feature contribution":
        components = contribution_components(profile, target, memory, parameters)
        subsets = subset_distribution(
            components[0],
            K,
            int(parameters["age actionable"]),
            int(parameters["immutable index"]),
        )
        return [
            (
                subset,
                probability,
                apply_signed_margin(
                    profile,
                    contribution_delta(
                        profile,
                        target,
                        memory,
                        zero_parameters,
                        subset,
                        components,
                    ),
                    components[2],
                    subset,
                    rho,
                ),
            )
            for subset, probability in subsets
        ]
    vector = exemplar_vector(profile, target, memory, xai, parameters)
    subsets = subset_distribution(
        np.abs(vector),
        K,
        int(parameters["age actionable"]),
        int(parameters["immutable index"]),
    )
    return [
        (
            subset,
            probability,
            apply_signed_margin(
                profile,
                exemplar_delta(
                    profile,
                    target,
                    memory,
                    xai,
                    zero_parameters,
                    subset,
                    vector,
                ),
                vector,
                subset,
                rho,
            ),
        )
        for subset, probability in subsets
    ]


def prepare(dataset: dict) -> tuple[list[dict], dict[str, np.ndarray], int]:
    training = dataset["training_pool"]
    test_cases = dataset["test_pool"]
    memory_cache = {
        (xai, eta): build_memory(training, xai, eta)
        for xai in XAI_LABELS
        for eta in ((0.0,) if xai == "none" else ETA_GRID)
    }
    profiles: dict[str, np.ndarray] = {}
    pending: list[dict] = []
    configuration_count = 0
    for case_index, case in enumerate(test_cases):
        profiles[f"original|{case_index}"] = normalize_case(case)
    for family in FAMILY_LABELS:
        for xai in XAI_LABELS:
            for configuration_index, parameters in enumerate(parameter_grid(family, xai)):
                configuration_count += 1
                memory = memory_cache[(xai, float(parameters["eta"]))]
                for case_index, case in enumerate(test_cases):
                    original = normalize_case(case)
                    target = -label_sign(case["prediction"]["label"])
                    distribution = signed_distribution(
                        family,
                        xai,
                        original,
                        target,
                        memory,
                        parameters,
                    )
                    subset, subset_probability, delta = max(
                        distribution, key=lambda proposal: proposal[1]
                    )
                    edited = np.clip(original + delta, 0.0, 1.0)
                    key = f"{family}|{xai}|{configuration_index}|{case_index}"
                    profiles[key] = edited
                    pending.append(
                        {
                            "family": family,
                            "xai": xai,
                            "configuration": configuration_index,
                            "parameters": dict(parameters),
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
    return pending, profiles, configuration_count


def score(pending: list[dict], boundaries: dict[str, float], dataset: dict):
    browser_model = dataset["browser_model"]
    reference = reference_matrices("diabetes", dataset)[0]
    rows = []
    for item in pending:
        case = item["case"]
        target_label = case["counterfactual"]["prediction"]["label"]
        original_probability = target_probability(
            item["original"], target_label, case, browser_model
        )
        parameters = item["parameters"]
        rows.append(
            {
                "model family": item["family"],
                "xai": item["xai"],
                "configuration": item["configuration"],
                "case": item["case index"] + 1,
                "instance id": case["instance_id"],
                "K": K,
                "eta": parameters["eta"],
                "alpha": parameters["alpha"],
                "rho": parameters["rho"],
                "lambda": parameters["lambda"],
                "beta": parameters["beta"],
                "age actionable": parameters["age actionable"],
                "MAP subset": "|".join(str(index) for index in item["subset"]),
                "MAP subset probability": item["subset probability"],
                "success": float(
                    prediction_label(item["edited"], case, browser_model)
                    == target_label
                ),
                "target confidence gain": target_probability(
                    item["edited"], target_label, case, browser_model
                )
                - original_probability,
                "boundary improvement": boundaries[f"original|{item['case index']}"]
                - boundaries[item["profile key"]],
                "plausibility": similarity(item["edited"], reference),
                "actionability": float(abs(item["delta"][4]) <= 1e-9),
                "edit L1": float(np.abs(item["delta"]).sum()),
            }
        )
    return rows


def summarize(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values: dict[tuple, list[float]] = defaultdict(list)
    for row in trial_rows:
        family, xai = str(row["model family"]), str(row["xai"])
        for parameter in applicable_parameters(family, xai):
            parameter_value = float(row[parameter])
            for metric in METRICS:
                values[(family, xai, parameter, parameter_value, metric)].append(
                    float(row[metric])
                )
    output = []
    for key, observations in sorted(values.items()):
        family, xai, parameter, parameter_value, metric = key
        array = np.asarray(observations)
        output.append(
            {
                "model family": family,
                "model family label": FAMILY_LABELS[family],
                "xai": xai,
                "xai label": XAI_LABELS[xai],
                "parameter": parameter,
                "parameter label": PARAMETERS[parameter]["label"],
                "parameter value": parameter_value,
                "metric": metric,
                "metric label": METRICS[metric],
                "factorial case-evaluations": len(array),
                "mean": float(array.mean()),
                "10th percentile": float(np.quantile(array, 0.10)),
                "90th percentile": float(np.quantile(array, 0.90)),
            }
        )
    return output


def endpoint_effects(summary_rows: list[dict[str, object]]):
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["model family"], row["xai"], row["parameter"], row["metric"])].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        low = min(rows, key=lambda row: float(row["parameter value"]))
        high = max(rows, key=lambda row: float(row["parameter value"]))
        family, xai, parameter, metric = key
        output.append(
            {
                "model family": family,
                "model family label": FAMILY_LABELS[family],
                "xai": xai,
                "xai label": XAI_LABELS[xai],
                "parameter": parameter,
                "parameter label": PARAMETERS[parameter]["label"],
                "low endpoint": low["parameter value"],
                "high endpoint": high["parameter value"],
                "metric": metric,
                "metric label": METRICS[metric],
                "mean change (high - low)": float(high["mean"]) - float(low["mean"]),
            }
        )
    return output


def make_curve_figure(summary_rows, family: str, parameter: str):
    rows = [
        row
        for row in summary_rows
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
                    for row in rows
                    if row["metric"] == metric and row["xai"] == xai
                ),
                key=lambda row: float(row["parameter value"]),
            )
            if not series:
                continue
            x = np.asarray([float(row["parameter value"]) for row in series])
            y = np.asarray([float(row["mean"]) for row in series])
            axis.plot(
                x,
                y,
                color=colors[xai],
                marker=markers[xai],
                linewidth=2,
                label=XAI_LABELS[xai],
            )
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
    fig.suptitle(
        f"Theory-only sweep: {PARAMETERS[parameter]['label']}\n"
        f"{FAMILY_LABELS[family]} · K={K} · other parameters uniformly marginalized",
        fontsize=16,
        y=0.985,
    )
    return fig


def write_figures(summary_rows):
    for family in FAMILY_LABELS:
        stem = family.replace(" ", "-")
        pairs = [
            parameter
            for parameter in PARAMETERS
            if any(
                row["model family"] == family and row["parameter"] == parameter
                for row in summary_rows
            )
        ]
        with PdfPages(OUTPUT_DIR / f"theoretical_sweeps_{stem}.pdf") as pdf:
            for parameter in pairs:
                fig = make_curve_figure(summary_rows, family, parameter)
                fig.savefig(
                    OUTPUT_DIR / f"theoretical_sweep_{stem}_{parameter.replace(' ', '-')}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def write_overview(effects):
    row_keys = []
    lookup = {
        (row["model family"], row["parameter"], row["xai"], row["metric"]): row
        for row in effects
    }
    for family in FAMILY_LABELS:
        for parameter in PARAMETERS:
            for xai in XAI_LABELS:
                if any((family, parameter, xai, metric) in lookup for metric in METRICS):
                    row_keys.append((family, parameter, xai))
    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(18, 12),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for axis, (metric, metric_label) in zip(axes, METRICS.items()):
        values = np.asarray(
            [
                float(lookup[(family, parameter, xai, metric)]["mean change (high - low)"])
                if (family, parameter, xai, metric) in lookup
                else math.nan
                for family, parameter, xai in row_keys
            ]
        )
        maximum = max(float(np.nanmax(np.abs(values))), 1e-12)
        axis.imshow(
            values[:, None],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
            aspect="auto",
        )
        axis.set_title(metric_label, fontsize=10.5)
        axis.set_xticks([])
        axis.set_yticks(range(len(row_keys)))
        if axis is axes[0]:
            axis.set_yticklabels(
                [
                    f"{FAMILY_LABELS[family].replace(' family', '')} · "
                    f"{PARAMETERS[parameter]['label']} · {XAI_LABELS[xai]}"
                    for family, parameter, xai in row_keys
                ],
                fontsize=8.5,
            )
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
                fontsize=8,
                color=color,
            )
    fig.suptitle(
        "Theory-only change from lowest to highest parameter value\n"
        f"K={K} · 20 test stimuli · other parameters uniformly marginalized · column-specific color scales",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(left=0.37, right=0.99, top=0.91, bottom=0.03)
    fig.savefig(
        OUTPUT_DIR / "theoretical_parameter_sweep_overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-boundaries", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    pending, profiles, configuration_count = prepare(dataset)
    boundary_path = OUTPUT_DIR / "theoretical_boundaries.json"
    if args.reuse_boundaries and boundary_path.exists():
        boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    else:
        print(f"Solving exact boundaries for {len(profiles)} profiles...", flush=True)
        boundaries = exact_boundaries(
            dataset["browser_model"], dataset["test_pool"][0], profiles
        )
        boundary_path.write_text(
            json.dumps(boundaries, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    trial_rows = score(pending, boundaries, dataset)
    summary_rows = summarize(trial_rows)
    effects = endpoint_effects(summary_rows)
    write_csv(OUTPUT_DIR / "theoretical_sweep_trials.csv", trial_rows)
    write_csv(OUTPUT_DIR / "theoretical_sweep_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "theoretical_sweep_endpoint_effects.csv", effects)
    write_figures(summary_rows)
    write_overview(effects)
    metadata = {
        "analysis": "theory-only factorial cognitive-model v0.1 parameter sweep",
        "participant data used": False,
        "participant fitted parameters used": False,
        "XAI conditions": list(XAI_LABELS),
        "model families": list(FAMILY_LABELS),
        "testing stimuli": len(dataset["test_pool"]),
        "K": K,
        "prediction rule": "maximum-probability feature subset",
        "other-parameter treatment": "uniform marginalization over the full factorial grid within family and XAI condition",
        "rho interpretation": "signed additive magnitude correction with truncation at zero, so negative rho shrinks changes without reversing direction",
        "parameter grids": {
            parameter: list(definition["values"])
            for parameter, definition in PARAMETERS.items()
        },
        "factorial configurations": configuration_count,
        "case-level simulations": len(trial_rows),
        "boundary profiles": len(profiles),
    }
    (OUTPUT_DIR / "theoretical_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
