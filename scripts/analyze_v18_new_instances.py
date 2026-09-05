"""Participant-level exploratory analysis of the v1.8 new-instance cohort only."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "qualtrics" / "v1.8_new_processed.csv"
OUTPUT = ROOT / "qualtrics" / "v1.8_new_instances_analysis.json"
REPORT = ROOT / "qualtrics" / "v1.8_new_instances_report.md"
ORDER = ("none", "attribution", "counterfactual")
METRICS = {
    "success": ("successful counterfactual (0/1)", "higher"),
    "move toward target": ("move towards target (0/1)", "higher"),
    "sparsity": ("sparsity", "higher"),
    "proximity (distance)": ("proximity", "lower"),
    "plausibility": ("plausibility", "higher"),
    "actionability": ("actionability (0/1)", "higher"),
    "boundary distance": ("boundary distance new", "lower"),
    "confidence improvement": ("delta confidence of target label", "higher"),
    "response time": ("response time (seconds)", "lower"),
}


def permutation_contrast(
    first: np.ndarray, second: np.ndarray, better: str
) -> dict[str, float | int | list[float]]:
    values = np.concatenate((first, second))
    n_first = len(first)
    observed = float(first.mean() - second.mean())
    oriented = observed if better == "higher" else -observed
    permutations = []
    for indices in itertools.combinations(range(len(values)), n_first):
        selected = set(indices)
        first_values = np.asarray([values[index] for index in indices])
        second_values = np.asarray(
            [value for index, value in enumerate(values) if index not in selected]
        )
        difference = float(first_values.mean() - second_values.mean())
        permutations.append(difference if better == "higher" else -difference)
    tolerance = 1e-12
    one_sided = sum(value >= oriented - tolerance for value in permutations) / len(permutations)
    two_sided = sum(abs(value) >= abs(oriented) - tolerance for value in permutations) / len(permutations)

    variance = first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second)
    if variance > 0:
        se = math.sqrt(variance)
        df = variance**2 / (
            (first.var(ddof=1) / len(first)) ** 2 / (len(first) - 1)
            + (second.var(ddof=1) / len(second)) ** 2 / (len(second) - 1)
        )
        critical = float(stats.t.ppf(0.975, df))
        interval = [observed - critical * se, observed + critical * se]
    else:
        interval = [observed, observed]
    return {
        "difference": observed,
        "benefit-oriented difference": oriented,
        "95% CI low": float(interval[0]),
        "95% CI high": float(interval[1]),
        "exact one-sided p in beneficial direction": float(one_sided),
        "exact two-sided p": float(two_sided),
        "permutations": len(permutations),
    }


def exact_omnibus(groups: dict[str, np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate([groups[condition] for condition in ORDER])
    sizes = {condition: len(groups[condition]) for condition in ORDER}
    grand = float(values.mean())

    def statistic(assignment: dict[str, np.ndarray]) -> float:
        return float(
            sum(
                len(items) * (float(items.mean()) - grand) ** 2
                for items in assignment.values()
            )
        )

    observed = statistic(groups)
    indices = set(range(len(values)))
    statistics = []
    for none_indices in itertools.combinations(indices, sizes["none"]):
        remaining = indices - set(none_indices)
        for attribution_indices in itertools.combinations(remaining, sizes["attribution"]):
            counterfactual_indices = remaining - set(attribution_indices)
            assignment = {
                "none": values[list(none_indices)],
                "attribution": values[list(attribution_indices)],
                "counterfactual": values[list(counterfactual_indices)],
            }
            statistics.append(statistic(assignment))
    p_value = sum(value >= observed - 1e-12 for value in statistics) / len(statistics)
    return {"between-group statistic": observed, "exact p": float(p_value), "permutations": len(statistics)}


def main() -> None:
    frame = pd.read_csv(DATA, low_memory=False)
    testing = frame.loc[frame["phase"].eq("testing")].copy()
    participant_means = (
        testing.groupby(["participant", "xai"], as_index=False)[
            [column for column, _ in METRICS.values()]
        ]
        .mean()
    )
    if len(participant_means) != 8:
        raise RuntimeError(f"Expected eight new-instance participants, found {len(participant_means)}")

    results = {}
    for label, (column, better) in METRICS.items():
        groups = {
            condition: participant_means.loc[
                participant_means["xai"].eq(condition), column
            ].to_numpy(dtype=float)
            for condition in ORDER
        }
        explained = np.concatenate((groups["attribution"], groups["counterfactual"]))
        results[label] = {
            "column": column,
            "better": better,
            "condition summaries": {
                condition: {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "n": len(values),
                }
                for condition, values in groups.items()
            },
            "contrasts": {
                "attribution - none": permutation_contrast(groups["attribution"], groups["none"], better),
                "counterfactual - none": permutation_contrast(groups["counterfactual"], groups["none"], better),
                "any explanation - none": permutation_contrast(explained, groups["none"], better),
            },
            "three-condition exact omnibus": exact_omnibus(groups),
        }

    omnibus_p = [results[label]["three-condition exact omnibus"]["exact p"] for label in METRICS]
    adjusted = multipletests(omnibus_p, method="holm")[1]
    for label, p_value in zip(METRICS, adjusted):
        results[label]["Holm p across nine omnibus outcomes"] = float(p_value)

    output = {
        "analysis status": "exploratory; participant is the independent unit",
        "cohort": "responses collected on v1.6 new instances only",
        "testing rows": len(testing),
        "participants": len(participant_means),
        "participants by condition": {
            condition: int(participant_means["xai"].eq(condition).sum()) for condition in ORDER
        },
        "participant means": participant_means.to_dict(orient="records"),
        "metrics": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    success = results["success"]
    report = [
        "# v1.8 new-instance cohort analysis",
        "",
        "This analysis includes only the eight participants who answered the v1.6 new instances. The participant is the independent unit.",
        "",
        "## Success outcome",
        "",
        "| Condition | n | Mean success |",
        "|---|---:|---:|",
    ]
    for condition in ORDER:
        summary = success["condition summaries"][condition]
        report.append(f"| {condition} | {summary['n']} | {summary['mean']:.1%} |")
    report.extend(
        [
            "",
            "| Contrast | Difference | 95% CI | Exact one-sided p | Exact two-sided p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, contrast in success["contrasts"].items():
        report.append(
            f"| {name} | {contrast['difference']:.1%} | "
            f"[{contrast['95% CI low']:.1%}, {contrast['95% CI high']:.1%}] | "
            f"{contrast['exact one-sided p in beneficial direction']:.3f} | "
            f"{contrast['exact two-sided p']:.3f} |"
        )
    report.extend(
        [
            "",
            "## All current outcomes",
            "",
            "| Outcome | Better | None | Attribution | Counterfactual | Omnibus exact p | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in results.items():
        summaries = result["condition summaries"]
        report.append(
            f"| {label} | {result['better']} | {summaries['none']['mean']:.3f} | "
            f"{summaries['attribution']['mean']:.3f} | {summaries['counterfactual']['mean']:.3f} | "
            f"{result['three-condition exact omnibus']['exact p']:.3f} | "
            f"{result['Holm p across nine omnibus outcomes']:.3f} |"
        )
    report.extend(
        [
            "",
            "Exact p-values enumerate participant-level label assignments. With only 2/2/4 participants, attainable p-values are coarse and confidence intervals are very wide.",
        ]
    )
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"analysis": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
