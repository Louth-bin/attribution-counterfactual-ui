"""Exploratory interim condition analysis for cumulative diabetes responses.

The participant is the independent unit.  Row-level adjusted estimates include
case fixed effects and participant-clustered standard errors.  The analysis is
descriptive and does not implement a sequential-testing stopping boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "v1.4 (n=30)": ROOT / "qualtrics" / "qualtrics_results_v1.4.csv",
    "v1.5 (n=44)": ROOT / "qualtrics" / "qualtrics_results_v1.5.csv",
    "v1.7 (n=50)": ROOT / "qualtrics" / "qualtrics_results_v1.7.csv",
}
OUTPUT = ROOT / "qualtrics" / "v1.7_interim_condition_analysis.json"
REPORT = ROOT / "qualtrics" / "v1.7_interim_condition_report.md"
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
    "response time (log)": ("analysis log response time", "lower"),
}


def q(name: str) -> str:
    return f'Q("{name}")'


def scalar(value: object) -> float | None:
    result = float(np.asarray(value).item())
    return result if math.isfinite(result) else None


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["phase"].str.casefold().eq("testing")].copy()
    frame["analysis log response time"] = np.log1p(
        pd.to_numeric(frame["response time (seconds)"], errors="coerce")
    )
    frame["xai"] = pd.Categorical(frame["xai"], categories=ORDER, ordered=True)
    return frame


def hedges_g(first: np.ndarray, second: np.ndarray) -> float | None:
    n1, n2 = len(first), len(second)
    pooled_df = n1 + n2 - 2
    pooled = math.sqrt(
        ((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)) / pooled_df
    )
    if pooled == 0:
        return None
    correction = 1 - 3 / (4 * pooled_df - 1)
    return float(correction * (first.mean() - second.mean()) / pooled)


def welch_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    difference = float(first.mean() - second.mean())
    variance = first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second)
    se = math.sqrt(variance)
    numerator = variance**2
    denominator = (
        (first.var(ddof=1) / len(first)) ** 2 / (len(first) - 1)
        + (second.var(ddof=1) / len(second)) ** 2 / (len(second) - 1)
    )
    df = numerator / denominator
    critical = stats.t.ppf(0.975, df)
    test = stats.ttest_ind(first, second, equal_var=False)
    return {
        "difference": difference,
        "95% CI low": float(difference - critical * se),
        "95% CI high": float(difference + critical * se),
        "p": float(test.pvalue),
        "Hedges g": hedges_g(first, second),
    }


def analyze_metric(frame: pd.DataFrame, column: str, better: str) -> dict[str, object]:
    use = frame[["participant", "xai", "instance id", column]].dropna().copy()
    participant_means = (
        use.groupby(["participant", "xai"], observed=True)[column].mean().reset_index()
    )
    groups = {
        condition: participant_means.loc[
            participant_means["xai"] == condition, column
        ].to_numpy(dtype=float)
        for condition in ORDER
    }
    summaries = {
        condition: {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "n": len(values),
        }
        for condition, values in groups.items()
    }

    formula = (
        f"{q(column)} ~ C(xai, Treatment(reference='none')) + C({q('instance id')})"
    )
    adjusted = smf.ols(formula, data=use).fit(
        cov_type="cluster", cov_kwds={"groups": use["participant"]}
    )
    names = list(adjusted.params.index)
    condition_names = [
        "C(xai, Treatment(reference='none'))[T.attribution]",
        "C(xai, Treatment(reference='none'))[T.counterfactual]",
    ]
    restriction = np.zeros((2, len(names)))
    for row, name in enumerate(condition_names):
        restriction[row, names.index(name)] = 1
    omnibus_p = scalar(adjusted.wald_test(restriction, scalar=True).pvalue)

    adjusted_contrasts = {}
    for condition, name in zip(("attribution", "counterfactual"), condition_names):
        index = names.index(name)
        estimate = float(adjusted.params.iloc[index])
        interval = np.asarray(adjusted.conf_int().iloc[index], dtype=float)
        adjusted_contrasts[f"{condition} - none"] = {
            "difference": estimate,
            "95% CI low": float(interval[0]),
            "95% CI high": float(interval[1]),
            "p": float(adjusted.pvalues.iloc[index]),
        }

    return {
        "better": better,
        "participant means": summaries,
        "participant-mean Welch contrasts": {
            "attribution - none": welch_difference(groups["attribution"], groups["none"]),
            "counterfactual - none": welch_difference(groups["counterfactual"], groups["none"]),
        },
        "case-adjusted participant-clustered omnibus p": omnibus_p,
        "case-adjusted participant-clustered contrasts": adjusted_contrasts,
    }


def main() -> None:
    output: dict[str, object] = {
        "analysis status": "exploratory interim; no sequential stopping boundary",
        "independent unit": "participant",
        "datasets": {},
    }
    for wave, path in DATASETS.items():
        frame = load(path)
        metrics = {
            label: analyze_metric(frame, column, better)
            for label, (column, better) in METRICS.items()
        }
        omnibus = [
            metrics[label]["case-adjusted participant-clustered omnibus p"]
            for label in METRICS
        ]
        adjusted_p = multipletests(omnibus, method="holm")[1]
        for label, p_value in zip(METRICS, adjusted_p):
            metrics[label]["Holm p across current metric family"] = float(p_value)
        output["datasets"][wave] = {
            "testing rows": len(frame),
            "participants": int(frame["participant"].nunique()),
            "participants by condition": {
                condition: int(frame.loc[frame["xai"] == condition, "participant"].nunique())
                for condition in ORDER
            },
            "metrics": metrics,
        }

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    current = output["datasets"]["v1.7 (n=50)"]
    trajectory_rows = []
    for wave in DATASETS:
        result = output["datasets"][wave]["metrics"]["success"]
        contrast = result["case-adjusted participant-clustered contrasts"][
            "counterfactual - none"
        ]
        trajectory_rows.append(
            f"| {wave} | {contrast['difference']:.3f} | "
            f"[{contrast['95% CI low']:.3f}, {contrast['95% CI high']:.3f}] | "
            f"{contrast['p']:.3f} |"
        )
    report = [
        "# v1.7 exploratory interim condition report",
        "",
        "This is a descriptive interim look. It does not use a prespecified sequential-testing boundary, so repeated peeking must not be treated as a confirmatory significance test.",
        "",
        f"Current sample: {current['participants']} participants "
        f"({current['participants by condition']['none']} none, "
        f"{current['participants by condition']['attribution']} attribution, "
        f"{current['participants by condition']['counterfactual']} counterfactual).",
        "",
        "## Counterfactual-versus-none success trajectory",
        "",
        "| Wave | Case-adjusted difference | 95% CI | p |",
        "|---|---:|---:|---:|",
        *trajectory_rows,
        "",
        "## Current condition tests",
        "",
        "| Outcome | Better | Omnibus p | Holm p | Counterfactual - none | Contrast p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, result in current["metrics"].items():
        contrast = result["case-adjusted participant-clustered contrasts"][
            "counterfactual - none"
        ]
        report.append(
            f"| {label} | {result['better']} | "
            f"{result['case-adjusted participant-clustered omnibus p']:.3f} | "
            f"{result['Holm p across current metric family']:.3f} | "
            f"{contrast['difference']:.3f} | {contrast['p']:.3f} |"
        )
    report.extend(
        [
            "",
            "Estimates use case fixed effects and participant-clustered standard errors. Holm p-values adjust the nine omnibus outcome tests in this exploratory family.",
        ]
    )
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"analysis": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
