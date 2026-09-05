"""Create a transparent participant-exclusion sensitivity analysis for v1.9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_INPUT = ROOT / "qualtrics" / "v1.9_latest_model_simulation_participants.csv"
TRIAL_INPUT = ROOT / "qualtrics" / "v1.9_latest_model_simulation_trials.csv"
FIT_INPUT = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_fits.csv"
DEFAULT_EXCLUSION = "R_7L78tZJ3cQDgpFH"
DEFAULT_OUTPUT = ROOT / "outputs" / "v19-additive-rho-exclusion-sensitivity"

METRICS = (
    "success",
    "target confidence gain",
    "boundary improvement",
    "plausibility",
    "actionability",
    "edit L1",
)


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson r": float(pearson.statistic),
        "pearson p": float(pearson.pvalue),
        "spearman rho": float(spearman.statistic),
        "spearman p": float(spearman.pvalue),
    }


def participant_performance(data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result = {}
    for metric in METRICS:
        observed = data[f"observed {metric}"]
        simulated = data[f"simulated {metric}"]
        summary = correlation(observed, simulated)
        summary.update(
            {
                "observed mean": float(observed.mean()),
                "simulated mean": float(simulated.mean()),
                "MAE": float((observed - simulated).abs().mean()),
            }
        )
        result[metric] = summary
    return result


def boundary_diagnostics(trials: pd.DataFrame, participants: pd.DataFrame) -> dict[str, object]:
    observed = "observed boundary improvement"
    simulated = "simulated boundary improvement"
    case_means = trials.groupby("case")[[observed, simulated]].mean()
    participant_centered_observed = trials[observed] - trials.groupby("participant")[observed].transform("mean")
    participant_centered_simulated = trials[simulated] - trials.groupby("participant")[simulated].transform("mean")
    case_centered_observed = trials[observed] - trials.groupby("case")[observed].transform("mean")
    case_centered_simulated = trials[simulated] - trials.groupby("case")[simulated].transform("mean")
    boundary_gap = participants[simulated] - participants[observed]
    edit_gap = participants["simulated edit L1"] - participants["observed edit L1"]
    return {
        "participant observed versus simulated": correlation(participants[observed], participants[simulated]),
        "trial observed versus simulated": correlation(trials[observed], trials[simulated]),
        "case-average observed versus simulated": correlation(case_means[observed], case_means[simulated]),
        "within-participant centered observed versus simulated": correlation(
            participant_centered_observed, participant_centered_simulated
        ),
        "within-case centered observed versus simulated": correlation(
            case_centered_observed, case_centered_simulated
        ),
        "participant observed boundary improvement versus observed edit L1": correlation(
            participants[observed], participants["observed edit L1"]
        ),
        "participant model-minus-observed boundary error versus edit-size error": correlation(
            boundary_gap, edit_gap
        ),
        "observed participant boundary SD": float(participants[observed].std(ddof=1)),
        "mean within-participant trial boundary SD": float(
            trials.groupby("participant")[observed].std(ddof=1).mean()
        ),
    }


def fit_summary(fits: pd.DataFrame) -> dict[str, object]:
    winners = fits.loc[fits["selected family by CV"].astype(str) == "1"].copy()
    columns = {
        "5-fold CV selection_nll": "mean selection NLL",
        "5-fold CV selection_f1": "mean selection F1",
        "5-fold CV amount_mae": "mean conditional amount MAE",
        "5-fold CV direction_accuracy": "mean direction accuracy",
        "5-fold CV loss": "mean combined loss",
    }

    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        values: dict[str, object] = {
            "participants": int(frame["participant"].nunique()),
            "selected family counts": frame["model family"].value_counts().sort_index().to_dict(),
        }
        for source, label in columns.items():
            values[label] = float(pd.to_numeric(frame[source]).mean())
        return values

    return {
        "overall": summarize(winners),
        "by condition": {
            condition: summarize(group) for condition, group in winners.groupby("xai", sort=True)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-participant", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = args.exclude_participant or [DEFAULT_EXCLUSION]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_participants = pd.read_csv(PARTICIPANT_INPUT)
    all_trials = pd.read_csv(TRIAL_INPUT)
    all_fits = pd.read_csv(FIT_INPUT)
    missing = sorted(set(excluded) - set(all_participants["participant"]))
    if missing:
        raise ValueError(f"Unknown participant(s): {missing}")

    retained_participants = all_participants.loc[
        ~all_participants["participant"].isin(excluded)
    ].copy()
    retained_trials = all_trials.loc[~all_trials["participant"].isin(excluded)].copy()
    retained_fits = all_fits.loc[~all_fits["participant"].isin(excluded)].copy()
    excluded_rows = all_participants.loc[all_participants["participant"].isin(excluded)].copy()

    result = {
        "analysis": "participant-exclusion sensitivity analysis",
        "exclusion rule": "post hoc behavioral outlier; raw records retained",
        "excluded participants": excluded,
        "excluded participant records": excluded_rows.to_dict(orient="records"),
        "full sample participants": int(all_participants["participant"].nunique()),
        "retained participants": int(retained_participants["participant"].nunique()),
        "retained trials": int(len(retained_trials)),
        "participant performance correlations": {
            "full sample": participant_performance(all_participants),
            "excluding flagged participant": participant_performance(retained_participants),
        },
        "boundary diagnostics": {
            "full sample": boundary_diagnostics(all_trials, all_participants),
            "excluding flagged participant": boundary_diagnostics(retained_trials, retained_participants),
        },
        "cognitive-model fit summary excluding flagged participant": fit_summary(retained_fits),
    }

    retained_participants.to_csv(output_dir / "retained_participant_metrics.csv", index=False)
    retained_trials.to_csv(output_dir / "retained_trial_metrics.csv", index=False)
    retained_fits.to_csv(output_dir / "retained_model_fits.csv", index=False)
    excluded_rows.to_csv(output_dir / "excluded_participant_record.csv", index=False)
    performance_rows = []
    for sample, metrics in result["participant performance correlations"].items():
        for metric, values in metrics.items():
            performance_rows.append({"sample": sample, "metric": metric, **values})
    pd.DataFrame(performance_rows).to_csv(
        output_dir / "performance_metric_comparison.csv", index=False
    )
    boundary_rows = []
    for sample, diagnostics in result["boundary diagnostics"].items():
        for diagnostic, values in diagnostics.items():
            if isinstance(values, dict):
                boundary_rows.append({"sample": sample, "diagnostic": diagnostic, **values})
            else:
                boundary_rows.append(
                    {"sample": sample, "diagnostic": diagnostic, "value": values}
                )
    pd.DataFrame(boundary_rows).to_csv(
        output_dir / "boundary_diagnostic_comparison.csv", index=False
    )
    (output_dir / "exclusion_sensitivity_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
