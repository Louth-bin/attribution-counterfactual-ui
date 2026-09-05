"""Summarize v0.1 fit metrics against interpretable baselines."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
FITS = ROOT / "qualtrics" / "v20_cognitive_model_v01_fits.csv"
OUTPUT = ROOT / "outputs" / "v20-v01-fit-quality-summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    means = np.mean(
        generator.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "25th percentile": float(np.quantile(values, 0.25)),
        "75th percentile": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def paired_comparison(
    model: np.ndarray, baseline: np.ndarray, seed: int
) -> dict[str, object]:
    improvement = baseline - model
    low, high = bootstrap_ci(improvement, seed)
    nonzero = improvement[np.abs(improvement) > 1e-12]
    test = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
    return {
        "mean improvement (baseline - model; positive favors model)": float(
            improvement.mean()
        ),
        "mean improvement 95% bootstrap CI low": low,
        "mean improvement 95% bootstrap CI high": high,
        "participants model better": int(np.sum(improvement > 1e-12)),
        "participants tied": int(np.sum(np.abs(improvement) <= 1e-12)),
        "participants model worse": int(np.sum(improvement < -1e-12)),
        "one-sided paired Wilcoxon p": float(test.pvalue),
    }


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(RESULTS):
        grouped[row["participant"]].append(row)
    fits = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["selected family by CV"] == "1"
    }
    participant_rows = []
    for participant, rows in sorted(grouped.items()):
        testing = [row for row in rows if row["phase"] == "testing"]
        direction = [
            float(row["move towards target (0/1)"])
            for row in testing
            if row["move towards target (0/1)"] != ""
        ]
        chance_nll = []
        zero_amount_mae = []
        for row in testing:
            changes = np.asarray(
                [float(row[f"x_{index}_change"]) for index in range(1, 6)]
            )
            selected = np.abs(changes) > 1e-8
            k = int(np.sum(selected))
            chance_nll.append(math.log(math.comb(5, k)))
            zero_amount_mae.append(
                float(np.mean(np.abs(changes[selected]))) if k else 0.0
            )
        fit = fits[participant]
        participant_rows.append(
            {
                "participant": participant,
                "qualified": len(direction) == 20 and float(np.mean(direction)) >= 0.70,
                "model family": fit["model family"],
                "xai": fit["xai"],
                "in-sample selection NLL": float(fit["in-sample selection_nll"]),
                "CV selection NLL": float(fit["5-fold CV selection_nll"]),
                "uniform-random selection NLL": float(np.mean(chance_nll)),
                "in-sample amount MAE": float(fit["in-sample amount_mae"]),
                "CV amount MAE": float(fit["5-fold CV amount_mae"]),
                "zero-change amount MAE": float(np.mean(zero_amount_mae)),
            }
        )

    summaries = []
    for cohort in ("direction-qualified 46", "all 62"):
        rows = (
            [row for row in participant_rows if row["qualified"]]
            if cohort == "direction-qualified 46"
            else participant_rows
        )
        arrays = {
            name: np.asarray([float(row[name]) for row in rows])
            for name in (
                "in-sample selection NLL",
                "CV selection NLL",
                "uniform-random selection NLL",
                "in-sample amount MAE",
                "CV amount MAE",
                "zero-change amount MAE",
            )
        }
        summaries.append(
            {
                "cohort": cohort,
                "participants": len(rows),
                "metrics": {name: describe(values) for name, values in arrays.items()},
                "selection comparison": paired_comparison(
                    arrays["CV selection NLL"],
                    arrays["uniform-random selection NLL"],
                    20260831 + len(rows),
                ),
                "amount comparison": paired_comparison(
                    arrays["CV amount MAE"],
                    arrays["zero-change amount MAE"],
                    20260901 + len(rows),
                ),
                "mean exact-subset probability implied by mean CV NLL": float(
                    math.exp(-arrays["CV selection NLL"].mean())
                ),
                "mean exact-subset probability implied by mean chance NLL": float(
                    math.exp(-arrays["uniform-random selection NLL"].mean())
                ),
            }
        )
    output = {
        "selection baseline": "Uniform random selection among all subsets of the participant's observed size K from five features; per-trial NLL = log(C(5,K)).",
        "amount baseline": "Predict zero change on every feature the participant actually selected; per-trial MAE is the participant's mean absolute normalized change on those features.",
        "summaries": summaries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
