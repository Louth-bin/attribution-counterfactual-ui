"""Compare local-only (alpha=0) and global-only (alpha=1) v0.1 models."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from ablate_v20_v01_parameters import (
    BUNDLE,
    FITS,
    RESULTS,
    cross_validated_ablation,
    read_csv,
    write_csv,
)
from src.cognitive_models.v0_1.model import observed_trials


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "v20-v01-parameter-ablation"


def compare_participant(job: tuple) -> dict[str, object]:
    participant, rows, fit, training, case_map, qualified = job
    trials = observed_trials(rows, case_map)
    metrics = {}
    for alpha in (0.0, 1.0):
        cv, _ = cross_validated_ablation(
            "feature contribution",
            fit["xai"],
            training,
            trials,
            "alpha",
            fixed_value=alpha,
        )
        metrics[alpha] = cv
    delta_nll = metrics[1.0]["selection_nll"] - metrics[0.0]["selection_nll"]
    return {
        "participant": participant,
        "direction-qualified cohort (0/1)": int(qualified),
        "xai": fit["xai"],
        "alpha=0 CV selection NLL": metrics[0.0]["selection_nll"],
        "alpha=1 CV selection NLL": metrics[1.0]["selection_nll"],
        "delta NLL (alpha=1 - alpha=0; positive favors alpha=0)": delta_nll,
        "alpha=0 CV selection F1": metrics[0.0]["selection_f1"],
        "alpha=1 CV selection F1": metrics[1.0]["selection_f1"],
        "better endpoint": (
            "alpha=0"
            if delta_nll > 1e-12
            else "alpha=1"
            if delta_nll < -1e-12
            else "tie"
        ),
    }


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize(rows: list[dict[str, object]], cohort: str) -> dict[str, object]:
    selected = (
        [row for row in rows if row["direction-qualified cohort (0/1)"] == 1]
        if cohort == "direction-qualified 46"
        else rows
    )
    deltas = np.asarray(
        [
            float(
                row[
                    "delta NLL (alpha=1 - alpha=0; positive favors alpha=0)"
                ]
            )
            for row in selected
        ]
    )
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if len(nonzero):
        test = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    else:
        statistic, p_value = 0.0, 1.0
    low, high = bootstrap_ci(deltas, 20260831 + len(selected))
    return {
        "cohort": cohort,
        "applicable feature-contribution participants": len(selected),
        "mean alpha=0 CV selection NLL": float(
            np.mean([float(row["alpha=0 CV selection NLL"]) for row in selected])
        ),
        "mean alpha=1 CV selection NLL": float(
            np.mean([float(row["alpha=1 CV selection NLL"]) for row in selected])
        ),
        "mean delta NLL (alpha=1 - alpha=0; positive favors alpha=0)": float(
            np.mean(deltas)
        ),
        "mean delta 95% bootstrap CI low": low,
        "mean delta 95% bootstrap CI high": high,
        "participants favoring alpha=0": sum(
            row["better endpoint"] == "alpha=0" for row in selected
        ),
        "participants tied": sum(row["better endpoint"] == "tie" for row in selected),
        "participants favoring alpha=1": sum(
            row["better endpoint"] == "alpha=1" for row in selected
        ),
        "one-sided paired Wilcoxon statistic": statistic,
        "one-sided paired Wilcoxon p (alpha=0 better)": p_value,
    }


def main() -> None:
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
        and row["model family"] == "feature contribution"
    }
    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training = dataset["training_pool"]
    jobs = [
        (
            participant,
            grouped[participant],
            fit,
            training,
            case_map,
            participant in qualified,
        )
        for participant, fit in sorted(fits.items())
    ]

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as executor:
        futures = {executor.submit(compare_participant, job): job[0] for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"compared alpha endpoints ({completed}/{len(jobs)})", flush=True)
    rows.sort(key=lambda row: str(row["participant"]))
    summaries = [
        summarize(rows, "direction-qualified 46"),
        summarize(rows, "all 62"),
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "v01_alpha_endpoint_participants.csv", rows)
    write_csv(OUTPUT_DIR / "v01_alpha_endpoint_summary.csv", summaries)
    (OUTPUT_DIR / "v01_alpha_endpoint_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
