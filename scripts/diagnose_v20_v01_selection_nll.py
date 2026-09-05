"""Diagnose whether v0.1 selection NLL reflects diffuse or wrong rankings."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_models.v0_1.model import (  # noqa: E402
    EPSILON,
    build_memory,
    contribution_distribution,
    exemplar_distribution,
    observed_trials,
    selection_parameter_grid,
)


RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
FITS = ROOT / "qualtrics" / "v20_cognitive_model_v01_fits.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTPUT_DIR = ROOT / "outputs" / "v20-v01-selection-nll-diagnostic"
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


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


def distribution_for_trial(family, condition, parameters, memory, trial):
    if family == "feature contribution":
        return contribution_distribution(
            trial["profile"],
            trial["target"],
            memory,
            parameters,
            trial["observed_k"],
        )
    return exemplar_distribution(
        trial["profile"],
        trial["target"],
        memory,
        condition,
        parameters,
        trial["observed_k"],
    )


def tempered_probabilities(probabilities: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 0.0:
        return np.full(len(probabilities), 1.0 / len(probabilities))
    log_values = gamma * np.log(np.maximum(probabilities, EPSILON))
    values = np.exp(log_values - float(np.max(log_values)))
    return values / values.sum()


def trial_base(family, condition, parameters, memory, trial):
    proposals = distribution_for_trial(
        family, condition, parameters, memory, trial
    )
    subsets = [subset for subset, _, _ in proposals]
    base = np.asarray([probability for _, probability, _ in proposals])
    observed = tuple(np.where(np.abs(trial["observed"]) > 1e-8)[0])
    position = subsets.index(observed) if observed in subsets else None
    return base, position


def trial_diagnostic(base, position, gamma):
    if len(base) == 0:
        return {
            "selection NLL": -math.log(EPSILON),
            "observed subset probability": 0.0,
            "observed subset rank": 1,
            "top-1 hit": 0,
            "top-3 hit": 0,
            "observed subset impossible": 1,
            "maximum subset probability": 0.0,
            "normalized entropy": 0.0,
            "effective subset count": 0.0,
            "possible subset count": 0,
        }
    probabilities = tempered_probabilities(base, gamma)
    if position is not None:
        observed_probability = float(probabilities[position])
        rank = 1 + int(
            np.sum(probabilities > observed_probability + 1e-12)
        )
        impossible = 0
    else:
        observed_probability = 0.0
        rank = len(probabilities) + 1
        impossible = 1
    entropy = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, EPSILON)))
    )
    maximum_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    return {
        "selection NLL": -math.log(max(observed_probability, EPSILON)),
        "observed subset probability": observed_probability,
        "observed subset rank": rank,
        "top-1 hit": int(rank == 1 and not impossible),
        "top-3 hit": int(rank <= 3 and not impossible),
        "observed subset impossible": impossible,
        "maximum subset probability": float(np.max(probabilities)),
        "normalized entropy": entropy / maximum_entropy if maximum_entropy else 0.0,
        "effective subset count": math.exp(entropy),
        "possible subset count": len(probabilities),
    }


def fit_selection(family, condition, training, trials, indices, temperatures):
    scored = []
    for parameters in selection_parameter_grid(condition, family):
        memory = build_memory(training, condition, parameters["eta"])
        bases = [
            trial_base(family, condition, parameters, memory, trials[index])
            for index in indices
        ]
        for gamma in temperatures:
            nll = np.mean(
                [
                    trial_diagnostic(base, position, gamma)["selection NLL"]
                    for base, position in bases
                ]
            )
            scored.append(
                (
                    float(nll),
                    abs(gamma - 1.0),
                    gamma,
                    parameters,
                )
            )
    return min(scored, key=lambda item: (item[0], item[1]))


def fit_temperature(
    family, condition, parameters, training, trials, indices, temperatures
):
    memory = build_memory(training, condition, parameters["eta"])
    bases = [
        trial_base(family, condition, parameters, memory, trials[index])
        for index in indices
    ]
    scored = []
    for gamma in temperatures:
        nll = np.mean(
            [
                trial_diagnostic(base, position, gamma)["selection NLL"]
                for base, position in bases
            ]
        )
        scored.append((float(nll), abs(gamma - 1.0), gamma))
    return min(scored, key=lambda item: (item[0], item[1]))


def diagnose_participant(participant, rows, fit, training, case_map, qualified):
    trials = observed_trials(rows, case_map)
    family, condition = fit["model family"], fit["xai"]
    fold_count = 5
    detail = []
    for fold in range(fold_count):
        train_indices = [
            index for index in range(len(trials)) if index % fold_count != fold
        ]
        test_indices = [
            index for index in range(len(trials)) if index % fold_count == fold
        ]
        _, _, _, current_parameters = fit_selection(
            family, condition, training, trials, train_indices, (1.0,)
        )
        _, _, selected_gamma = fit_temperature(
            family,
            condition,
            current_parameters,
            training,
            trials,
            train_indices,
            GAMMA_GRID,
        )
        tempered_parameters = current_parameters
        current_memory = build_memory(
            training, condition, current_parameters["eta"]
        )
        tempered_memory = build_memory(
            training, condition, tempered_parameters["eta"]
        )
        for index in test_indices:
            current_base = trial_base(
                family,
                condition,
                current_parameters,
                current_memory,
                trials[index],
            )
            current = trial_diagnostic(
                *current_base,
                1.0,
            )
            tempered_base = trial_base(
                family,
                condition,
                tempered_parameters,
                tempered_memory,
                trials[index],
            )
            tempered = trial_diagnostic(
                *tempered_base,
                selected_gamma,
            )
            detail.append(
                {
                    "participant": participant,
                    "direction-qualified cohort (0/1)": int(qualified),
                    "xai": condition,
                    "model family held fixed": family,
                    "fold": fold,
                    "trial index": index,
                    "selected inverse temperature gamma": selected_gamma,
                    **{f"current {name}": value for name, value in current.items()},
                    **{f"temperature-scaled {name}": value for name, value in tempered.items()},
                }
            )
    return detail


def bootstrap_ci(values: np.ndarray, seed: int):
    generator = np.random.default_rng(seed)
    means = np.mean(
        generator.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize(detail, cohort):
    rows = (
        [row for row in detail if row["direction-qualified cohort (0/1)"] == 1]
        if cohort == "direction-qualified 46"
        else detail
    )
    participants = sorted({str(row["participant"]) for row in rows})
    participant_values = []
    for participant in participants:
        values = [row for row in rows if row["participant"] == participant]
        participant_values.append(
            {
                "participant": participant,
                "current NLL": float(
                    np.mean([row["current selection NLL"] for row in values])
                ),
                "temperature-scaled NLL": float(
                    np.mean(
                        [
                            row["temperature-scaled selection NLL"]
                            for row in values
                        ]
                    )
                ),
            }
        )
    current_nll = np.asarray([row["current NLL"] for row in participant_values])
    scaled_nll = np.asarray(
        [row["temperature-scaled NLL"] for row in participant_values]
    )
    improvement = current_nll - scaled_nll
    low, high = bootstrap_ci(improvement, 20260831 + len(participants))
    nonzero = improvement[np.abs(improvement) > 1e-12]
    test = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
    current_probabilities = np.asarray(
        [row["current observed subset probability"] for row in rows]
    )
    possible = np.asarray([row["current possible subset count"] for row in rows])
    hit_rows = [row for row in rows if row["current top-1 hit"] == 1]
    miss_rows = [row for row in rows if row["current top-1 hit"] == 0]
    gammas = np.asarray(
        [row["selected inverse temperature gamma"] for row in rows]
    )
    return {
        "cohort": cohort,
        "participants": len(participants),
        "held-out trials": len(rows),
        "current CV selection NLL": float(current_nll.mean()),
        "current geometric-mean observed-subset probability": float(
            math.exp(-current_nll.mean())
        ),
        "arithmetic-mean observed-subset probability": float(
            current_probabilities.mean()
        ),
        "top-1 exact-subset hit rate": float(
            np.mean([row["current top-1 hit"] for row in rows])
        ),
        "top-3 exact-subset coverage": float(
            np.mean([row["current top-3 hit"] for row in rows])
        ),
        "impossible observed-subset rate": float(
            np.mean([row["current observed subset impossible"] for row in rows])
        ),
        "mean observed-subset rank": float(
            np.mean([row["current observed subset rank"] for row in rows])
        ),
        "median observed-subset rank": float(
            np.median([row["current observed subset rank"] for row in rows])
        ),
        "mean maximum subset probability": float(
            np.mean([row["current maximum subset probability"] for row in rows])
        ),
        "mean normalized entropy (0 deterministic, 1 uniform)": float(
            np.mean([row["current normalized entropy"] for row in rows])
        ),
        "mean effective subset count": float(
            np.mean([row["current effective subset count"] for row in rows])
        ),
        "mean possible subset count": float(possible.mean()),
        "mean NLL when observed subset ranked first": float(
            np.mean([row["current selection NLL"] for row in hit_rows])
        ),
        "mean NLL when observed subset not ranked first": float(
            np.mean([row["current selection NLL"] for row in miss_rows])
        ),
        "temperature-scaled CV NLL": float(scaled_nll.mean()),
        "temperature-scaling mean NLL improvement": float(improvement.mean()),
        "temperature-scaling improvement 95% CI low": low,
        "temperature-scaling improvement 95% CI high": high,
        "temperature-scaling one-sided paired Wilcoxon p": float(test.pvalue),
        "participants improved by temperature scaling": int(
            np.sum(improvement > 1e-12)
        ),
        "participants worsened by temperature scaling": int(
            np.sum(improvement < -1e-12)
        ),
        "median selected inverse temperature gamma": float(np.median(gammas)),
        "mean selected inverse temperature gamma": float(np.mean(gammas)),
        "fold predictions selecting gamma > 1": float(np.mean(gammas > 1.0)),
        "fold predictions selecting gamma = 1": float(np.mean(gammas == 1.0)),
        "fold predictions selecting gamma < 1": float(np.mean(gammas < 1.0)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_rows = read_csv(RESULTS)
    grouped = defaultdict(list)
    directions = defaultdict(list)
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
    fits = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["selected family by CV"] == "1"
    }
    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    detail = []
    for completed, participant in enumerate(sorted(grouped), start=1):
        detail.extend(
            diagnose_participant(
                participant,
                grouped[participant],
                fits[participant],
                dataset["training_pool"],
                case_map,
                participant in qualified,
            )
        )
        print(f"diagnosed {completed}/62", flush=True)
    summaries = [
        summarize(detail, "direction-qualified 46"),
        summarize(detail, "all 62"),
    ]
    write_csv(OUTPUT_DIR / "selection_nll_trial_diagnostics.csv", detail)
    (OUTPUT_DIR / "selection_nll_diagnostic_summary.json").write_text(
        json.dumps(
            {
                "inverse temperature grid": GAMMA_GRID,
                "interpretation": "gamma > 1 sharpens the feature-subset distribution; gamma < 1 flattens it; rankings are unchanged for a fixed parameterization.",
                "method": "Five-fold held-out diagnostic. Current parameters are refit within each fold at gamma=1. With those parameters and feature rankings held fixed, inverse temperature gamma is selected within the training fold and evaluated on its test fold; winning family is held fixed.",
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
