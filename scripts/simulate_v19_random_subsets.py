"""Exactly marginalize the v1.9 probabilistic model over attribute subsets."""

from __future__ import annotations

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

from scripts.evaluate_v19_latest_model_simulations import (  # noqa: E402
    BUNDLE,
    FITS,
    METRICS,
    RESULTS,
    exact_boundaries,
    prediction_label,
    read_csv,
    target_probability,
    write_csv,
)
from scripts.fit_parsimonious_weighted_models import (  # noqa: E402
    case_metrics,
    observed_trials,
    training_representation,
)
from scripts.fit_probabilistic_attribute_selection import (  # noqa: E402
    contribution_distribution,
    memory_distribution,
)
from scripts.run_model_only_strategy_study import reference_matrices, similarity  # noqa: E402


EXCLUDED: set[str] = set()
OUTPUT_DIR = ROOT / "outputs" / "v19-probability-weighted-36"
TRIAL_OUTPUT = OUTPUT_DIR / "probability_weighted_trial_metrics.csv"
PARTICIPANT_OUTPUT = OUTPUT_DIR / "probability_weighted_participant_metrics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "probability_weighted_simulation_summary.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    dataset = experiment["datasets"]["diabetes"]
    browser_model = dataset["browser_model"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training = dataset["training_pool"]
    reference = reference_matrices("diabetes", dataset)[0]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(RESULTS):
        if row["participant"] not in EXCLUDED:
            grouped[row["participant"]].append(row)
    fit_map = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["selected family by CV"] == "1" and row["participant"] not in EXCLUDED
    }
    if len(grouped) != 36 or len(fit_map) != 36:
        raise RuntimeError(
            f"Expected 36 participants and fits, found {len(grouped)} and {len(fit_map)}"
        )

    pending = []
    profiles: dict[str, np.ndarray] = {}
    for participant, participant_rows in sorted(grouped.items()):
        fit = fit_map[participant]
        condition = fit["xai"]
        params = {
            name: float(fit[name]) if fit[name] not in ("", "nan") else math.nan
            for name in ("eta", "alpha", "rho", "lambda", "beta")
        }
        params["age actionable"] = int(float(fit["age actionable"]))
        params["immutable index"] = int(float(fit["immutable index"]))
        representation = training_representation(training, condition, params["eta"])
        trials = observed_trials(participant_rows, case_map)
        source_testing = sorted(
            (row for row in participant_rows if row["phase"] == "testing"),
            key=lambda row: int(float(row["case"])),
        )
        for trial, source in zip(trials, source_testing):
            case_id = int(float(source["instance id"]))
            case = case_map[case_id]
            distribution = (
                contribution_distribution(
                    trial["x"], trial["target"], representation, params, trial["observed_k"]
                )
                if fit["model family"] == "feature contribution"
                else memory_distribution(
                    trial["x"], trial["target"], representation, condition, params,
                    trial["observed_k"],
                )
            )
            sampled = []
            for proposal_index, (subset, probability, delta) in enumerate(distribution):
                profile = np.clip(trial["x"] + delta, 0.0, 1.0)
                key = f"{participant}|{case_id}|{proposal_index}"
                profiles[key] = profile
                sampled.append(
                    {
                        "weight": float(probability),
                        "subset": subset,
                        "subset probability": float(probability),
                        "delta": delta,
                        "profile": profile,
                        "profile key": key,
                        "agreement": case_metrics(delta, trial["observed"]),
                    }
                )
            pending.append(
                {
                    "participant": participant,
                    "xai": condition,
                    "case": source["case"],
                    "instance id": case_id,
                    "selected model family": fit["model family"],
                    "trial": trial,
                    "source": source,
                    "case data": case,
                    "sampled": sampled,
                }
            )

    print(f"Solving boundaries for {len(profiles)} possible subset proposals...", flush=True)
    boundaries = exact_boundaries(browser_model, dataset["test_pool"][0], profiles)
    trial_rows = []
    for item in pending:
        trial = item["trial"]
        source = item["source"]
        case = item["case data"]
        target_label = source["target label"]
        original_probability = target_probability(trial["x"], target_label, case, browser_model)

        def expected(function):
            return float(sum(sample["weight"] * function(sample) for sample in item["sampled"]))

        def expected_finite(function):
            values = [
                (sample["weight"], float(function(sample)))
                for sample in item["sampled"]
            ]
            finite = [(weight, value) for weight, value in values if math.isfinite(value)]
            total_weight = sum(weight for weight, _ in finite)
            return (
                float(sum(weight * value for weight, value in finite) / total_weight)
                if total_weight > 0
                else math.nan
            )

        row = {
            "participant": item["participant"],
            "xai": item["xai"],
            "case": item["case"],
            "instance id": item["instance id"],
            "selected model family": item["selected model family"],
            "possible subsets": len(item["sampled"]),
            "mean selection F1": expected(lambda sample: sample["agreement"][1]),
            "mean amount MAE": expected(lambda sample: sample["agreement"][2]),
            "mean direction accuracy": expected_finite(lambda sample: sample["agreement"][3]),
            "observed success": float(source["successful counterfactual (0/1)"]),
            "simulated success": expected(
                lambda sample: prediction_label(sample["profile"], case, browser_model) == target_label
            ),
            "observed target confidence gain": float(source["delta confidence of target label"]),
            "simulated target confidence gain": expected(
                lambda sample: target_probability(sample["profile"], target_label, case, browser_model)
                - original_probability
            ),
            "observed boundary improvement": -float(
                source["boundary distance change (new - original)"]
            ),
            "simulated boundary improvement": expected(
                lambda sample: float(source["boundary distance original"])
                - boundaries[sample["profile key"]]
            ),
            "observed plausibility": float(source["plausibility"]),
            "simulated plausibility": expected(
                lambda sample: similarity(sample["profile"], reference)
            ),
            "observed actionability": float(source["actionability (0/1)"]),
            "simulated actionability": expected(
                lambda sample: abs(sample["delta"][4]) <= 1e-9
            ),
            "observed edit L1": float(
                sum(abs(float(source[f"x_{index}_change"])) for index in range(1, 6))
            ),
            "simulated edit L1": expected(lambda sample: np.abs(sample["delta"]).sum()),
        }
        trial_rows.append(row)
    write_csv(TRIAL_OUTPUT, trial_rows)

    by_participant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        by_participant[str(row["participant"])].append(row)
    participant_rows = []
    for participant, values in sorted(by_participant.items()):
        output = {
            "participant": participant,
            "xai": values[0]["xai"],
            "selected model family": values[0]["selected model family"],
            "testing cases": len(values),
            "mean selection F1": float(np.mean([row["mean selection F1"] for row in values])),
            "mean amount MAE": float(np.mean([row["mean amount MAE"] for row in values])),
            "mean direction accuracy": float(np.nanmean([row["mean direction accuracy"] for row in values])),
        }
        for metric in METRICS:
            for source in ("observed", "simulated"):
                output[f"{source} {metric}"] = float(
                    np.mean([row[f"{source} {metric}"] for row in values])
                )
        participant_rows.append(output)
    write_csv(PARTICIPANT_OUTPUT, participant_rows)

    metric_results = {}
    for metric, label in METRICS.items():
        observed = np.asarray([row[f"observed {metric}"] for row in participant_rows])
        simulated = np.asarray([row[f"simulated {metric}"] for row in participant_rows])
        pearson = stats.pearsonr(observed, simulated)
        spearman = stats.spearmanr(observed, simulated)
        metric_results[metric] = {
            "label": label,
            "n": len(observed),
            "observed mean": float(observed.mean()),
            "simulated mean": float(simulated.mean()),
            "pearson r": float(pearson.statistic),
            "pearson p": float(pearson.pvalue),
            "spearman rho": float(spearman.statistic),
            "spearman p": float(spearman.pvalue),
            "MAE": float(np.mean(np.abs(observed - simulated))),
        }
    summary = {
        "model": "additive-rho probabilistic attribute-selection model",
        "simulation": "exact probability-weighted marginalization over all subsets conditional on observed trial K",
        "excluded participants": sorted(EXCLUDED),
        "participants": len(participant_rows),
        "trials": len(trial_rows),
        "metrics": metric_results,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
