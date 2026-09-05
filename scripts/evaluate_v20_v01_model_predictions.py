"""Compare participant outcomes with winning cognitive-model v0.1 predictions."""

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
    METRICS,
    exact_boundaries,
    prediction_label,
    read_csv,
    target_probability,
    write_csv,
)
from scripts.fit_parsimonious_weighted_models import case_metrics  # noqa: E402
from scripts.run_model_only_strategy_study import (  # noqa: E402
    reference_matrices,
    similarity,
)
from src.cognitive_models.v0_1.model import (  # noqa: E402
    build_memory,
    contribution_distribution,
    exemplar_distribution,
    observed_trials,
)


RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
FITS = ROOT / "qualtrics" / "v20_cognitive_model_v01_fits.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTPUT_DIR = ROOT / "outputs" / "v20-v01-participant-correlations"
TRIAL_OUTPUT = OUTPUT_DIR / "v01_model_prediction_trials.csv"
PARTICIPANT_OUTPUT = OUTPUT_DIR / "v01_model_prediction_participants.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "v01_model_prediction_summary.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    browser_model = dataset["browser_model"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training = dataset["training_pool"]
    reference = reference_matrices("diabetes", dataset)[0]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(RESULTS):
        grouped[row["participant"]].append(row)
    fit_map = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["selected family by CV"] == "1"
    }
    if len(grouped) != 62 or len(fit_map) != 62 or set(grouped) != set(fit_map):
        raise RuntimeError(
            f"Expected the same 62 participants in results and fits; found "
            f"{len(grouped)} and {len(fit_map)}"
        )

    trial_rows: list[dict[str, object]] = []
    model_profiles: dict[str, np.ndarray] = {}
    for participant, participant_rows in sorted(grouped.items()):
        fit = fit_map[participant]
        condition = fit["xai"]
        parameters = {
            name: float(fit[name]) if fit[name] not in ("", "nan") else math.nan
            for name in ("eta", "alpha", "rho", "lambda", "beta")
        }
        parameters["age actionable"] = int(float(fit["age actionable"]))
        parameters["immutable index"] = int(float(fit["immutable index"]))
        memory = build_memory(training, condition, parameters["eta"])
        trials = observed_trials(participant_rows, case_map)
        source_testing = sorted(
            (row for row in participant_rows if row["phase"] == "testing"),
            key=lambda row: int(float(row["case"])),
        )
        if len(trials) != 20 or len(source_testing) != 20:
            raise RuntimeError(f"Participant {participant} does not have 20 tests")

        for trial, source in zip(trials, source_testing):
            case_id = int(float(source["instance id"]))
            case = case_map[case_id]
            if fit["model family"] == "feature contribution":
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
                    condition,
                    parameters,
                    trial["observed_k"],
                )
            if not distribution:
                raise RuntimeError(
                    f"No eligible prediction for {participant}, instance {case_id}"
                )
            _, subset_probability, model_delta = max(
                distribution, key=lambda item: item[1]
            )
            model_profile = np.clip(trial["profile"] + model_delta, 0.0, 1.0)
            agreement = case_metrics(model_delta, trial["observed"])
            target_label = source["target label"]
            original_target_probability = target_probability(
                trial["profile"], target_label, case, browser_model
            )
            model_target_probability = target_probability(
                model_profile, target_label, case, browser_model
            )
            profile_key = f"{participant}|{case_id}"
            model_profiles[profile_key] = model_profile
            trial_rows.append(
                {
                    "participant": participant,
                    "xai": condition,
                    "case": source["case"],
                    "instance id": case_id,
                    "selected model family": fit["model family"],
                    "model MAP subset probability": float(subset_probability),
                    "selection F1": float(agreement[1]),
                    "amount MAE": float(agreement[2]),
                    "direction accuracy": float(agreement[3]),
                    "observed success": float(source["successful counterfactual (0/1)"]),
                    "simulated success": float(
                        prediction_label(model_profile, case, browser_model)
                        == target_label
                    ),
                    "observed target confidence gain": float(
                        source["delta confidence of target label"]
                    ),
                    "simulated target confidence gain": float(
                        model_target_probability - original_target_probability
                    ),
                    "observed boundary improvement": -float(
                        source["boundary distance change (new - original)"]
                    ),
                    "observed boundary distance original": float(
                        source["boundary distance original"]
                    ),
                    "observed plausibility": float(source["plausibility"]),
                    "simulated plausibility": similarity(model_profile, reference),
                    "observed actionability": float(source["actionability (0/1)"]),
                    "simulated actionability": float(abs(model_delta[4]) <= 1e-9),
                    "observed edit L1": float(
                        sum(
                            abs(float(source[f"x_{index}_change"]))
                            for index in range(1, 6)
                        )
                    ),
                    "simulated edit L1": float(np.abs(model_delta).sum()),
                    "_profile_key": profile_key,
                }
            )

    print(f"boundary_profiles={len(model_profiles)}", flush=True)
    boundaries = exact_boundaries(
        browser_model, dataset["test_pool"][0], model_profiles
    )
    for row in trial_rows:
        row["simulated boundary distance new"] = boundaries[row.pop("_profile_key")]
        row["simulated boundary improvement"] = (
            float(row["observed boundary distance original"])
            - float(row["simulated boundary distance new"])
        )
    write_csv(TRIAL_OUTPUT, trial_rows)

    grouped_trials: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped_trials[str(row["participant"])].append(row)
    participant_rows = []
    for participant, rows in sorted(grouped_trials.items()):
        participant_result: dict[str, object] = {
            "participant": participant,
            "xai": rows[0]["xai"],
            "selected model family": rows[0]["selected model family"],
            "testing cases": len(rows),
            "mean selection F1": float(np.mean([row["selection F1"] for row in rows])),
            "mean amount MAE": float(np.mean([row["amount MAE"] for row in rows])),
            "mean direction accuracy": float(
                np.nanmean([row["direction accuracy"] for row in rows])
            ),
        }
        for metric in METRICS:
            for source in ("observed", "simulated"):
                participant_result[f"{source} {metric}"] = float(
                    np.mean([row[f"{source} {metric}"] for row in rows])
                )
        participant_rows.append(participant_result)
    write_csv(PARTICIPANT_OUTPUT, participant_rows)

    metric_results = {}
    for metric, label in METRICS.items():
        observed = np.asarray(
            [row[f"observed {metric}"] for row in participant_rows]
        )
        simulated = np.asarray(
            [row[f"simulated {metric}"] for row in participant_rows]
        )
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
        "model": "cognitive model v0.1",
        "participants": len(participant_rows),
        "testing_trials": len(trial_rows),
        "simulation": (
            "Maximum-probability feature subset for the winning family at each "
            "participant's fitted v0.1 parameters and observed trial K"
        ),
        "aggregation": (
            "Participant and model performance averaged over the same 20 testing "
            "cases before across-participant correlation"
        ),
        "metrics": metric_results,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
