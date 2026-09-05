"""Simulate all 104 older participants using conditional-rho fitted models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calculate_v09_boundary_distance import WachterBoundarySearch, model_logit
from scripts.fit_parsimonious_weighted_models import case_metrics, label_sign, normalize_case, observed_trials, training_representation
from scripts.fit_probabilistic_attribute_selection import contribution_distribution, memory_distribution
from scripts.run_model_only_strategy_study import reference_matrices, similarity

RESULTS = ROOT / "qualtrics" / "qualtrics_results_v1.1.schema-export.csv"
FITS = ROOT / "qualtrics" / "parsimonious_probabilistic_model_fits_v1.4_all_old_domains.csv"
BUNDLE = ROOT / "static" / "experiment-data.json"
TRIAL_OUTPUT = ROOT / "qualtrics" / "v1.4_all_old_domains_model_vs_baseline_trials.csv"
PARTICIPANT_OUTPUT = ROOT / "qualtrics" / "v1.4_all_old_domains_model_vs_participant_statistics.csv"
SUMMARY_OUTPUT = ROOT / "qualtrics" / "v1.4_all_old_domains_model_vs_baseline_summary.json"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def prediction_label(profile, case, browser_model):
    logit = model_logit(browser_model, profile, case)
    labels = case["prediction_labels"]
    return labels[1] if logit >= 0 else labels[0]


def baseline_delta(participant, case_id, x, target, rep, k, age_actionable, immutable_index):
    eligible = list(range(5))
    if not age_actionable and immutable_index in eligible:
        eligible.remove(immutable_index)
    seed = int.from_bytes(hashlib.sha256(f"{participant}|{case_id}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=min(k, len(eligible)), replace=False)
    direction = target * rep["polarity"]
    delta = np.zeros(5)
    delta[selected] = 0.10 * direction[selected]
    return np.clip(x + delta, 0.0, 1.0) - x


def boundary_chunk(args):
    browser_model, representative_case, items = args
    solver = WachterBoundarySearch(
        browser_model,
        representative_case,
        allow_reference_outside_bounds=True,
    )
    return [(key, solver.solve(np.asarray(profile, dtype=float)).distance) for key, profile in items]


def exact_boundaries(browser_model, representative_case, profiles):
    unique = {}
    for key, profile in profiles.items():
        unique.setdefault(tuple(np.round(profile, 12)), []).append(key)
    items = [(profile, list(profile)) for profile in unique]
    chunks = [items[index::4] for index in range(4)]
    solved = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(boundary_chunk, [(browser_model, representative_case, chunk) for chunk in chunks]):
            solved.update(result)
    return {key: solved[tuple(np.round(profile, 12))] for key, profile in profiles.items()}


def correlations(rows, prefix):
    result = {}
    for metric in ("validity", "boundary distance", "plausibility"):
        observed = np.asarray([row[f"observed {metric}"] for row in rows], dtype=float)
        predicted = np.asarray([row[f"{prefix} {metric}"] for row in rows], dtype=float)
        result[metric] = {
            "pearson r": float(np.corrcoef(observed, predicted)[0, 1]),
            "MAE": float(np.mean(np.abs(observed - predicted))),
            "observed mean": float(observed.mean()),
            "predicted mean": float(predicted.mean()),
        }
    return result


def main():
    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    datasets = experiment["datasets"]
    case_maps = {
        domain: {
            int(case["instance_id"]): case
            for case in dataset["training_pool"] + dataset["test_pool"]
        }
        for domain, dataset in datasets.items()
    }
    references = {domain: reference_matrices(domain, dataset)[0] for domain, dataset in datasets.items()}

    rows = read_csv(RESULTS)
    grouped = defaultdict(list)
    for row in rows: grouped[row["participant"]].append(row)
    fits = [row for row in read_csv(FITS) if row["selected family by CV"] == "1"]
    fit_map = {row["participant"]: row for row in fits}
    grouped = defaultdict(list, {
        participant: participant_rows for participant, participant_rows in grouped.items()
        if participant in fit_map
    })
    training_by_domain = {}
    for domain in datasets:
        training_ids = sorted({
            int(float(row["instance id"]))
            for row in rows if row["domain"] == domain and row["phase"] == "training"
        })
        training_by_domain[domain] = [case_maps[domain][case_id] for case_id in training_ids]

    trial_rows = []
    profiles = {}
    for participant, participant_rows in sorted(grouped.items()):
        fit = fit_map[participant]
        domain = fit["domain"]
        condition = fit["xai"]
        params = {name: float(fit[name]) if fit[name] not in ("", "nan") else math.nan for name in ("eta", "alpha", "rho", "lambda", "beta")}
        params["age actionable"] = int(float(fit["age actionable"]))
        params["immutable index"] = int(float(fit["immutable index"]))
        rep = training_representation(training_by_domain[domain], condition, params["eta"])
        trials = observed_trials(participant_rows, case_maps[domain])
        source_testing = sorted((row for row in participant_rows if row["phase"] == "testing"), key=lambda row: int(float(row["case"])))
        for trial, source in zip(trials, source_testing):
            case_id = int(float(source["instance id"]))
            if fit["model family"] == "feature contribution":
                distribution = contribution_distribution(trial["x"], trial["target"], rep, params, trial["observed_k"])
            else:
                distribution = memory_distribution(trial["x"], trial["target"], rep, condition, params, trial["observed_k"])
            _, probability, model_delta = max(distribution, key=lambda item: item[1])
            baseline = baseline_delta(participant, case_id, trial["x"], trial["target"], rep, trial["observed_k"], params["age actionable"], params["immutable index"])
            model_profile = np.clip(trial["x"] + model_delta, 0.0, 1.0)
            baseline_profile = np.clip(trial["x"] + baseline, 0.0, 1.0)
            categorical_indices = [
                index for index, kind in enumerate(case_maps[domain][case_id]["feature_types"])
                if kind == "categorical"
            ]
            if categorical_indices:
                model_profile[categorical_indices] = np.round(model_profile[categorical_indices])
                baseline_profile[categorical_indices] = np.round(baseline_profile[categorical_indices])
                model_delta = model_profile - trial["x"]
                baseline = baseline_profile - trial["x"]
            target_label = source["target label"]
            model_metric = case_metrics(model_delta, trial["observed"])
            baseline_metric = case_metrics(baseline, trial["observed"])
            key_model = f"{domain}|{participant}|{case_id}|model"
            key_baseline = f"{domain}|{participant}|{case_id}|baseline"
            profiles[key_model] = model_profile
            profiles[key_baseline] = baseline_profile
            trial_rows.append({
                "participant": participant, "domain": domain, "xai": condition, "case": source["case"], "instance id": case_id,
                "selected model family": fit["model family"], "model MAP subset probability": probability,
                "observed selection F1 model": model_metric[1], "observed selection F1 baseline": baseline_metric[1],
                "observed amount MAE model": model_metric[2], "observed amount MAE baseline": baseline_metric[2],
                "observed validity": float(source["successful counterfactual (0/1)"]),
                "observed boundary distance": float(source["boundary distance new"]),
                "observed plausibility": float(source["plausibility"]),
                "model validity": float(prediction_label(model_profile, case_maps[domain][case_id], datasets[domain]["browser_model"]) == target_label),
                "baseline validity": float(prediction_label(baseline_profile, case_maps[domain][case_id], datasets[domain]["browser_model"]) == target_label),
                "model plausibility": similarity(model_profile, references[domain]),
                "baseline plausibility": similarity(baseline_profile, references[domain]),
                "_model_key": key_model, "_baseline_key": key_baseline,
            })

    print(f"boundary_profiles={len(profiles)}", flush=True)
    boundaries = {}
    for domain, dataset in datasets.items():
        domain_profiles = {key: profile for key, profile in profiles.items() if key.startswith(domain + "|")}
        print(f"boundary_domain={domain} profiles={len(domain_profiles)}", flush=True)
        boundaries.update(exact_boundaries(dataset["browser_model"], dataset["test_pool"][0], domain_profiles))
    for row in trial_rows:
        row["model boundary distance"] = boundaries[row.pop("_model_key")]
        row["baseline boundary distance"] = boundaries[row.pop("_baseline_key")]
    write_csv(TRIAL_OUTPUT, trial_rows)

    participant_rows = []
    grouped_trials = defaultdict(list)
    for row in trial_rows: grouped_trials[row["participant"]].append(row)
    for participant, values in sorted(grouped_trials.items()):
        result = {"participant": participant, "domain": values[0]["domain"], "xai": values[0]["xai"], "selected model family": values[0]["selected model family"]}
        for metric in ("validity", "boundary distance", "plausibility"):
            for source in ("observed", "model", "baseline"):
                result[f"{source} {metric}"] = float(np.mean([row[f"{source} {metric}"] for row in values]))
        result["model selection F1"] = float(np.mean([row["observed selection F1 model"] for row in values]))
        result["baseline selection F1"] = float(np.mean([row["observed selection F1 baseline"] for row in values]))
        result["model amount MAE"] = float(np.mean([row["observed amount MAE model"] for row in values]))
        result["baseline amount MAE"] = float(np.mean([row["observed amount MAE baseline"] for row in values]))
        participant_rows.append(result)
    write_csv(PARTICIPANT_OUTPUT, participant_rows)

    summary = {
        "participants": len(participant_rows),
        "domains": {domain: sum(row["domain"] == domain for row in participant_rows) for domain in datasets},
        "baseline": "observed K and fitted actionability; uniformly random eligible feature subset; fixed 0.10 normalized target-direction change",
        "model_prediction": "maximum-probability attribute subset under each participant's full-data fitted probabilistic model",
        "selection": {
            "model F1": float(np.mean([row["model selection F1"] for row in participant_rows])),
            "baseline F1": float(np.mean([row["baseline selection F1"] for row in participant_rows])),
            "model amount MAE": float(np.mean([row["model amount MAE"] for row in participant_rows])),
            "baseline amount MAE": float(np.mean([row["baseline amount MAE"] for row in participant_rows])),
        },
        "model": correlations(participant_rows, "model"),
        "baseline downstream": correlations(participant_rows, "baseline"),
        "model by domain": {
            domain: correlations([row for row in participant_rows if row["domain"] == domain], "model")
            for domain in datasets
        },
        "model by condition": {
            condition: correlations([row for row in participant_rows if row["xai"] == condition], "model")
            for condition in ("none", "attribution", "counterfactual")
        },
        "model by domain and condition": {
            domain: {
                condition: correlations([
                    row for row in participant_rows
                    if row["domain"] == domain and row["xai"] == condition
                ], "model")
                for condition in ("none", "attribution", "counterfactual")
            }
            for domain in datasets
        },
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
