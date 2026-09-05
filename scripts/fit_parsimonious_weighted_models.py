"""Fit two parsimonious edit models and audit parameter recovery.

The analysis uses only complete diabetes participants with 12 training and 20
testing rows.  It combines the 29 such participants in v1.5 with the one new
participant-only v1.6 file.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V15 = ROOT / "qualtrics" / "qualtrics_results_v1.5.csv"
V16 = ROOT / "qualtrics" / "qualtrics_results_v1.6.csv"
BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_results.json"
OUT_FITS = ROOT / "qualtrics" / "parsimonious_model_fits_v1.0.csv"
OUT_RECOVERY = ROOT / "qualtrics" / "parsimonious_parameter_recovery_v1.0.csv"
OUT_SUMMARY = ROOT / "qualtrics" / "parsimonious_model_summary_v1.0.json"

FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
ETA_GRID = (0.0, 0.5, 1.0)
ALPHA_GRID = (0.0, 0.5, 1.0)
RHO_GRID = (0.5, 1.0, 1.5)
LAMBDA_GRID = (0.0, 2.0, 6.0)
BETA_GRID = (0.0, 0.5, 1.0)
EPS = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_case(case: dict) -> np.ndarray:
    result = []
    for value, bounds in zip(case["feature_values"], case["feature_ranges"]):
        try:
            low, high = map(float, bounds)
            normalized = (float(value) - low) / (high - low)
        except (TypeError, ValueError):
            categories = [str(item).strip().lower() for item in bounds]
            normalized = categories.index(str(value).strip().lower()) / max(len(categories) - 1, 1)
        result.append(np.clip(normalized, 0.0, 1.0))
    return np.asarray(result, dtype=float)


def normalize_values(values: list[float], case: dict) -> np.ndarray:
    result = []
    for value, bounds in zip(values, case["feature_ranges"]):
        try:
            low, high = map(float, bounds)
            normalized = (float(value) - low) / (high - low)
        except (TypeError, ValueError):
            categories = [str(item).strip().lower() for item in bounds]
            normalized = categories.index(str(value).strip().lower()) / max(len(categories) - 1, 1)
        result.append(np.clip(normalized, 0.0, 1.0))
    return np.asarray(result, dtype=float)


def label_sign(label: str) -> int:
    positive_labels = {"diabetes", "expensive", "below limit"}
    return 1 if str(label).strip().lower() in positive_labels else -1


def unit(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    return values / total if total > EPS else np.full(len(values), 1.0 / len(values))


def training_representation(cases: list[dict], condition: str, eta: float) -> dict:
    x = np.vstack([normalize_case(case) for case in cases])
    y = np.asarray([label_sign(case["prediction"]["label"]) for case in cases])
    positive, negative = x[y == 1], x[y == -1]
    mu_pos, mu_neg = positive.mean(axis=0), negative.mean(axis=0)
    centre = (mu_pos + mu_neg) / 2.0
    diagnosticity = unit(np.abs(mu_pos - mu_neg) / (positive.std(axis=0) + negative.std(axis=0) + EPS))
    polarity = np.sign(mu_pos - mu_neg)
    polarity[polarity == 0] = 1

    if condition == "attribution":
        explanation = unit(np.mean([
            np.abs(np.asarray(case["attribution"]["values"], dtype=float)) for case in cases
        ], axis=0))
    elif condition == "counterfactual":
        explanation = unit(np.mean([
            np.abs(normalize_values(case["counterfactual"]["feature_values"], case) - normalize_case(case))
            for case in cases
        ], axis=0))
    else:
        explanation = diagnosticity
        eta = 0.0
    relevance = unit((1.0 - eta) * diagnosticity + eta * explanation)
    return {
        "x": x, "y": y, "centre": centre, "polarity": polarity,
        "relevance": relevance,
        "endpoints": np.vstack([normalize_values(case["counterfactual"]["feature_values"], case) for case in cases]),
    }


def contribution_prediction(x: np.ndarray, target: int, rep: dict, k: int, rho: float, alpha: float) -> np.ndarray:
    theta = rep["relevance"] * rep["polarity"]
    opposition = -target * theta * (x - rep["centre"])
    # Shift-and-normalize preserves the complete case-specific ranking at
    # alpha=0, including differences among features that already support the
    # target.  Pure global relevance is recovered exactly at alpha=1.
    local_score = unit(opposition - float(np.min(opposition)))
    selection_score = (1.0 - alpha) * local_score + alpha * rep["relevance"]
    order = np.argsort(selection_score)[::-1]
    selected = order[:k]
    destination = np.where(target * theta >= 0, 1.0, 0.0)
    # Allocate the edit along a relevance-weighted feasible-change vector.
    # Consequently, a selected feature receives a larger change when it is
    # more relevant and/or has more room to move in the target direction.
    direction = rep["relevance"] * (destination - x)
    base = float(target * np.dot(theta, x - rep["centre"]))
    gain = float(target * np.dot(theta[selected], direction[selected]))
    if gain <= EPS:
        fraction = 0.0
    else:
        # Relevance scales the direction, so the largest feasible common
        # multiplier is 1 / max(relevance) among the selected features.
        # Clipping at that value lets the most relevant selected feature use
        # its full available range without prematurely truncating at 1.
        max_relevance = float(np.max(rep["relevance"][selected])) if len(selected) else 0.0
        max_fraction = 1.0 / max(max_relevance, EPS)
        fraction = np.clip(-base / gain, 0.0, max_fraction)
        if fraction <= EPS:
            fraction = min(0.1, 1.0 / max(rho, EPS))
    delta = np.zeros(5)
    delta[selected] = np.clip(rho * fraction * direction[selected], -1.0, 1.0)
    return np.clip(x + delta, 0.0, 1.0) - x


def memory_prediction(
    x: np.ndarray, target: int, rep: dict, condition: str,
    k: int, rho: float, locality: float, beta: float,
) -> np.ndarray:
    relevance = rep["relevance"]
    if condition == "counterfactual":
        eligible = np.where(-rep["y"] == target)[0]
        sources = rep["x"][eligible]
        endpoints = rep["endpoints"][eligible]
        changes = endpoints - sources
        distances = np.sum(relevance * np.abs(sources - x), axis=1)
        weights = np.exp(-locality * distances)
        weights /= weights.sum()
        endpoint_vector = weights @ endpoints - x
        edit_vector = weights @ changes
        vector = (1.0 - beta) * endpoint_vector + beta * edit_vector
    else:
        eligible = np.where(rep["y"] == target)[0]
        examples = rep["x"][eligible]
        distances = np.sum(relevance * np.abs(examples - x), axis=1)
        weights = np.exp(-locality * distances)
        weights /= weights.sum()
        vector = weights @ examples - x
    selected = np.argsort(np.abs(vector))[::-1][:k]
    delta = np.zeros(5)
    delta[selected] = rho * vector[selected]
    return np.clip(x + delta, 0.0, 1.0) - x


def selection_f1(pred: np.ndarray, observed: np.ndarray) -> float:
    p = set(np.where(np.abs(pred) > 1e-8)[0])
    o = set(np.where(np.abs(observed) > 1e-8)[0])
    if not p and not o:
        return 1.0
    if not p or not o:
        return 0.0
    return 2.0 * len(p & o) / (len(p) + len(o))


def case_metrics(pred: np.ndarray, observed: np.ndarray) -> tuple[float, float, float, float]:
    f1 = selection_f1(pred, observed)
    union = np.abs(pred) + np.abs(observed) > 1e-8
    amount_mae = float(np.mean(np.abs(pred[union] - observed[union]))) if union.any() else 0.0
    intersection = (np.abs(pred) > 1e-8) & (np.abs(observed) > 1e-8)
    direction = float(np.mean(np.sign(pred[intersection]) == np.sign(observed[intersection]))) if intersection.any() else math.nan
    loss = 0.5 * (1.0 - f1) + 0.5 * amount_mae
    return loss, f1, amount_mae, direction


def parameter_grid(condition: str, family: str):
    etas = (0.0,) if condition == "none" else ETA_GRID
    if family == "feature contribution":
        for eta, alpha, rho in itertools.product(etas, ALPHA_GRID, RHO_GRID):
            yield {"eta": eta, "alpha": alpha, "rho": rho, "lambda": math.nan, "beta": math.nan}
    else:
        betas = BETA_GRID if condition == "counterfactual" else (0.0,)
        for eta, rho, locality, beta in itertools.product(etas, RHO_GRID, LAMBDA_GRID, betas):
            yield {"eta": eta, "alpha": math.nan, "rho": rho, "lambda": locality, "beta": beta}


def predict(family: str, condition: str, x: np.ndarray, target: int, rep: dict, params: dict, k: int) -> np.ndarray:
    if family == "feature contribution":
        return contribution_prediction(x, target, rep, k, params["rho"], params["alpha"])
    return memory_prediction(x, target, rep, condition, k, params["rho"], params["lambda"], params["beta"])


def load_participants() -> dict[str, list[dict[str, str]]]:
    combined = read_csv(V15) + read_csv(V16)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combined:
        grouped[row["participant"]].append(row)
    return {
        participant: rows for participant, rows in grouped.items()
        if len(rows) == 32
        and sum(row["phase"] == "training" for row in rows) == 12
        and sum(row["phase"] == "testing" for row in rows) == 20
    }


def observed_trials(rows: list[dict[str, str]], case_map: dict[int, dict]):
    trials = []
    for row in sorted((r for r in rows if r["phase"] == "testing"), key=lambda r: int(float(r["case"]))):
        case_id = int(float(row["instance id"]))
        case = case_map[case_id]
        trials.append({
            "case": case,
            "x": normalize_case(case),
            "target": label_sign(row["target label"]),
            "observed": np.asarray([float(row[f"x_{i}_change"]) for i in range(1, 6)]),
        })
        trials[-1]["observed_k"] = int(np.sum(np.abs(trials[-1]["observed"]) > 1e-8))
    return trials


def evaluate_params(family: str, condition: str, params: dict, training: list[dict], trials: list[dict], indices=None):
    rep = training_representation(training, condition, params["eta"])
    selected = range(len(trials)) if indices is None else indices
    metrics = []
    for index in selected:
        trial = trials[index]
        proposal = predict(family, condition, trial["x"], trial["target"], rep, params, trial["observed_k"])
        metrics.append(case_metrics(proposal, trial["observed"]))
    values = np.asarray([[m[0], m[1], m[2]] for m in metrics])
    directions = [m[3] for m in metrics if math.isfinite(m[3])]
    return {
        "loss": float(values[:, 0].mean()),
        "selection_f1": float(values[:, 1].mean()),
        "amount_mae": float(values[:, 2].mean()),
        "direction_accuracy": float(np.mean(directions)) if directions else math.nan,
    }


def fit_family(family: str, condition: str, training: list[dict], trials: list[dict]):
    candidates = list(parameter_grid(condition, family))
    scored = [(evaluate_params(family, condition, p, training, trials), p) for p in candidates]
    full_metric, best = min(scored, key=lambda item: (item[0]["loss"], tuple(str(v) for v in item[1].values())))
    fold_metrics = []
    for fold in range(5):
        train_idx = [i for i in range(20) if i % 5 != fold]
        test_idx = [i for i in range(20) if i % 5 == fold]
        fold_best = min(candidates, key=lambda p: evaluate_params(family, condition, p, training, trials, train_idx)["loss"])
        fold_metrics.append(evaluate_params(family, condition, fold_best, training, trials, test_idx))
    cv = {name: float(np.nanmean([m[name] for m in fold_metrics])) for name in full_metric}
    return best, full_metric, cv


def recovery(training: list[dict], test_cases: list[dict], recovery_k: list[int], condition: str, family: str, rng: np.random.Generator):
    candidates = list(parameter_grid(condition, family))
    reps = {eta: training_representation(training, condition, eta) for eta in sorted({p["eta"] for p in candidates})}
    targets = [-label_sign(case["prediction"]["label"]) for case in test_cases]
    xs = [normalize_case(case) for case in test_cases]
    predictions = np.asarray([
        np.vstack([predict(family, condition, x, target, reps[p["eta"]], p, k) for x, target, k in zip(xs, targets, recovery_k)])
        for p in candidates
    ])
    sample_indices = rng.choice(len(candidates), size=min(40, len(candidates)), replace=False)
    rows = []
    noise_settings = (
        ("none", 0.0, 0.0),
        ("amount noise sd=.05", 0.05, 0.0),
        ("selection swaps 10%", 0.0, 0.10),
        ("selection swaps 20%", 0.0, 0.20),
        ("selection swaps 40%", 0.0, 0.40),
        ("amount sd=.05 + selection swaps 20%", 0.05, 0.20),
    )
    for noise_name, sigma, selection_noise in noise_settings:
        exact = Counter()
        absolute_errors = defaultdict(list)
        for index in sample_indices:
            observed = predictions[index].copy()
            if sigma:
                active = np.abs(observed) > 1e-8
                observed[active] += rng.normal(0.0, sigma, size=int(active.sum()))
                observed = np.clip(observed, -1.0, 1.0)
            if selection_noise:
                observed = corrupt_selection(observed, selection_noise, rng)
            losses = []
            for proposal in predictions:
                losses.append(np.mean([case_metrics(p, o)[0] for p, o in zip(proposal, observed)]))
            recovered_index = int(np.argmin(losses))
            true, recovered = candidates[index], candidates[recovered_index]
            for key in ("eta", "alpha", "rho", "lambda", "beta"):
                if math.isfinite(float(true[key])):
                    exact[key] += int(float(true[key]) == float(recovered[key]))
                    absolute_errors[key].append(abs(float(true[key]) - float(recovered[key])))
        n = len(sample_indices)
        row = {"condition": condition, "model family": family, "noise": noise_name, "simulations": n}
        for key in ("eta", "alpha", "rho", "lambda", "beta"):
            row[f"{key} exact recovery"] = exact[key] / n if key in absolute_errors else ""
            row[f"{key} mean absolute error"] = float(np.mean(absolute_errors[key])) if key in absolute_errors else ""
        rows.append(row)
    return rows


def corrupt_selection(observed: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    result = observed.copy()
    for row in result:
        if rng.random() >= probability:
            continue
        active = np.where(np.abs(row) > 1e-8)[0]
        inactive = np.where(np.abs(row) <= 1e-8)[0]
        if not len(active) or not len(inactive):
            continue
        removed = int(rng.choice(active))
        added = int(rng.choice(inactive))
        amount = row[removed]
        row[removed] = 0.0
        row[added] = amount if rng.random() < 0.75 else -amount
    return result


def family_recovery(training: list[dict], test_cases: list[dict], recovery_k: list[int], condition: str, rng: np.random.Generator):
    targets = [-label_sign(case["prediction"]["label"]) for case in test_cases]
    xs = [normalize_case(case) for case in test_cases]
    libraries = {}
    parameter_sets = {}
    for family in ("feature contribution", "weighted examples"):
        params = list(parameter_grid(condition, family))
        reps = {eta: training_representation(training, condition, eta) for eta in sorted({p["eta"] for p in params})}
        libraries[family] = np.asarray([
            np.vstack([predict(family, condition, x, target, reps[p["eta"]], p, k) for x, target, k in zip(xs, targets, recovery_k)])
            for p in params
        ])
        parameter_sets[family] = params
    rows = []
    for true_family in libraries:
        sample = rng.choice(len(libraries[true_family]), size=min(40, len(libraries[true_family])), replace=False)
        noise_settings = (
            ("none", 0.0, 0.0),
            ("amount noise sd=.05", 0.05, 0.0),
            ("selection swaps 10%", 0.0, 0.10),
            ("selection swaps 20%", 0.0, 0.20),
            ("selection swaps 40%", 0.0, 0.40),
            ("amount sd=.05 + selection swaps 10%", 0.05, 0.10),
            ("amount sd=.05 + selection swaps 20%", 0.05, 0.20),
        )
        for noise_name, sigma, selection_noise in noise_settings:
            correct = 0
            margins = []
            for index in sample:
                observed = libraries[true_family][index].copy()
                if sigma:
                    active = np.abs(observed) > 1e-8
                    observed[active] += rng.normal(0.0, sigma, size=int(active.sum()))
                    observed = np.clip(observed, -1.0, 1.0)
                if selection_noise:
                    observed = corrupt_selection(observed, selection_noise, rng)
                best_losses = {}
                for candidate_family, proposals in libraries.items():
                    best_losses[candidate_family] = min(
                        np.mean([case_metrics(p, o)[0] for p, o in zip(proposal, observed)])
                        for proposal in proposals
                    )
                recovered = min(best_losses, key=best_losses.get)
                correct += int(recovered == true_family)
                other = next(name for name in best_losses if name != true_family)
                margins.append(best_losses[other] - best_losses[true_family])
            rows.append({
                "condition": condition, "model family": true_family,
                "noise": noise_name + " | family recovery", "simulations": len(sample),
                "eta exact recovery": "", "eta mean absolute error": "",
                "alpha exact recovery": "", "alpha mean absolute error": "",
                "rho exact recovery": "", "rho mean absolute error": "",
                "lambda exact recovery": "", "lambda mean absolute error": "",
                "beta exact recovery": "", "beta mean absolute error": "",
                "family exact recovery": correct / len(sample),
                "mean true-vs-other loss margin": float(np.mean(margins)),
            })
    return rows


def main() -> None:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    training = bundle["training_pool"]
    case_map = {int(case["instance_id"]): case for case in bundle["test_pool"]}
    participants = load_participants()
    if len(participants) != 30:
        raise RuntimeError(f"Expected 30 complete participants, found {len(participants)}")
    fit_rows = []
    for participant, rows in sorted(participants.items()):
        condition = rows[0]["xai"]
        trials = observed_trials(rows, case_map)
        family_results = []
        for family in ("feature contribution", "weighted examples"):
            params, insample, cv = fit_family(family, condition, training, trials)
            result = {
                "participant": participant, "xai": condition, "model family": family,
                **params,
                "k": "observed per instance",
                "in-sample combined loss": insample["loss"],
                "in-sample selection F1": insample["selection_f1"],
                "in-sample amount MAE": insample["amount_mae"],
                "in-sample direction accuracy": insample["direction_accuracy"],
                "5-fold CV combined loss": cv["loss"],
                "5-fold CV selection F1": cv["selection_f1"],
                "5-fold CV amount MAE": cv["amount_mae"],
                "5-fold CV direction accuracy": cv["direction_accuracy"],
            }
            family_results.append(result)
        winner = min(family_results, key=lambda row: row["5-fold CV combined loss"])["model family"]
        for result in family_results:
            result["selected family by CV"] = int(result["model family"] == winner)
            fit_rows.append(result)
    write_csv(OUT_FITS, fit_rows)

    common_ids = sorted(case_id for case_id in case_map if case_id < 140000)[:20]
    test_cases = [case_map[case_id] for case_id in common_ids]
    observed_k_by_case: dict[int, list[int]] = defaultdict(list)
    for rows in participants.values():
        for row in rows:
            if row["phase"] != "testing":
                continue
            case_id = int(float(row["instance id"]))
            observed_k_by_case[case_id].append(sum(int(float(row[f"x_{i}_changed"])) for i in range(1, 6)))
    recovery_k = []
    for case_id in common_ids:
        counts = Counter(observed_k_by_case[case_id])
        recovery_k.append(counts.most_common(1)[0][0] if counts else 2)
    rng = np.random.default_rng(20260828)
    recovery_rows = []
    for condition in ("none", "attribution", "counterfactual"):
        for family in ("feature contribution", "weighted examples"):
            recovery_rows.extend(recovery(training, test_cases, recovery_k, condition, family, rng))
        recovery_rows.extend(family_recovery(training, test_cases, recovery_k, condition, rng))
    for row in recovery_rows:
        row.setdefault("family exact recovery", "")
        row.setdefault("mean true-vs-other loss margin", "")
    write_csv(OUT_RECOVERY, recovery_rows)

    winners = [row for row in fit_rows if row["selected family by CV"] == 1]
    summary = {
        "participants": len(participants),
        "participants_by_condition": dict(Counter(rows[0]["xai"] for rows in participants.values())),
        "fit_definition": "0.5*(1-case-level selection F1) + 0.5*normalized amount MAE over the union of predicted and observed changed features",
        "model_winners_by_condition": {
            condition: dict(Counter(row["model family"] for row in winners if row["xai"] == condition))
            for condition in ("none", "attribution", "counterfactual")
        },
        "mean_cv_metrics": {},
        "best_parameter_counts": {},
        "files": {"fits": str(OUT_FITS), "recovery": str(OUT_RECOVERY)},
    }
    for condition in ("none", "attribution", "counterfactual"):
        for family in ("feature contribution", "weighted examples"):
            subset = [r for r in fit_rows if r["xai"] == condition and r["model family"] == family]
            key = f"{condition} | {family}"
            summary["mean_cv_metrics"][key] = {
                name: float(np.nanmean([float(r[name]) for r in subset]))
                for name in ("5-fold CV combined loss", "5-fold CV selection F1", "5-fold CV amount MAE", "5-fold CV direction accuracy")
            }
            summary["best_parameter_counts"][key] = {
                parameter: dict(Counter(str(r[parameter]) for r in subset))
                for parameter in ("eta", "alpha", "rho", "lambda", "beta")
                if any(math.isfinite(float(r[parameter])) for r in subset)
            }
    summary["k_definition"] = "Observed number of changed attributes for each participant-instance; not fitted or recovered"
    summary["recovery_k_by_case"] = dict(zip(map(str, common_ids), recovery_k))
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
