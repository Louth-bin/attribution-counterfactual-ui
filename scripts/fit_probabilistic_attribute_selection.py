"""Fit additive-rho models with probabilistic attribute-set selection.

K is observed per participant-instance.  The only new fitted quantity is a
binary indicator for whether Age is considered actionable.  Selection uses a
fixed-temperature distribution over all K-feature subsets, so no stochasticity
parameter is fitted.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_parsimonious_weighted_models import (
    ALPHA_GRID, BETA_GRID, BUNDLE, ETA_GRID, LAMBDA_GRID,
    case_metrics, label_sign, load_participants, normalize_case,
    observed_trials, training_representation, unit,
)


OUTPUT = ROOT / "qualtrics" / "parsimonious_probabilistic_model_fits_v1.3.csv"
SUMMARY = ROOT / "qualtrics" / "parsimonious_probabilistic_model_summary_v1.3.json"
EPS = 1e-12
AGE_INDEX = 4
# Rho is an additional normalized change applied in the model's proposed
# direction to every selected feature.  It is no longer a multiplier.
ADDITIVE_RHO_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parameter_grid(condition: str, family: str, actionability_options=(0, 1), immutable_index=AGE_INDEX):
    etas = (0.0,) if condition == "none" else ETA_GRID
    if family == "feature contribution":
        for eta, alpha, rho, age in itertools.product(etas, ALPHA_GRID, ADDITIVE_RHO_GRID, actionability_options):
            yield {"eta": eta, "alpha": alpha, "rho": rho, "lambda": math.nan, "beta": math.nan, "age actionable": age, "immutable index": immutable_index}
    else:
        betas = BETA_GRID if condition == "counterfactual" else (0.0,)
        for eta, rho, locality, beta, age in itertools.product(etas, ADDITIVE_RHO_GRID, LAMBDA_GRID, betas, actionability_options):
            yield {"eta": eta, "alpha": math.nan, "rho": rho, "lambda": locality, "beta": beta, "age actionable": age, "immutable index": immutable_index}


def selection_parameter_grid(
    condition: str,
    family: str,
    actionability_options=(0, 1),
    immutable_index=AGE_INDEX,
):
    """Generate feature-choice parameters once, without redundant rho copies."""
    etas = (0.0,) if condition == "none" else ETA_GRID
    if family == "feature contribution":
        for eta, alpha, age in itertools.product(
            etas, ALPHA_GRID, actionability_options
        ):
            yield {
                "eta": eta,
                "alpha": alpha,
                "rho": 0.0,
                "lambda": math.nan,
                "beta": math.nan,
                "age actionable": age,
                "immutable index": immutable_index,
            }
    else:
        betas = BETA_GRID if condition == "counterfactual" else (0.0,)
        for eta, locality, beta, age in itertools.product(
            etas, LAMBDA_GRID, betas, actionability_options
        ):
            yield {
                "eta": eta,
                "alpha": math.nan,
                "rho": 0.0,
                "lambda": locality,
                "beta": beta,
                "age actionable": age,
                "immutable index": immutable_index,
            }


def subset_probabilities(scores: np.ndarray, k: int, age_actionable: int, immutable_index=AGE_INDEX):
    eligible = list(range(5))
    if not age_actionable and immutable_index in eligible:
        eligible.remove(immutable_index)
    if k > len(eligible):
        return []
    if k == 0:
        return [(tuple(), 1.0)]
    values = scores[eligible]
    sd = float(np.std(values))
    standardized = (values - float(np.mean(values))) / (sd if sd > EPS else 1.0)
    z = dict(zip(eligible, standardized))
    subsets = list(itertools.combinations(eligible, k))
    logits = np.asarray([sum(z[f] for f in subset) for subset in subsets])
    weights = np.exp(logits - float(np.max(logits)))
    weights /= weights.sum()
    return list(zip(subsets, weights))


def contribution_components(x, target, rep, params):
    theta = rep["relevance"] * rep["polarity"]
    opposition = -target * theta * (x - rep["centre"])
    local = unit(opposition - float(np.min(opposition)))
    scores = (1.0 - params["alpha"]) * local + params["alpha"] * rep["relevance"]
    destination = np.where(target * theta >= 0, 1.0, 0.0)
    # Use the same local/global balance for selecting a feature and allocating
    # its share of the remaining target-supporting movement.
    direction = scores * (destination - x)
    base = float(target * np.dot(theta, x - rep["centre"]))
    return scores, theta, direction, base


def contribution_delta_for_subset(x, target, rep, params, subset, components=None):
    if components is None:
        components = contribution_components(x, target, rep, params)
    _, theta, direction, base = components
    selected = np.asarray(subset, dtype=int)
    gain = float(target * np.dot(theta[selected], direction[selected])) if len(selected) else 0.0
    if gain <= EPS:
        fraction = 0.0
    else:
        max_relevance = float(np.max(rep["relevance"][selected])) if len(selected) else 0.0
        max_fraction = 1.0 / max(max_relevance, EPS)
        fraction = np.clip(-base / gain, 0.0, max_fraction)
        if fraction <= EPS:
            fraction = 0.1
    delta = np.zeros(5)
    base_delta = fraction * direction[selected]
    delta[selected] = np.clip(
        base_delta + params["rho"] * np.sign(direction[selected]),
        -1.0,
        1.0,
    )
    return np.clip(x + delta, 0.0, 1.0) - x


def contribution_distribution(x, target, rep, params, k):
    components = contribution_components(x, target, rep, params)
    scores = components[0]
    proposals = []
    for subset, probability in subset_probabilities(scores, k, params["age actionable"], params.get("immutable index", AGE_INDEX)):
        delta = contribution_delta_for_subset(
            x, target, rep, params, subset, components=components
        )
        proposals.append((subset, float(probability), delta))
    return proposals


def memory_vector(x, target, rep, condition, params):
    relevance = rep["relevance"]
    if condition == "counterfactual":
        eligible = np.where(-rep["y"] == target)[0]
        sources = rep["x"][eligible]
        endpoints = rep["endpoints"][eligible]
        distances = np.sum(relevance * np.abs(sources - x), axis=1)
        weights = np.exp(-params["lambda"] * distances)
        weights /= weights.sum()
        endpoint_vector = weights @ endpoints - x
        amount_vector = weights @ (endpoints - sources)
        vector = (1.0 - params["beta"]) * endpoint_vector + params["beta"] * amount_vector
    else:
        eligible = np.where(rep["y"] == target)[0]
        examples = rep["x"][eligible]
        distances = np.sum(relevance * np.abs(examples - x), axis=1)
        weights = np.exp(-params["lambda"] * distances)
        weights /= weights.sum()
        vector = weights @ examples - x
    return vector


def memory_delta_for_subset(x, target, rep, condition, params, subset, vector=None):
    if vector is None:
        vector = memory_vector(x, target, rep, condition, params)
    delta = np.zeros(5)
    selected = np.asarray(subset, dtype=int)
    base_delta = vector[selected]
    delta[selected] = base_delta + params["rho"] * np.sign(base_delta)
    return np.clip(x + delta, 0.0, 1.0) - x


def memory_distribution(x, target, rep, condition, params, k):
    vector = memory_vector(x, target, rep, condition, params)
    proposals = []
    for subset, probability in subset_probabilities(np.abs(vector), k, params["age actionable"], params.get("immutable index", AGE_INDEX)):
        delta = memory_delta_for_subset(
            x, target, rep, condition, params, subset, vector=vector
        )
        proposals.append((subset, float(probability), delta))
    return proposals


def evaluate(family, condition, params, training, trials, indices=None):
    rep = training_representation(training, condition, params["eta"])
    selected_indices = range(len(trials)) if indices is None else indices
    totals = {"loss": [], "selection_f1": [], "amount_mae": [], "direction_accuracy": [], "selection_nll": []}
    for index in selected_indices:
        trial = trials[index]
        observed_set = tuple(np.where(np.abs(trial["observed"]) > 1e-8)[0])
        proposals = (
            contribution_distribution(trial["x"], trial["target"], rep, params, trial["observed_k"])
            if family == "feature contribution"
            else memory_distribution(trial["x"], trial["target"], rep, condition, params, trial["observed_k"])
        )
        if proposals:
            metrics = [(probability, case_metrics(delta, trial["observed"])) for _, probability, delta in proposals]
            # Attribute-selection quality remains an expectation over the model's
            # distribution of possible K-feature subsets.
            expected_f1 = sum(p * values[1] for p, values in metrics)
            probability_observed = sum(
                probability for subset, probability, _ in proposals
                if subset == observed_set
            )
        else:
            expected_f1 = 0.0
            probability_observed = 0.0
        totals["selection_f1"].append(expected_f1)

        # Fit rho after forcing the attributes actually selected by the
        # participant.  This is deliberately independent of the model's feature
        # selection eligibility/probability (including the Age actionability
        # parameter): assuming the selected features match, how well does rho
        # recover the participant's edit amounts?
        observed_features = np.asarray(observed_set, dtype=int)
        conditional_delta = (
            contribution_delta_for_subset(
                trial["x"], trial["target"], rep, params, observed_set
            )
            if family == "feature contribution"
            else memory_delta_for_subset(
                trial["x"], trial["target"], rep, condition, params, observed_set
            )
        )
        if len(observed_features):
            conditional_amount = float(np.mean(np.abs(
                conditional_delta[observed_features] - trial["observed"][observed_features]
            )))
            conditional_direction = float(np.mean(
                np.sign(conditional_delta[observed_features])
                == np.sign(trial["observed"][observed_features])
            ))
        else:
            conditional_amount = 0.0
            conditional_direction = math.nan
        totals["amount_mae"].append(conditional_amount)
        totals["direction_accuracy"].append(conditional_direction)
        totals["loss"].append(0.5 * (1.0 - expected_f1) + 0.5 * conditional_amount)
        totals["selection_nll"].append(-math.log(max(probability_observed, EPS)))
    result = {}
    for name, values in totals.items():
        finite = [value for value in values if math.isfinite(value)]
        result[name] = float(np.mean(finite)) if finite else math.nan
    return result


def select_parameters(scored):
    """Select feature-choice parameters first, then fit rho conditionally."""
    best_selection_nll = min(metrics["selection_nll"] for metrics, _ in scored)
    selection_equivalent = [
        item for item in scored
        if math.isclose(
            item[0]["selection_nll"], best_selection_nll,
            rel_tol=1e-12, abs_tol=1e-12,
        )
    ]
    return min(selection_equivalent, key=lambda item: item[0]["amount_mae"])


def fit_indices(family, condition, training, trials, candidates, indices=None):
    """Fit selection first, then search rho only for selection-optimal fits."""
    selection_scored = [
        (evaluate(family, condition, params, training, trials, indices), params)
        for params in candidates
    ]
    best_selection_nll = min(
        metrics["selection_nll"] for metrics, _ in selection_scored
    )
    selection_equivalent = [
        params
        for metrics, params in selection_scored
        if math.isclose(
            metrics["selection_nll"], best_selection_nll,
            rel_tol=1e-12, abs_tol=1e-12,
        )
    ]
    conditional_scored = []
    for selection_params in selection_equivalent:
        for rho in ADDITIVE_RHO_GRID:
            params = {**selection_params, "rho": rho}
            conditional_scored.append(
                (evaluate(family, condition, params, training, trials, indices), params)
            )
    return min(conditional_scored, key=lambda item: item[0]["amount_mae"])


def fit_family(family, condition, training, trials, actionability_options=(0, 1), immutable_index=AGE_INDEX):
    candidates = list(
        selection_parameter_grid(
            condition, family, actionability_options, immutable_index
        )
    )
    full_metrics, best = fit_indices(
        family, condition, training, trials, candidates
    )
    folds = []
    fold_count = min(5, len(trials))
    for fold in range(fold_count):
        train_idx = [i for i in range(len(trials)) if i % fold_count != fold]
        test_idx = [i for i in range(len(trials)) if i % fold_count == fold]
        _, chosen = fit_indices(
            family, condition, training, trials, candidates, train_idx
        )
        folds.append(evaluate(family, condition, chosen, training, trials, test_idx))
    cv = {name: float(np.nanmean([fold[name] for fold in folds])) for name in full_metrics}
    return best, full_metrics, cv


def main() -> None:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    training = bundle["training_pool"]
    case_map = {int(case["instance_id"]): case for case in bundle["test_pool"]}
    participants = load_participants()
    rows = []
    for participant, participant_rows in sorted(participants.items()):
        condition = participant_rows[0]["xai"]
        trials = observed_trials(participant_rows, case_map)
        fitted = []
        for family in ("feature contribution", "weighted examples"):
            params, insample, cv = fit_family(family, condition, training, trials)
            row = {"participant": participant, "xai": condition, "model family": family, "k": "observed per instance", **params}
            for name, value in insample.items(): row[f"in-sample {name}"] = value
            for name, value in cv.items(): row[f"5-fold CV {name}"] = value
            fitted.append(row)
        winner = min(fitted, key=lambda row: (row["5-fold CV selection_nll"], row["5-fold CV amount_mae"]))["model family"]
        for row in fitted:
            row["selected family by CV"] = int(row["model family"] == winner)
            rows.append(row)
    write_csv(OUTPUT, rows)
    winners = [row for row in rows if row["selected family by CV"]]
    summary = {
        "participants": len(participants),
        "selection_distribution": "P(S|x,K) proportional to exp(sum of standardized feature scores in S), over all K-subsets; fixed temperature 1",
        "amount_fitting": (
            "rho is an additive normalized margin per selected feature and is fitted "
            "to amount MAE conditional on the participant's observed feature subset"
        ),
        "k": "observed per participant-instance",
        "age_actionable": "binary fitted parameter; 0 removes Age from eligible subsets",
        "winners": {condition: dict(Counter(row["model family"] for row in winners if row["xai"] == condition)) for condition in ("none", "attribution", "counterfactual")},
        "winner_metrics": {
            condition: {name: float(np.mean([row[name] for row in winners if row["xai"] == condition])) for name in ("5-fold CV selection_f1", "5-fold CV amount_mae", "5-fold CV direction_accuracy", "5-fold CV loss", "5-fold CV selection_nll")}
            for condition in ("none", "attribution", "counterfactual")
        },
        "age_actionable_counts": {condition: dict(Counter(str(row["age actionable"]) for row in winners if row["xai"] == condition)) for condition in ("none", "attribution", "counterfactual")},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
