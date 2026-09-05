"""Current probabilistic additive-rho baseline, frozen as version 0.1."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np


FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
MODEL_FAMILIES = ("feature contribution", "weighted examples")
ETA_GRID = (0.0, 0.5, 1.0)
ALPHA_GRID = (0.0, 0.5, 1.0)
RHO_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
LAMBDA_GRID = (0.0, 2.0, 6.0)
BETA_GRID = (0.0, 0.5, 1.0)
AGE_INDEX = 4
EPSILON = 1e-12


def normalize_values(values: list[float], case: dict) -> np.ndarray:
    normalized = []
    for value, bounds in zip(values, case["feature_ranges"]):
        try:
            low, high = map(float, bounds)
            result = (float(value) - low) / (high - low)
        except (TypeError, ValueError):
            categories = [str(item).strip().lower() for item in bounds]
            result = categories.index(str(value).strip().lower()) / max(
                len(categories) - 1, 1
            )
        normalized.append(np.clip(result, 0.0, 1.0))
    return np.asarray(normalized, dtype=float)


def normalize_case(case: dict) -> np.ndarray:
    return normalize_values(case["feature_values"], case)


def label_sign(label: str) -> int:
    positive = {"diabetes", "expensive", "below limit"}
    return 1 if str(label).strip().lower() in positive else -1


def unit(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    return values / total if total > EPSILON else np.full(len(values), 1 / len(values))


def build_memory(cases: list[dict], condition: str, eta: float) -> dict:
    """Build the v0.1 batch memory from every displayed training case."""
    profiles = np.vstack([normalize_case(case) for case in cases])
    labels = np.asarray([label_sign(case["prediction"]["label"]) for case in cases])
    positive, negative = profiles[labels == 1], profiles[labels == -1]
    positive_mean, negative_mean = positive.mean(axis=0), negative.mean(axis=0)
    centre = (positive_mean + negative_mean) / 2
    diagnosticity = unit(
        np.abs(positive_mean - negative_mean)
        / (positive.std(axis=0) + negative.std(axis=0) + EPSILON)
    )
    polarity = np.sign(positive_mean - negative_mean)
    polarity[polarity == 0] = 1

    if condition == "attribution":
        explanation = unit(np.mean([
            np.abs(np.asarray(case["attribution"]["values"], dtype=float))
            for case in cases
        ], axis=0))
    elif condition == "counterfactual":
        explanation = unit(np.mean([
            np.abs(
                normalize_values(case["counterfactual"]["feature_values"], case)
                - normalize_case(case)
            )
            for case in cases
        ], axis=0))
    else:
        explanation = diagnosticity
        eta = 0.0

    return {
        "profiles": profiles,
        "labels": labels,
        "centre": centre,
        "polarity": polarity,
        "relevance": unit((1 - eta) * diagnosticity + eta * explanation),
        "endpoints": np.vstack([
            normalize_values(case["counterfactual"]["feature_values"], case)
            for case in cases
        ]),
    }


def subset_distribution(
    scores: np.ndarray,
    k: int,
    age_actionable: int,
    immutable_index: int = AGE_INDEX,
) -> list[tuple[tuple[int, ...], float]]:
    """Return P(S|x,K) over all eligible K-feature subsets."""
    eligible = list(range(len(FEATURES)))
    if not age_actionable and immutable_index in eligible:
        eligible.remove(immutable_index)
    if k > len(eligible):
        return []
    if k == 0:
        return [(tuple(), 1.0)]

    values = scores[eligible]
    standard_deviation = float(np.std(values))
    standardized = (values - float(np.mean(values))) / (
        standard_deviation if standard_deviation > EPSILON else 1.0
    )
    score_by_feature = dict(zip(eligible, standardized))
    subsets = list(itertools.combinations(eligible, k))
    logits = np.asarray([
        sum(score_by_feature[feature] for feature in subset)
        for subset in subsets
    ])
    probabilities = np.exp(logits - float(np.max(logits)))
    probabilities /= probabilities.sum()
    return list(zip(subsets, probabilities.astype(float)))


def contribution_components(profile, target, memory, parameters):
    weights = memory["relevance"] * memory["polarity"]
    opposition = -target * weights * (profile - memory["centre"])
    local_attention = unit(opposition - float(np.min(opposition)))
    scores = (
        (1 - parameters["alpha"]) * local_attention
        + parameters["alpha"] * memory["relevance"]
    )
    destination = np.where(target * weights >= 0, 1.0, 0.0)
    # The same balance of local opposition and global relevance controls both
    # feature selection and allocation of the remaining feasible movement.
    direction = scores * (destination - profile)
    boundary_evidence = float(target * np.dot(weights, profile - memory["centre"]))
    return scores, weights, direction, boundary_evidence


def contribution_delta(
    profile,
    target,
    memory,
    parameters,
    subset,
    components=None,
):
    if components is None:
        components = contribution_components(profile, target, memory, parameters)
    _, weights, direction, boundary_evidence = components
    selected = np.asarray(subset, dtype=int)
    gain = (
        float(target * np.dot(weights[selected], direction[selected]))
        if len(selected)
        else 0.0
    )
    if gain <= EPSILON:
        fraction = 0.0
    else:
        max_relevance = (
            float(np.max(memory["relevance"][selected])) if len(selected) else 0.0
        )
        fraction = np.clip(
            -boundary_evidence / gain,
            0.0,
            1.0 / max(max_relevance, EPSILON),
        )
        if fraction <= EPSILON:
            fraction = 0.1

    delta = np.zeros(len(FEATURES))
    base_delta = fraction * direction[selected]
    delta[selected] = np.clip(
        base_delta + parameters["rho"] * np.sign(direction[selected]),
        -1.0,
        1.0,
    )
    return np.clip(profile + delta, 0.0, 1.0) - profile


def contribution_distribution(profile, target, memory, parameters, k):
    components = contribution_components(profile, target, memory, parameters)
    return [
        (
            subset,
            probability,
            contribution_delta(
                profile, target, memory, parameters, subset, components
            ),
        )
        for subset, probability in subset_distribution(
            components[0],
            k,
            parameters["age actionable"],
            parameters.get("immutable index", AGE_INDEX),
        )
    ]


def exemplar_vector(profile, target, memory, condition, parameters):
    relevance = memory["relevance"]
    if condition == "counterfactual":
        eligible = np.where(-memory["labels"] == target)[0]
        sources = memory["profiles"][eligible]
        endpoints = memory["endpoints"][eligible]
        distances = np.sum(relevance * np.abs(sources - profile), axis=1)
        retrieval = np.exp(-parameters["lambda"] * distances)
        retrieval /= retrieval.sum()
        endpoint_vector = retrieval @ endpoints - profile
        remembered_change = retrieval @ (endpoints - sources)
        return (
            (1 - parameters["beta"]) * endpoint_vector
            + parameters["beta"] * remembered_change
        )

    eligible = np.where(memory["labels"] == target)[0]
    examples = memory["profiles"][eligible]
    distances = np.sum(relevance * np.abs(examples - profile), axis=1)
    retrieval = np.exp(-parameters["lambda"] * distances)
    retrieval /= retrieval.sum()
    return retrieval @ examples - profile


def exemplar_delta(
    profile,
    target,
    memory,
    condition,
    parameters,
    subset,
    vector=None,
):
    if vector is None:
        vector = exemplar_vector(profile, target, memory, condition, parameters)
    selected = np.asarray(subset, dtype=int)
    delta = np.zeros(len(FEATURES))
    base_delta = vector[selected]
    delta[selected] = base_delta + parameters["rho"] * np.sign(base_delta)
    return np.clip(profile + delta, 0.0, 1.0) - profile


def exemplar_distribution(profile, target, memory, condition, parameters, k):
    vector = exemplar_vector(profile, target, memory, condition, parameters)
    return [
        (
            subset,
            probability,
            exemplar_delta(
                profile, target, memory, condition, parameters, subset, vector
            ),
        )
        for subset, probability in subset_distribution(
            np.abs(vector),
            k,
            parameters["age actionable"],
            parameters.get("immutable index", AGE_INDEX),
        )
    ]


def selection_f1(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted_set = set(np.where(np.abs(predicted) > 1e-8)[0])
    observed_set = set(np.where(np.abs(observed) > 1e-8)[0])
    if not predicted_set and not observed_set:
        return 1.0
    if not predicted_set or not observed_set:
        return 0.0
    return 2 * len(predicted_set & observed_set) / (
        len(predicted_set) + len(observed_set)
    )


def evaluate(
    family: str,
    condition: str,
    parameters: dict,
    training: list[dict],
    trials: list[dict],
    indices: Iterable[int] | None = None,
) -> dict[str, float]:
    """Score selection probabilistically and rho conditional on observed features."""
    memory = build_memory(training, condition, parameters["eta"])
    selected_indices = range(len(trials)) if indices is None else indices
    totals = {
        "loss": [],
        "selection_f1": [],
        "amount_mae": [],
        "direction_accuracy": [],
        "selection_nll": [],
    }
    for index in selected_indices:
        trial = trials[index]
        observed = trial["observed"]
        observed_set = tuple(np.where(np.abs(observed) > 1e-8)[0])
        if family == "feature contribution":
            proposals = contribution_distribution(
                trial["profile"],
                trial["target"],
                memory,
                parameters,
                trial["observed_k"],
            )
            conditional_delta = contribution_delta(
                trial["profile"],
                trial["target"],
                memory,
                parameters,
                observed_set,
            )
        else:
            proposals = exemplar_distribution(
                trial["profile"],
                trial["target"],
                memory,
                condition,
                parameters,
                trial["observed_k"],
            )
            conditional_delta = exemplar_delta(
                trial["profile"],
                trial["target"],
                memory,
                condition,
                parameters,
                observed_set,
            )

        expected_f1 = sum(
            probability * selection_f1(delta, observed)
            for _, probability, delta in proposals
        )
        observed_probability = sum(
            probability
            for subset, probability, _ in proposals
            if subset == observed_set
        )
        selected = np.asarray(observed_set, dtype=int)
        if len(selected):
            amount_mae = float(np.mean(np.abs(
                conditional_delta[selected] - observed[selected]
            )))
            direction_accuracy = float(np.mean(
                np.sign(conditional_delta[selected]) == np.sign(observed[selected])
            ))
        else:
            amount_mae = 0.0
            direction_accuracy = math.nan

        totals["selection_f1"].append(expected_f1)
        totals["amount_mae"].append(amount_mae)
        totals["direction_accuracy"].append(direction_accuracy)
        totals["loss"].append(0.5 * (1 - expected_f1) + 0.5 * amount_mae)
        totals["selection_nll"].append(
            -math.log(max(observed_probability, EPSILON))
        )

    return {
        name: float(np.mean([value for value in values if math.isfinite(value)]))
        if any(math.isfinite(value) for value in values)
        else math.nan
        for name, values in totals.items()
    }


def selection_parameter_grid(
    condition: str,
    family: str,
    actionability_options: tuple[int, ...] = (0, 1),
    immutable_index: int = AGE_INDEX,
) -> Iterable[dict]:
    eta_values = (0.0,) if condition == "none" else ETA_GRID
    if family == "feature contribution":
        combinations = itertools.product(
            eta_values, ALPHA_GRID, actionability_options
        )
        for eta, alpha, age_actionable in combinations:
            yield {
                "eta": eta,
                "alpha": alpha,
                "rho": 0.0,
                "lambda": math.nan,
                "beta": math.nan,
                "age actionable": age_actionable,
                "immutable index": immutable_index,
            }
    else:
        beta_values = BETA_GRID if condition == "counterfactual" else (0.0,)
        combinations = itertools.product(
            eta_values, LAMBDA_GRID, beta_values, actionability_options
        )
        for eta, locality, beta, age_actionable in combinations:
            yield {
                "eta": eta,
                "alpha": math.nan,
                "rho": 0.0,
                "lambda": locality,
                "beta": beta,
                "age actionable": age_actionable,
                "immutable index": immutable_index,
            }


def fit_indices(
    family: str,
    condition: str,
    training: list[dict],
    trials: list[dict],
    candidates: list[dict],
    indices: Iterable[int] | None = None,
) -> tuple[dict[str, float], dict]:
    selection_scores = [
        (evaluate(family, condition, candidate, training, trials, indices), candidate)
        for candidate in candidates
    ]
    best_nll = min(score["selection_nll"] for score, _ in selection_scores)
    selection_winners = [
        candidate
        for score, candidate in selection_scores
        if math.isclose(
            score["selection_nll"],
            best_nll,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ]
    amount_scores = []
    for selection_parameters in selection_winners:
        for rho in RHO_GRID:
            parameters = {**selection_parameters, "rho": rho}
            amount_scores.append((
                evaluate(family, condition, parameters, training, trials, indices),
                parameters,
            ))
    return min(amount_scores, key=lambda item: item[0]["amount_mae"])


def fit_family(
    family: str,
    condition: str,
    training: list[dict],
    trials: list[dict],
    immutable_index: int = AGE_INDEX,
    actionability_options: tuple[int, ...] = (0, 1),
) -> tuple[dict, dict[str, float], dict[str, float]]:
    """Return fitted parameters, in-sample metrics, and five-fold CV metrics."""
    candidates = list(selection_parameter_grid(
        condition,
        family,
        actionability_options=actionability_options,
        immutable_index=immutable_index,
    ))
    in_sample, parameters = fit_indices(
        family, condition, training, trials, candidates
    )
    fold_count = min(5, len(trials))
    folds = []
    for fold in range(fold_count):
        train_indices = [
            index for index in range(len(trials)) if index % fold_count != fold
        ]
        test_indices = [
            index for index in range(len(trials)) if index % fold_count == fold
        ]
        _, fold_parameters = fit_indices(
            family,
            condition,
            training,
            trials,
            candidates,
            train_indices,
        )
        folds.append(evaluate(
            family,
            condition,
            fold_parameters,
            training,
            trials,
            test_indices,
        ))
    cross_validated = {
        name: float(np.nanmean([fold[name] for fold in folds]))
        for name in in_sample
    }
    return parameters, in_sample, cross_validated


def observed_trials(
    rows: list[dict[str, str]],
    case_map: dict[int, dict],
) -> list[dict]:
    trials = []
    testing_rows = sorted(
        (row for row in rows if row["phase"] == "testing"),
        key=lambda row: int(float(row.get("case") or row["trial number"])),
    )
    for row in testing_rows:
        case = case_map[int(float(row["instance id"]))]
        observed = np.asarray([
            float(row[f"x_{index}_change"]) for index in range(1, 6)
        ])
        trials.append({
            "case": case,
            "profile": normalize_case(case),
            "target": label_sign(row["target label"]),
            "observed": observed,
            "observed_k": int(np.sum(np.abs(observed) > 1e-8)),
        })
    return trials
