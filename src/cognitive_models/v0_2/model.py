"""Sequential full-state cognitive model with optional memory variation."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

import numpy as np


FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
MODEL_FAMILIES = ("feature contribution", "weighted examples")
MEMORY_VARIANTS = ("perfect memory", "memory variation")
ATTENTION_RATE_GRID = (0.25, 0.5, 1.0)
XAI_WEIGHT_GRID = (0.0, 0.5, 1.0)
ALPHA_GRID = (0.0, 0.5, 1.0)
RHO_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
SPECIFICITY_GRID = (0.0, 2.0, 6.0)
BETA_GRID = (0.0, 0.5, 1.0)
DECAY_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
AGE_INDEX = 4
MEMORY_MATCH_SCALE = 1.0
EPSILON = 1e-12


@dataclass(frozen=True)
class MentalState:
    """The participant's complete derived state after one training trial."""

    encoded_at: int
    source_profile: np.ndarray
    source_label: int
    endpoint: np.ndarray
    negative_mean: np.ndarray
    positive_mean: np.ndarray
    attention_logits: np.ndarray

    @property
    def attention(self) -> np.ndarray:
        return softmax(self.attention_logits)

    @property
    def centre(self) -> np.ndarray:
        return (self.negative_mean + self.positive_mean) / 2

    @property
    def polarity(self) -> np.ndarray:
        return np.sign(self.positive_mean - self.negative_mean)


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


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    values = np.exp(logits - float(np.max(logits)))
    return values / values.sum()


def explanation_attention(case: dict, condition: str) -> np.ndarray | None:
    if condition == "attribution":
        return unit(np.abs(np.asarray(case["attribution"]["values"], dtype=float)))
    if condition == "counterfactual":
        return unit(np.abs(
            normalize_values(case["counterfactual"]["feature_values"], case)
            - normalize_case(case)
        ))
    return None


def alcove_attention_gradient(
    profile: np.ndarray,
    label: int,
    previous_profiles: list[np.ndarray],
    previous_labels: list[int],
    attention_logits: np.ndarray,
    specificity: float,
) -> np.ndarray:
    """Gradient of exemplar-category cross entropy through ALCOVE attention."""
    if not previous_profiles:
        return np.zeros(len(FEATURES))

    attention = softmax(attention_logits)
    distances = np.abs(np.vstack(previous_profiles) - profile)
    similarities = np.exp(-specificity * (distances @ attention))
    labels = np.asarray(previous_labels)
    target_mask = labels == label
    target_evidence = 1.0 + float(similarities[target_mask].sum())
    total_evidence = 2.0 + float(similarities.sum())
    target_distance = (
        (similarities[target_mask, None] * distances[target_mask]).sum(axis=0)
        / target_evidence
    )
    total_distance = (
        (similarities[:, None] * distances).sum(axis=0)
        / total_evidence
    )
    gradient_attention = specificity * (target_distance - total_distance)
    return attention * (
        gradient_attention - float(np.dot(attention, gradient_attention))
    )


def build_states(
    cases: list[dict],
    condition: str,
    attention_rate: float,
    xai_weight: float,
    specificity: float,
) -> list[MentalState]:
    """Learn sequentially and store the complete state after every exposure."""
    states = []
    profiles: list[np.ndarray] = []
    labels: list[int] = []
    attention_logits = np.zeros(len(FEATURES))

    for index, case in enumerate(cases, start=1):
        profile = normalize_case(case)
        label = label_sign(case["prediction"]["label"])
        gradient = alcove_attention_gradient(
            profile, label, profiles, labels, attention_logits, specificity
        )
        explanation = explanation_attention(case, condition)
        if explanation is not None:
            gradient += xai_weight * (softmax(attention_logits) - explanation)
        attention_logits = attention_logits - attention_rate * gradient

        profiles.append(profile)
        labels.append(label)
        matrix = np.vstack(profiles)
        label_array = np.asarray(labels)
        positive = matrix[label_array == 1]
        negative = matrix[label_array == -1]
        positive_mean = (
            positive.mean(axis=0) if len(positive) else np.full(len(FEATURES), 0.5)
        )
        negative_mean = (
            negative.mean(axis=0) if len(negative) else np.full(len(FEATURES), 0.5)
        )
        states.append(MentalState(
            encoded_at=index,
            source_profile=profile,
            source_label=label,
            endpoint=normalize_values(case["counterfactual"]["feature_values"], case),
            negative_mean=negative_mean,
            positive_mean=positive_mean,
            attention_logits=attention_logits.copy(),
        ))
    return states


def subset_distribution(
    scores: np.ndarray,
    k: int,
    age_actionable: int,
    immutable_index: int = AGE_INDEX,
) -> list[tuple[tuple[int, ...], float]]:
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
    probabilities = softmax(np.asarray([
        sum(score_by_feature[feature] for feature in subset)
        for subset in subsets
    ]))
    return list(zip(subsets, probabilities.astype(float)))


def state_distribution(
    profile: np.ndarray,
    states: list[MentalState],
    parameters: dict,
    test_position: int,
) -> list[tuple[MentalState, float]]:
    if parameters["memory variant"] == "perfect memory":
        return [(states[-1], 1.0)]

    current_time = len(states) + test_position + 1
    logits = []
    for state in states:
        age = max(current_time - state.encoded_at, 1)
        mismatch = float(np.sum(
            state.attention * np.abs(profile - state.source_profile)
        ))
        logits.append(
            -parameters["memory decay"] * math.log(age)
            - MEMORY_MATCH_SCALE * mismatch
        )
    probabilities = softmax(np.asarray(logits))
    return list(zip(states, probabilities.astype(float)))


def blend_states(memories: list[tuple[MentalState, float]]) -> MentalState:
    """Retrieve one complete mental state as an activation-weighted blend."""
    states = [state for state, _ in memories]
    return MentalState(
        encoded_at=max(state.encoded_at for state in states),
        source_profile=sum(
            probability * state.source_profile
            for state, probability in memories
        ),
        source_label=states[-1].source_label,
        endpoint=sum(
            probability * state.endpoint
            for state, probability in memories
        ),
        negative_mean=sum(
            probability * state.negative_mean
            for state, probability in memories
        ),
        positive_mean=sum(
            probability * state.positive_mean
            for state, probability in memories
        ),
        attention_logits=sum(
            probability * state.attention_logits
            for state, probability in memories
        ),
    )


def exemplar_distribution_over_memory(
    profile: np.ndarray,
    target: int,
    states: list[MentalState],
    condition: str,
    parameters: dict,
    test_position: int,
) -> list[tuple[MentalState, float]]:
    if condition == "counterfactual":
        eligible = [state for state in states if -state.source_label == target]
    else:
        eligible = [state for state in states if state.source_label == target]
    if not eligible:
        return []

    current_time = len(states) + test_position + 1
    final_attention = states[-1].attention
    logits = []
    for state in eligible:
        attention = (
            final_attention
            if parameters["memory variant"] == "perfect memory"
            else state.attention
        )
        distance = float(np.sum(attention * np.abs(state.source_profile - profile)))
        recency = 0.0
        if parameters["memory variant"] == "memory variation":
            age = max(current_time - state.encoded_at, 1)
            recency = parameters["memory decay"] * math.log(age)
        logits.append(-parameters["specificity"] * distance - recency)
    probabilities = softmax(np.asarray(logits))
    return list(zip(eligible, probabilities.astype(float)))


def contribution_components(profile, target, state, parameters):
    weights = state.attention * state.polarity
    opposition = -target * weights * (profile - state.centre)
    local_attention = unit(opposition - float(np.min(opposition)))
    scores = (
        (1 - parameters["alpha"]) * local_attention
        + parameters["alpha"] * state.attention
    )
    signed_weights = target * weights
    destination = np.where(
        signed_weights > 0,
        1.0,
        np.where(signed_weights < 0, 0.0, profile),
    )
    direction = scores * (destination - profile)
    boundary_evidence = float(target * np.dot(weights, profile - state.centre))
    return scores, weights, direction, boundary_evidence


def contribution_delta(profile, target, state, parameters, subset, components=None):
    if components is None:
        components = contribution_components(profile, target, state, parameters)
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
        max_attention = float(np.max(state.attention[selected])) if len(selected) else 0.0
        fraction = np.clip(
            -boundary_evidence / gain,
            0.0,
            1.0 / max(max_attention, EPSILON),
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


def exemplar_vector(profile, state, condition, parameters):
    if condition == "counterfactual":
        endpoint_vector = state.endpoint - profile
        remembered_change = state.endpoint - state.source_profile
        return (
            (1 - parameters["beta"]) * endpoint_vector
            + parameters["beta"] * remembered_change
        )
    return state.source_profile - profile


def exemplar_delta(profile, parameters, subset, vector):
    selected = np.asarray(subset, dtype=int)
    delta = np.zeros(len(FEATURES))
    base_delta = vector[selected]
    delta[selected] = base_delta + parameters["rho"] * np.sign(base_delta)
    return np.clip(profile + delta, 0.0, 1.0) - profile


def proposal_distribution(
    family: str,
    condition: str,
    profile: np.ndarray,
    target: int,
    states: list[MentalState],
    parameters: dict,
    k: int,
    test_position: int,
) -> tuple[list[tuple[tuple[int, ...], float, np.ndarray]], MentalState | np.ndarray | None]:
    proposals = []
    if family == "feature contribution":
        memories = state_distribution(profile, states, parameters, test_position)
        state = blend_states(memories)
        components = contribution_components(profile, target, state, parameters)
        for subset, subset_probability in subset_distribution(
            components[0], k, parameters["age actionable"]
        ):
            proposals.append((
                subset,
                subset_probability,
                contribution_delta(
                    profile, target, state, parameters, subset, components
                ),
            ))
        conditional_source = state
    else:
        memories = exemplar_distribution_over_memory(
            profile, target, states, condition, parameters, test_position
        )
        if not memories:
            return [], None
        vector = sum(
            memory_probability * exemplar_vector(
                profile, state, condition, parameters
            )
            for state, memory_probability in memories
        )
        for subset, subset_probability in subset_distribution(
            np.abs(vector), k, parameters["age actionable"]
        ):
            proposals.append((
                subset,
                subset_probability,
                exemplar_delta(profile, parameters, subset, vector),
            ))
        conditional_source = vector
    return proposals, conditional_source


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
    state_cache: dict[tuple[float, float, float], list[MentalState]] | None = None,
    evaluation_cache: dict | None = None,
) -> dict[str, float]:
    if evaluation_cache is not None:
        parameter_key = tuple(sorted(
            (
                name,
                "nan" if isinstance(value, float) and math.isnan(value) else value,
            )
            for name, value in parameters.items()
        ))
        cache_key = (family, condition, parameter_key)
        if cache_key not in evaluation_cache:
            evaluation_cache[cache_key] = [
                evaluate(
                    family,
                    condition,
                    parameters,
                    training,
                    trials,
                    [index],
                    state_cache,
                    None,
                )
                for index in range(len(trials))
            ]
        chosen = (
            range(len(trials))
            if indices is None
            else list(indices)
        )
        return {
            name: float(np.nanmean([
                evaluation_cache[cache_key][index][name]
                for index in chosen
            ]))
            for name in evaluation_cache[cache_key][0]
        }

    states = (
        state_cache[(
            parameters["attention rate"],
            parameters["xai weight"],
            parameters["specificity"],
        )]
        if state_cache is not None
        else build_states(
            training,
            condition,
            parameters["attention rate"],
            parameters["xai weight"],
            parameters["specificity"],
        )
    )
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
        proposals, conditional_source = proposal_distribution(
            family,
            condition,
            trial["profile"],
            trial["target"],
            states,
            parameters,
            trial["observed_k"],
            index,
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
        amount_mae = 0.0
        direction_accuracy = 0.0
        if conditional_source is not None and len(selected):
            if family == "feature contribution":
                conditional_delta = contribution_delta(
                    trial["profile"],
                    trial["target"],
                    conditional_source,
                    parameters,
                    observed_set,
                )
            else:
                conditional_delta = exemplar_delta(
                    trial["profile"], parameters, observed_set, conditional_source
                )
            amount_mae = float(np.mean(np.abs(
                conditional_delta[selected] - observed[selected]
            )))
            direction_accuracy = float(np.mean(
                np.sign(conditional_delta[selected]) == np.sign(observed[selected])
            ))
        elif len(selected):
            amount_mae = 1.0
            direction_accuracy = 0.0
        else:
            direction_accuracy = math.nan

        totals["selection_f1"].append(expected_f1)
        totals["amount_mae"].append(amount_mae)
        totals["direction_accuracy"].append(direction_accuracy)
        totals["loss"].append(0.5 * (1 - expected_f1) + 0.5 * amount_mae)
        totals["selection_nll"].append(-math.log(max(observed_probability, EPSILON)))

    return {
        name: float(np.mean([value for value in values if math.isfinite(value)]))
        if any(math.isfinite(value) for value in values)
        else math.nan
        for name, values in totals.items()
    }


def selection_parameter_grid(
    condition: str,
    family: str,
    memory_variant: str,
    actionability_options: tuple[int, ...] = (0, 1),
) -> Iterable[dict]:
    rate_values = ATTENTION_RATE_GRID
    xai_values = (0.0,) if condition == "none" else XAI_WEIGHT_GRID
    decay_values = (math.nan,) if memory_variant == "perfect memory" else DECAY_GRID
    if family == "feature contribution":
        combinations = itertools.product(
            rate_values,
            xai_values,
            SPECIFICITY_GRID,
            ALPHA_GRID,
            decay_values,
            actionability_options,
        )
        for rate, xai_weight, specificity, alpha, decay, age_actionable in combinations:
            yield {
                "memory variant": memory_variant,
                "attention rate": rate,
                "xai weight": xai_weight,
                "specificity": specificity,
                "alpha": alpha,
                "rho": 0.0,
                "beta": math.nan,
                "memory decay": decay,
                "age actionable": age_actionable,
            }
    else:
        beta_values = BETA_GRID if condition == "counterfactual" else (0.0,)
        combinations = itertools.product(
            rate_values,
            xai_values,
            SPECIFICITY_GRID,
            beta_values,
            decay_values,
            actionability_options,
        )
        for rate, xai_weight, specificity, beta, decay, age_actionable in combinations:
            yield {
                "memory variant": memory_variant,
                "attention rate": rate,
                "xai weight": xai_weight,
                "specificity": specificity,
                "alpha": math.nan,
                "rho": 0.0,
                "beta": beta,
                "memory decay": decay,
                "age actionable": age_actionable,
            }


def fit_indices(
    family: str,
    condition: str,
    training: list[dict],
    trials: list[dict],
    candidates: list[dict],
    indices: Iterable[int] | None = None,
    state_cache: dict[tuple[float, float, float], list[MentalState]] | None = None,
    evaluation_cache: dict | None = None,
) -> tuple[dict[str, float], dict]:
    selection_scores = [
        (
            evaluate(
                family,
                condition,
                candidate,
                training,
                trials,
                indices,
                state_cache,
                evaluation_cache,
            ),
            candidate,
        )
        for candidate in candidates
    ]
    best_nll = min(score["selection_nll"] for score, _ in selection_scores)
    selection_winners = [
        candidate
        for score, candidate in selection_scores
        if math.isclose(score["selection_nll"], best_nll, rel_tol=1e-12, abs_tol=1e-12)
    ]
    amount_scores = []
    for selection_parameters in selection_winners:
        for rho in RHO_GRID:
            parameters = {**selection_parameters, "rho": rho}
            amount_scores.append((
                evaluate(
                    family,
                    condition,
                    parameters,
                    training,
                    trials,
                    indices,
                    state_cache,
                    evaluation_cache,
                ),
                parameters,
            ))
    return min(amount_scores, key=lambda item: item[0]["amount_mae"])


def fit_family(
    family: str,
    condition: str,
    memory_variant: str,
    training: list[dict],
    trials: list[dict],
) -> tuple[dict, dict[str, float], dict[str, float]]:
    candidates = list(selection_parameter_grid(condition, family, memory_variant))
    learning_values = sorted({
        (
            candidate["attention rate"],
            candidate["xai weight"],
            candidate["specificity"],
        )
        for candidate in candidates
    })
    state_cache = {
        (rate, xai_weight, specificity): build_states(
            training, condition, rate, xai_weight, specificity
        )
        for rate, xai_weight, specificity in learning_values
    }
    evaluation_cache = {}
    in_sample, parameters = fit_indices(
        family,
        condition,
        training,
        trials,
        candidates,
        state_cache=state_cache,
        evaluation_cache=evaluation_cache,
    )
    fold_count = min(5, len(trials))
    folds = []
    for fold in range(fold_count):
        train_indices = [index for index in range(len(trials)) if index % fold_count != fold]
        test_indices = [index for index in range(len(trials)) if index % fold_count == fold]
        _, fold_parameters = fit_indices(
            family,
            condition,
            training,
            trials,
            candidates,
            train_indices,
            state_cache,
            evaluation_cache,
        )
        folds.append(evaluate(
            family,
            condition,
            fold_parameters,
            training,
            trials,
            test_indices,
            state_cache,
            evaluation_cache,
        ))
    cross_validated = {
        name: float(np.nanmean([fold[name] for fold in folds]))
        for name in in_sample
    }
    return parameters, in_sample, cross_validated


def observed_trials(rows: list[dict[str, str]], case_map: dict[int, dict]) -> list[dict]:
    trials = []
    testing_rows = sorted(
        (row for row in rows if row["phase"] == "testing"),
        key=lambda row: int(float(row["case"])),
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
