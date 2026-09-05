"""CoAX-style memory model for counterfactual feature selection and edit amounts.

The model crosses three stored representations (base instances, additive
evidence, and counterfactual edits) with two memory scopes (prototype and
exemplar). All variants share power-law recency, a variable-cardinality
selection rule, a global lapse, and an additive normalized edit margin.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Sequence

import numpy as np


FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
REPRESENTATIONS = ("base", "additive", "counterfactual")
SCOPES = ("prototype", "exemplar")
MODEL_VARIANTS = tuple(
    f"{representation}_{scope}"
    for representation in REPRESENTATIONS
    for scope in SCOPES
)

DECAY_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DISTANCE_GRID = (0.0, 2.0, 6.0, 12.0, 24.0)
SPREAD_GRID = (0.025, 0.05, 0.10, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MARGIN_GRID = (0.0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40)
EDIT_BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
AGE_ACTIONABLE_GRID = (0, 1)
LAPSE_GRID = (
    0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.80,
)
NUMERICAL_EPSILON = 1e-10


@dataclass(frozen=True)
class Memory:
    profiles: np.ndarray
    labels: np.ndarray
    ages: np.ndarray
    canonical_attributions: np.ndarray
    counterfactual_deltas: np.ndarray
    counterfactual_targets: np.ndarray
    condition: str


@dataclass(frozen=True)
class Trial:
    profile: np.ndarray
    target: int
    observed: np.ndarray
    instance_id: int
    order: int

    @property
    def observed_subset(self) -> tuple[int, ...]:
        return tuple(np.flatnonzero(np.abs(self.observed) > 1e-8).tolist())


@dataclass(frozen=True)
class ModelParameters:
    decay: float
    distance: float
    spread: float
    margin: float = 0.0
    edit_blend: float = 0.0
    age_actionable: int = 1


@dataclass(frozen=True)
class Proposal:
    scores: np.ndarray
    boundary_delta: np.ndarray
    direction: np.ndarray


def label_sign(label: str) -> int:
    """Use Diabetes as the canonical positive class in this experiment."""
    return 1 if str(label).strip().casefold() == "diabetes" else -1


def normalize_values(values: Sequence[float], case: dict) -> np.ndarray:
    normalized = []
    for value, bounds in zip(values, case["feature_ranges"]):
        low, high = map(float, bounds)
        normalized.append((float(value) - low) / (high - low))
    return np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)


def normalize_case(case: dict) -> np.ndarray:
    return normalize_values(case["feature_values"], case)


def canonical_attribution(case: dict) -> np.ndarray:
    """Return contributions oriented so positive values support Diabetes.

    The experiment stores signed LIME contributions with positive bars pointing
    to attribution.direction_labels.right. Convert that display orientation to
    a fixed class orientation before combining explanations across cases.
    """
    values = np.asarray(case["attribution"]["values"], dtype=float)
    right = case["attribution"].get("direction_labels", {}).get("right")
    return values if label_sign(right) == 1 else -values


def counterfactual_additive_cue(delta: np.ndarray, target: int) -> np.ndarray:
    """Infer a signed coefficient cue from a successful counterfactual.

    Smaller sufficient changes imply stronger feature influence. Multiplying by
    the target sign converts the action direction into a canonical coefficient
    direction: positive means that increasing the feature supports Diabetes.
    """
    cue = np.zeros_like(delta, dtype=float)
    changed = np.abs(delta) > NUMERICAL_EPSILON
    cue[changed] = (
        target * np.sign(delta[changed])
        / (np.abs(delta[changed]) + NUMERICAL_EPSILON)
    )
    total = float(np.abs(cue).sum())
    return cue / total if total > NUMERICAL_EPSILON else cue


def build_memory(training_cases: Sequence[dict], condition: str) -> Memory:
    profiles = np.vstack([normalize_case(case) for case in training_cases])
    labels = np.asarray(
        [label_sign(case["prediction"]["label"]) for case in training_cases],
        dtype=int,
    )
    deltas = np.vstack([
        normalize_values(case["counterfactual"]["feature_values"], case)
        - normalize_case(case)
        for case in training_cases
    ])
    targets = np.asarray([
        label_sign(case["counterfactual"]["prediction"]["label"])
        for case in training_cases
    ])
    if condition == "attribution":
        additive = np.vstack([canonical_attribution(case) for case in training_cases])
    elif condition == "counterfactual":
        additive = np.vstack([
            counterfactual_additive_cue(delta, target)
            for delta, target in zip(deltas, targets)
        ])
    else:
        additive = np.zeros_like(profiles)
    # Equal trial intervals; the final exposure is age one at the first test.
    ages = np.arange(len(training_cases), 0, -1, dtype=float)
    return Memory(
        profiles=profiles,
        labels=labels,
        ages=ages,
        canonical_attributions=additive,
        counterfactual_deltas=deltas,
        counterfactual_targets=targets,
        condition=condition,
    )


def base_weights(memory: Memory, decay: float, test_offset: int = 0) -> np.ndarray:
    return np.power(memory.ages + test_offset, -decay)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= NUMERICAL_EPSILON:
        return np.mean(values, axis=0)
    return np.asarray(weights @ values / total, dtype=float)


def weighted_diagnosticity(memory: Memory, weights: np.ndarray) -> np.ndarray:
    """Activation-weighted Welch-t-type discriminative feature strength."""
    statistics = np.zeros(memory.profiles.shape[1], dtype=float)
    groups = []
    for label in (-1, 1):
        mask = memory.labels == label
        group_values = memory.profiles[mask]
        group_weights = weights[mask]
        normalized = group_weights / max(float(group_weights.sum()), NUMERICAL_EPSILON)
        mean = normalized @ group_values
        variance = normalized @ np.square(group_values - mean)
        effective_n = 1.0 / max(float(np.square(normalized).sum()), NUMERICAL_EPSILON)
        groups.append((mean, variance, effective_n))
    (negative_mean, negative_var, negative_n), (positive_mean, positive_var, positive_n) = groups
    denominator = np.sqrt(
        negative_var / max(negative_n, 1.0)
        + positive_var / max(positive_n, 1.0)
        + NUMERICAL_EPSILON
    )
    statistics = np.abs(positive_mean - negative_mean) / denominator
    maximum = float(np.max(statistics))
    return statistics / maximum if maximum > NUMERICAL_EPSILON else np.ones_like(statistics)


def aggregation_weights(
    memory: Memory,
    profile: np.ndarray,
    label: int,
    scope: str,
    decay: float,
    distance_sensitivity: float,
    attention: np.ndarray,
    transition_target: int | None = None,
    test_offset: int = 0,
) -> np.ndarray:
    activation = base_weights(memory, decay, test_offset)
    mask = memory.labels == label
    if transition_target is not None:
        mask &= memory.counterfactual_targets == transition_target
    weights = activation * mask.astype(float)
    if scope == "exemplar":
        attention_sum = max(float(attention.sum()), NUMERICAL_EPSILON)
        distance = np.sum(
            (attention / attention_sum) * np.abs(memory.profiles - profile), axis=1
        )
        weights *= np.exp(-distance_sensitivity * distance)
    return weights


def class_representation(
    memory: Memory,
    profile: np.ndarray,
    label: int,
    scope: str,
    parameters: ModelParameters,
    attention: np.ndarray,
    test_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = aggregation_weights(
        memory,
        profile,
        label,
        scope,
        parameters.decay,
        parameters.distance,
        attention,
        test_offset=test_offset,
    )
    return (
        weighted_mean(memory.profiles, weights),
        weighted_mean(memory.canonical_attributions, weights),
    )


def proposal_for_trial(
    memory: Memory,
    trial: Trial,
    representation: str,
    scope: str,
    parameters: ModelParameters,
) -> Proposal:
    test_offset = trial.order - 1
    activation = base_weights(memory, parameters.decay, test_offset)
    diagnosticity = weighted_diagnosticity(memory, activation)
    target_values, target_additive = class_representation(
        memory,
        trial.profile,
        trial.target,
        scope,
        parameters,
        diagnosticity,
        test_offset,
    )
    target_difference = target_values - trial.profile

    if representation == "base":
        delta = diagnosticity * target_difference
        scores = np.abs(delta)
        if not parameters.age_actionable:
            scores[-1] = 0.0
        return Proposal(scores, delta, target_difference)

    if representation == "additive":
        current_values, current_additive = class_representation(
            memory,
            trial.profile,
            -trial.target,
            scope,
            parameters,
            diagnosticity,
            test_offset,
        )
        if memory.condition == "attribution":
            current_contribution = trial.target * current_additive
            target_contribution = trial.target * target_additive
            denominator = target_contribution - current_contribution
            fraction = np.divide(
                -current_contribution,
                denominator,
                out=np.full_like(denominator, 0.5),
                where=np.abs(denominator) > NUMERICAL_EPSILON,
            )
            crosses_zero = current_contribution * target_contribution <= 0
            fraction = np.clip(fraction, 0.0, 1.0)
            delta = fraction * target_difference
            scores = np.abs(target_difference) * crosses_zero.astype(float)
        else:
            # Counterfactual cues estimate coefficient direction, not a SHAP
            # contribution. Their zero-contribution point is the midpoint of
            # the current- and target-class value representations.
            all_weights = base_weights(memory, parameters.decay, test_offset)
            if scope == "exemplar":
                distances = np.sum(
                    diagnosticity / max(float(diagnosticity.sum()), NUMERICAL_EPSILON)
                    * np.abs(memory.profiles - trial.profile),
                    axis=1,
                )
                all_weights *= np.exp(-parameters.distance * distances)
            coefficient = weighted_mean(memory.canonical_attributions, all_weights)
            midpoint = (current_values + target_values) / 2.0
            delta = midpoint - trial.profile
            supports_target = trial.target * coefficient * target_difference > 0
            scores = np.abs(target_difference) * supports_target.astype(float)
        if float(scores.max()) <= NUMERICAL_EPSILON:
            scores = np.abs(target_difference)
        if not parameters.age_actionable:
            scores[-1] = 0.0
        return Proposal(scores, delta, target_difference)

    if representation == "counterfactual":
        weights = aggregation_weights(
            memory,
            trial.profile,
            -trial.target,
            scope,
            parameters.decay,
            parameters.distance,
            diagnosticity,
            transition_target=trial.target,
            test_offset=test_offset,
        )
        # Consolidate each feature independently.  Frequency says how often a
        # feature was edited; its direction and amount are averaged only over
        # episodes in which that feature was actually edited.  This avoids
        # treating a demonstrated feature pair as an indivisible memory and
        # avoids diluting a conditional edit by all non-editing episodes.
        edited = np.abs(memory.counterfactual_deltas) > NUMERICAL_EPSILON
        weighted_edited = weights[:, None] * edited.astype(float)
        total_weight = max(float(weights.sum()), NUMERICAL_EPSILON)
        feature_weight = weighted_edited.sum(axis=0)
        frequency = feature_weight / total_weight
        denominator = np.maximum(feature_weight, NUMERICAL_EPSILON)
        remembered_change = (
            weighted_edited * memory.counterfactual_deltas
        ).sum(axis=0) / denominator
        endpoints = memory.profiles + memory.counterfactual_deltas
        remembered_endpoint = (
            weighted_edited * endpoints
        ).sum(axis=0) / denominator
        endpoint_change = remembered_endpoint - trial.profile
        has_memory = feature_weight > NUMERICAL_EPSILON
        edit_vector = np.where(
            has_memory,
            (1.0 - parameters.edit_blend) * endpoint_change
            + parameters.edit_blend * remembered_change,
            0.0,
        )
        scores = frequency * np.abs(edit_vector)
        if not parameters.age_actionable:
            scores[-1] = 0.0
        return Proposal(scores, edit_vector, edit_vector)

    raise ValueError(f"Unknown representation: {representation}")


def all_nonempty_subsets(feature_count: int = len(FEATURES)) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(1, feature_count + 1)
        for subset in itertools.combinations(range(feature_count), size)
    )


SUBSETS = all_nonempty_subsets()
SELECTION_SUBSETS = list(SUBSETS)
ACTIVE_AGE_ACTIONABLE_GRID = AGE_ACTIONABLE_GRID


def configure_selection_space(
    locked_indices: Sequence[int] = (),
    cardinality: int | None = None,
) -> None:
    """Condition feature selection on constraints imposed by the interface."""
    global ACTIVE_AGE_ACTIONABLE_GRID
    locked = set(locked_indices)
    SELECTION_SUBSETS[:] = [
        subset for subset in SUBSETS
        if not locked.intersection(subset)
        and (cardinality is None or len(subset) == cardinality)
    ]
    if not SELECTION_SUBSETS:
        raise ValueError("Interface constraints leave no eligible feature subsets")
    # Age is editable unless it is explicitly among the locked features.
    ACTIVE_AGE_ACTIONABLE_GRID = (0,) if len(FEATURES) - 1 in locked else (1,)


def selection_space() -> tuple[tuple[int, ...], ...]:
    return tuple(SELECTION_SUBSETS)


def inclusion_probabilities(
    scores: np.ndarray,
    spread: float,
    lapse: float = 0.0,
) -> np.ndarray:
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    maximum = float(scores.max())
    if maximum <= NUMERICAL_EPSILON:
        core = np.full(len(FEATURES), 0.5)
    else:
        # A Poisson-hazard link gives variable cardinality without an
        # intercept or observed K. Larger spread raises each independent
        # feature's probability of being included.
        core = 1.0 - np.exp(-spread * scores / maximum)
    # The lapse now acts independently on every feature. In the lapse state a
    # feature is an uninformed Bernoulli(.5) choice; this is not a post-hoc
    # mixture over completed subsets.
    return (1.0 - lapse) * core + lapse * 0.5


def independent_subset_distribution(
    scores: np.ndarray,
    spread: float,
    lapse: float,
) -> np.ndarray:
    # A Poisson-hazard link gives variable cardinality without an intercept or
    # fixed K. Near zero, conditioning on a nonempty edit approximates a
    # one-feature draw proportional to score; larger spread raises inclusion
    # probabilities and produces broader multi-feature edits.
    inclusion = inclusion_probabilities(scores, spread, lapse)
    probabilities = np.asarray([
        math.prod(
            inclusion[index] if index in subset else 1.0 - inclusion[index]
            for index in range(len(FEATURES))
        )
        for subset in SELECTION_SUBSETS
    ])
    total = float(probabilities.sum())
    if total <= NUMERICAL_EPSILON:
        return np.full(len(SELECTION_SUBSETS), 1.0 / len(SELECTION_SUBSETS))
    return probabilities / total


def subset_distribution(scores: np.ndarray, spread: float, lapse: float) -> np.ndarray:
    return independent_subset_distribution(scores, spread, lapse)


def core_subset_distribution(scores: np.ndarray, spread: float) -> np.ndarray:
    return independent_subset_distribution(scores, spread, 0.0)


def selection_f1(predicted: tuple[int, ...], observed: tuple[int, ...]) -> float:
    predicted_set, observed_set = set(predicted), set(observed)
    return 2.0 * len(predicted_set & observed_set) / (
        len(predicted_set) + len(observed_set)
    )


def apply_margin(
    profile: np.ndarray,
    proposal: Proposal,
    subset: tuple[int, ...],
    margin: float,
) -> np.ndarray:
    delta = np.zeros(len(FEATURES), dtype=float)
    selected = np.asarray(subset, dtype=int)
    if len(selected):
        direction = np.sign(proposal.direction[selected])
        fallback = np.sign(proposal.boundary_delta[selected])
        direction = np.where(direction == 0, fallback, direction)
        delta[selected] = proposal.boundary_delta[selected] + margin * direction
    return np.clip(profile + delta, 0.0, 1.0) - profile


def parameter_grid(representation: str, scope: str) -> tuple[ModelParameters, ...]:
    distances = DISTANCE_GRID if scope == "exemplar" else (0.0,)
    edit_blends = EDIT_BLEND_GRID if representation == "counterfactual" else (0.0,)
    return tuple(
        ModelParameters(
            decay=decay,
            distance=distance,
            spread=spread,
            edit_blend=edit_blend,
            age_actionable=age_actionable,
        )
        for decay, distance, spread, edit_blend, age_actionable in itertools.product(
            DECAY_GRID, distances, SPREAD_GRID, edit_blends, ACTIVE_AGE_ACTIONABLE_GRID
        )
    )


def available_variants(condition: str) -> tuple[str, ...]:
    representations = ["base"]
    if condition in {"attribution", "counterfactual"}:
        representations.append("additive")
    if condition == "counterfactual":
        representations.append("counterfactual")
    return tuple(
        f"{representation}_{scope}"
        for representation in representations
        for scope in SCOPES
    )


def split_variant(variant: str) -> tuple[str, str]:
    return tuple(variant.rsplit("_", 1))  # type: ignore[return-value]


def proposal_cache(
    memory: Memory,
    trials: Sequence[Trial],
    variant: str,
) -> tuple[tuple[ModelParameters, tuple[Proposal, ...]], ...]:
    representation, scope = split_variant(variant)
    # Spread changes only the Bernoulli inclusion probabilities, not the
    # representation itself. Build each representation once and reuse it over
    # the spread grid rather than recomputing the same prototype/retrieval.
    unique: dict[tuple[float, float, float, int], tuple[Proposal, ...]] = {}
    output = []
    for parameters in parameter_grid(representation, scope):
        key = (
            parameters.decay,
            parameters.distance,
            parameters.edit_blend,
            parameters.age_actionable,
        )
        if key not in unique:
            unique[key] = tuple(
                proposal_for_trial(memory, trial, representation, scope, parameters)
                for trial in trials
            )
        output.append((parameters, unique[key]))
    return tuple(output)


def probability_matrix(
    cache: tuple[tuple[ModelParameters, tuple[Proposal, ...]], ...],
    trials: Sequence[Trial],
    lapse: float,
) -> np.ndarray:
    scores = np.asarray([
        [proposal.scores for proposal in proposals]
        for _, proposals in cache
    ])
    maximum = np.max(scores, axis=2, keepdims=True)
    relative = np.divide(
        scores,
        maximum,
        out=np.zeros_like(scores),
        where=maximum > NUMERICAL_EPSILON,
    )
    spreads = np.asarray([parameters.spread for parameters, _ in cache])[:, None, None]
    core = 1.0 - np.exp(-spreads * relative)
    no_signal = maximum <= NUMERICAL_EPSILON
    core = np.where(no_signal, 0.5, core)
    inclusion = (1.0 - lapse) * core + lapse * 0.5
    observed = np.zeros((len(trials), len(FEATURES)), dtype=bool)
    for index, trial in enumerate(trials):
        observed[index, list(trial.observed_subset)] = True
    raw = np.prod(np.where(observed[None, :, :], inclusion, 1.0 - inclusion), axis=2)
    subset_masks = np.zeros((len(SELECTION_SUBSETS), len(FEATURES)), dtype=bool)
    for subset_index, subset in enumerate(SELECTION_SUBSETS):
        subset_masks[subset_index, list(subset)] = True
    allowed_raw = np.prod(
        np.where(
            subset_masks[None, None, :, :],
            inclusion[:, :, None, :],
            1.0 - inclusion[:, :, None, :],
        ),
        axis=3,
    )
    normalizer = allowed_raw.sum(axis=2)
    return raw / np.maximum(normalizer, NUMERICAL_EPSILON)


def observed_subset_probability(
    scores: np.ndarray,
    spread: float,
    lapse: float,
    observed: tuple[int, ...],
) -> float:
    """Probability of one observed subset without enumerating all subsets."""
    inclusion = inclusion_probabilities(scores, spread, lapse)
    observed_set = set(observed)
    raw_probability = float(math.prod(
        inclusion[index] if index in observed_set else 1.0 - inclusion[index]
        for index in range(len(FEATURES))
    ))
    allowed_probability = sum(
        math.prod(
            inclusion[index] if index in subset else 1.0 - inclusion[index]
            for index in range(len(FEATURES))
        )
        for subset in SELECTION_SUBSETS
    )
    return raw_probability / max(allowed_probability, NUMERICAL_EPSILON)


def core_observed_probability(
    scores: np.ndarray,
    spread: float,
    observed: tuple[int, ...],
) -> float:
    return observed_subset_probability(scores, spread, 0.0, observed)


def core_probability_matrix(
    cache: tuple[tuple[ModelParameters, tuple[Proposal, ...]], ...],
    trials: Sequence[Trial],
) -> np.ndarray:
    return np.asarray([
        [
            core_observed_probability(
                proposal.scores, parameters.spread, trial.observed_subset
            )
            for proposal, trial in zip(proposals, trials)
        ]
        for parameters, proposals in cache
    ])


def fold_indices(trial_count: int, fold_count: int = 5):
    fold_count = min(fold_count, trial_count)
    for fold in range(fold_count):
        test = np.asarray([index for index in range(trial_count) if index % fold_count == fold])
        train = np.asarray([index for index in range(trial_count) if index % fold_count != fold])
        yield train, test


def cross_validated_selection_nll(
    probabilities: np.ndarray,
) -> float:
    losses = -np.log(np.maximum(probabilities, NUMERICAL_EPSILON))
    held_out = []
    for train, test in fold_indices(probabilities.shape[1]):
        winner = int(np.argmin(np.mean(losses[:, train], axis=1)))
        held_out.extend(losses[winner, test].tolist())
    return float(np.mean(held_out))


def fit_global_lapse(
    participant_data: Sequence[tuple[str, Memory, Sequence[Trial]]],
) -> tuple[float, list[dict[str, float]]]:
    """Choose one lapse by pooled participant-level cross-validated NLL."""
    totals = {lapse: [] for lapse in LAPSE_GRID}
    for _, memory, trials in participant_data:
        caches = {
            variant: proposal_cache(memory, trials, variant)
            for variant in available_variants(memory.condition)
        }
        for lapse in LAPSE_GRID:
            variant_scores = [
                cross_validated_selection_nll(
                    probability_matrix(caches[variant], trials, lapse)
                )
                for variant in caches
            ]
            totals[lapse].append(min(variant_scores))
    rows = [
        {
            "lapse": lapse,
            "pooled_cv_selection_nll": float(np.mean(values)),
        }
        for lapse, values in totals.items()
    ]
    winner = min(rows, key=lambda row: row["pooled_cv_selection_nll"])
    return float(winner["lapse"]), rows


def amount_metrics(
    trials: Sequence[Trial],
    proposals: Sequence[Proposal],
    margin: float,
    indices: Iterable[int],
) -> dict[str, float]:
    absolute_errors = []
    directions = []
    zero_errors = []
    for index in indices:
        trial, proposal = trials[index], proposals[index]
        selected = np.asarray(trial.observed_subset, dtype=int)
        predicted = apply_margin(
            trial.profile, proposal, trial.observed_subset, margin
        )
        if len(selected):
            absolute_errors.extend(
                np.abs(predicted[selected] - trial.observed[selected]).tolist()
            )
            directions.extend(
                (np.sign(predicted[selected]) == np.sign(trial.observed[selected])).tolist()
            )
            zero_errors.extend(np.abs(trial.observed[selected]).tolist())
    return {
        "amount_mae": float(np.mean(absolute_errors)),
        "direction_accuracy": float(np.mean(directions)),
        "zero_amount_mae": float(np.mean(zero_errors)),
    }


def selection_metrics(
    trials: Sequence[Trial],
    proposals: Sequence[Proposal],
    parameters: ModelParameters,
    lapse: float,
    indices: Iterable[int],
) -> dict[str, float]:
    nll, expected_f1, map_f1, exact, cardinality_error = [], [], [], [], []
    subset_index = {
        subset: index for index, subset in enumerate(SELECTION_SUBSETS)
    }
    for index in indices:
        trial, proposal = trials[index], proposals[index]
        probabilities = subset_distribution(
            proposal.scores, parameters.spread, lapse
        )
        observed = trial.observed_subset
        nll.append(-math.log(max(probabilities[subset_index[observed]], NUMERICAL_EPSILON)))
        expected_f1.append(sum(
            probability * selection_f1(subset, observed)
            for subset, probability in zip(SELECTION_SUBSETS, probabilities)
        ))
        predicted = SELECTION_SUBSETS[int(np.argmax(probabilities))]
        map_f1.append(selection_f1(predicted, observed))
        exact.append(float(predicted == observed))
        cardinality_error.append(abs(len(predicted) - len(observed)))
    return {
        "selection_nll": float(np.mean(nll)),
        "expected_selection_f1": float(np.mean(expected_f1)),
        "map_selection_f1": float(np.mean(map_f1)),
        "exact_subset_accuracy": float(np.mean(exact)),
        "cardinality_mae": float(np.mean(cardinality_error)),
    }


def fit_variant(
    memory: Memory,
    trials: Sequence[Trial],
    variant: str,
    lapse: float,
) -> tuple[dict, list[dict]]:
    cache = proposal_cache(memory, trials, variant)
    probabilities = probability_matrix(cache, trials, lapse)
    losses = -np.log(np.maximum(probabilities, NUMERICAL_EPSILON))
    full_index = int(np.argmin(np.mean(losses, axis=1)))
    full_parameters, full_proposals = cache[full_index]
    full_margin = min(
        MARGIN_GRID,
        key=lambda margin: amount_metrics(
            trials, full_proposals, margin, range(len(trials))
        )["amount_mae"],
    )
    in_sample = {
        **selection_metrics(
            trials, full_proposals, full_parameters, lapse, range(len(trials))
        ),
        **amount_metrics(trials, full_proposals, full_margin, range(len(trials))),
    }

    trial_rows = []
    fold_metrics = []
    for fold, (train, test) in enumerate(fold_indices(len(trials))):
        candidate_index = int(np.argmin(np.mean(losses[:, train], axis=1)))
        parameters, proposals = cache[candidate_index]
        margin = min(
            MARGIN_GRID,
            key=lambda value: amount_metrics(trials, proposals, value, train)["amount_mae"],
        )
        metrics = {
            **selection_metrics(trials, proposals, parameters, lapse, test),
            **amount_metrics(trials, proposals, margin, test),
        }
        fold_metrics.append(metrics)
        for index in test:
            trial, proposal = trials[index], proposals[index]
            distribution = subset_distribution(
                proposal.scores, parameters.spread, lapse
            )
            predicted_subset = SELECTION_SUBSETS[int(np.argmax(distribution))]
            predicted_delta = apply_margin(
                trial.profile, proposal, predicted_subset, margin
            )
            conditional_delta = apply_margin(
                trial.profile, proposal, trial.observed_subset, margin
            )
            trial_rows.append({
                "fold": fold,
                "trial_order": trial.order,
                "instance_id": trial.instance_id,
                "target": trial.target,
                "observed_subset": "/".join(FEATURES[j] for j in trial.observed_subset),
                "predicted_subset": "/".join(FEATURES[j] for j in predicted_subset),
                "observed_delta": "/".join(f"{value:.6g}" for value in trial.observed),
                "predicted_delta": "/".join(f"{value:.6g}" for value in predicted_delta),
                "conditional_delta": "/".join(f"{value:.6g}" for value in conditional_delta),
                "observed_subset_probability": float(
                    distribution[{subset: i for i, subset in enumerate(SELECTION_SUBSETS)}[trial.observed_subset]]
                ),
                "decay": parameters.decay,
                "distance": parameters.distance,
                "selection_spread": parameters.spread,
                "edit_blend": parameters.edit_blend,
                "age_actionable": parameters.age_actionable,
                "margin": margin,
                "feature_scores": "/".join(f"{value:.6g}" for value in proposal.scores),
                "base_delta": "/".join(f"{value:.6g}" for value in proposal.boundary_delta),
            })
    cv = {
        name: float(np.mean([fold[name] for fold in fold_metrics]))
        for name in fold_metrics[0]
    }
    representation, scope = split_variant(variant)
    result = {
        "variant": variant,
        "representation": representation,
        "scope": scope,
        "lapse_global": lapse,
        "decay": full_parameters.decay,
        "distance": full_parameters.distance if scope == "exemplar" else math.nan,
        "selection_spread": full_parameters.spread,
        "edit_blend": full_parameters.edit_blend,
        "age_actionable": full_parameters.age_actionable,
        "margin": full_margin,
        **{f"in_sample_{name}": value for name, value in in_sample.items()},
        **{f"cv_{name}": value for name, value in cv.items()},
    }
    return result, trial_rows


def fit_participant(
    participant: str,
    memory: Memory,
    trials: Sequence[Trial],
    lapse: float,
) -> tuple[list[dict], list[dict]]:
    fits, trial_rows = [], []
    for variant in available_variants(memory.condition):
        result, predictions = fit_variant(memory, trials, variant, lapse)
        result.update({
            "participant": participant,
            "xai": memory.condition,
            "testing_cases": len(trials),
        })
        fits.append(result)
        for row in predictions:
            row.update({
                "participant": participant,
                "xai": memory.condition,
                "variant": variant,
            })
        trial_rows.extend(predictions)
    winner = min(
        fits,
        key=lambda row: (row["cv_selection_nll"], row["cv_amount_mae"]),
    )
    for row in fits:
        row["selected_by_cv"] = int(row is winner)
    return fits, trial_rows
