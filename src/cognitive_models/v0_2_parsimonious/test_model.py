"""Deterministic checks for cognitive model v0.2."""

import numpy as np

from .model import (
    SUBSETS,
    core_subset_distribution,
    counterfactual_additive_cue,
    inclusion_probabilities,
    observed_subset_probability,
    subset_distribution,
)


def test_core_subset_distribution_sums_to_one():
    probabilities = core_subset_distribution(np.asarray([1.0, 0.5, 0.0, 0.25, 0.1]), 1.0)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0)


def test_lapse_gives_every_nonempty_subset_positive_probability():
    probabilities = subset_distribution(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]), 0.5, 0.1)
    assert len(probabilities) == len(SUBSETS) == 31
    assert np.all(probabilities > 0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_lapse_is_applied_independently_to_features():
    scores = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    inclusion = inclusion_probabilities(scores, spread=1.0, lapse=0.2)
    assert np.isclose(inclusion[0], 0.8 * (1.0 - np.exp(-1.0)) + 0.1)
    assert np.allclose(inclusion[1:], 0.1)


def test_subset_probability_factorizes_over_features():
    scores = np.asarray([1.0, 0.5, 0.0, 0.0, 0.0])
    inclusion = inclusion_probabilities(scores, spread=0.7, lapse=0.1)
    probabilities = subset_distribution(scores, spread=0.7, lapse=0.1)
    one = probabilities[SUBSETS.index((0,))]
    both = probabilities[SUBSETS.index((0, 1))]
    assert np.isclose(both / one, inclusion[1] / (1.0 - inclusion[1]))


def test_direct_observed_probability_matches_enumeration():
    scores = np.asarray([1.0, 0.5, 0.2, 0.0, 0.1])
    probabilities = subset_distribution(scores, spread=0.7, lapse=0.1)
    observed = (0, 2)
    assert np.isclose(
        probabilities[SUBSETS.index(observed)],
        observed_subset_probability(scores, 0.7, 0.1, observed),
    )


def test_larger_parameter_spreads_selection():
    scores = np.asarray([1.0, 0.25, 0.0, 0.0, 0.0])
    concentrated = core_subset_distribution(scores, 0.1)
    spread = core_subset_distribution(scores, 4.0)
    both = SUBSETS.index((0, 1))
    assert spread[both] > concentrated[both]


def test_counterfactual_cue_is_canonical_and_normalized():
    cue = counterfactual_additive_cue(np.asarray([-0.2, 0.1, 0.0]), target=1)
    assert cue[0] < 0 < cue[1]
    assert np.isclose(np.abs(cue).sum(), 1.0)
