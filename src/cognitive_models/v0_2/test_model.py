"""Small deterministic checks for the v0.2 probability calculations."""

import numpy as np

from .model import MentalState, blend_states, subset_distribution


def test_subset_probabilities_sum_to_one():
    proposals = subset_distribution(np.arange(5, dtype=float), 2, 1)
    assert len(proposals) == 10
    assert np.isclose(sum(probability for _, probability in proposals), 1.0)


def test_immutable_age_is_excluded():
    proposals = subset_distribution(np.arange(5, dtype=float), 2, 0)
    assert all(4 not in subset for subset, _ in proposals)


def test_complete_states_are_blended_before_readout():
    first = MentalState(
        1,
        np.zeros(5),
        -1,
        np.zeros(5),
        np.full(5, 0.2),
        np.full(5, 0.6),
        np.zeros(5),
    )
    second = MentalState(
        2,
        np.ones(5),
        1,
        np.ones(5),
        np.full(5, 0.4),
        np.full(5, 0.8),
        np.full(5, 2.0),
    )
    blended = blend_states([(first, 0.25), (second, 0.75)])
    assert np.allclose(blended.negative_mean, 0.35)
    assert np.allclose(blended.positive_mean, 0.75)
    assert np.allclose(blended.attention_logits, 1.5)
