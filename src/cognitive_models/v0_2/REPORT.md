# Unified-specificity v0.2 fit report

Dataset: 36 complete participants, 12 ordered training exposures and 20 testing
responses per participant.

The model uses one fitted `specificity` parameter in both places where profile
similarity is calculated:

1. ALCOVE-style error-driven attention learning during training;
2. exemplar and counterfactual-edit retrieval during testing.

There are no separate c and lambda parameters. Complete attention and class-mean
states are stored after every exposure and activation-weighted before feature
selection and conditional edit-amount calculations.

## Selected-family cross-validated results

| Model | Selection NLL | Selection F1 | Conditional amount MAE | Combined loss | Family winners |
| --- | ---: | ---: | ---: | ---: | --- |
| v0.1 | 1.5045 | 0.5532 | **0.1965** | 0.3217 | 25 feature / 11 exemplar |
| v0.2 perfect memory | 1.4422 | 0.5508 | 0.2207 | 0.3350 | 31 feature / 5 exemplar |
| v0.2 memory variation | **1.3926** | **0.5603** | 0.1983 | **0.3190** | 24 feature / 12 exemplar |

Memory variation versus perfect memory:

- selection NLL: -0.0496, paired p = .299;
- selection F1: +0.0095, paired p = .430;
- conditional amount MAE: -0.0225, paired p = .041;
- combined loss: -0.0160, paired p = .072.

Memory variation has lower selection NLL for 16/36 participants and lower
combined loss for 24/36.

## Comparison with v0.1

Memory variation improves selection NLL by 0.1119 relative to v0.1 (paired
p = .046). Selection F1 is 0.0072 higher, conditional amount MAE is 0.0017
higher, and combined loss is 0.0027 lower; those three differences are not
statistically distinguishable from zero.

Thus memory variation is descriptively best and significantly improves exact
subset likelihood, but its overall edit prediction remains effectively tied
with v0.1.

## Condition pattern

Memory-variation minus perfect-memory combined loss:

- counterfactual: -0.0183;
- attribution: -0.0097;
- no explanation: -0.0187.

Negative changes favour memory variation.

## Parameter diagnostics

Selected specificity values across both memory variants:

- specificity = 0: 32/72 selected models;
- specificity = 2: 12/72;
- specificity = 6: 28/72.

Selected memory-variation decay values:

- d = 0: 16 participants;
- d = 0.5: 1 participant;
- d = 1: 1 participant;
- d = 2: 6 participants;
- d = 4: 12 participants.

Both distributions are polarized. The fitted values describe broad versus
selective similarity and no-recency versus near-most-recent retrieval regimes;
they should not be interpreted as precise continuous psychological estimates.
