# Participant consistency and threshold-like editing in Qualtrics v2.0

## Executive conclusion

Participants show meaningful **target-specific endpoint behavior**, but not a
single, highly precise threshold strategy. A leave-one-out model that predicts
an edit from the participant's other final values for the same feature and
target label was more accurate than a model that assumes the participant
repeats a fixed change amount:

- Median endpoint-model MAE: **0.078** of the normalized feature range.
- Median fixed-change-model MAE: **0.110**.
- Endpoint model won for **39 of 52 participants (75%)**.
- Paired Wilcoxon one-sided p = **0.000059**, FDR q = **0.000234**.

The target label matters. The target-specific endpoint model also beat an
endpoint model that ignored target label:

- Target-specific median MAE: **0.078**.
- Target-agnostic median MAE: **0.132**.
- Target-specific model won for **40 of 52 participants (77%)**.
- Paired Wilcoxon one-sided p = **0.0000023**, FDR q = **0.000019**.

The most defensible interpretation is that many participants behave as if they
have separate feature endpoints for “Diabetes” and “No Diabetes.” This does not
prove that they consciously represented those endpoints as thresholds.

## How threshold consistency was tested

The analysis used 1,040 testing rows from 52 participants and 2,017 changed
feature endpoints. All values are normalized to the experiment ranges.

For every changed feature on every testing trial, three leave-one-out
predictions were constructed:

1. **Target-specific endpoint model:** median of the participant's other final
   values for the same feature and target label.
2. **Fixed-change model:** current original value plus the participant's median
   change amount on other trials involving the same feature and target.
3. **Target-agnostic endpoint model:** median of the participant's other final
   values for that feature, ignoring target label.

Leave-one-out prediction prevents the evaluated endpoint from being used to
predict itself. Only repeated participant-feature-target cells were evaluated.

The within-cell slope of final value on original value provides a second
diagnostic:

- Slope near 0 is consistent with returning to a fixed endpoint.
- Slope near 1 is consistent with adding a fixed change amount.

The overall median slope was **0.294**, closer to an endpoint rule than a
fixed-change rule, but clearly not a perfect endpoint rule.

## The most consistent participants

No participant met a deliberately stringent “near-exact threshold” definition:

- at least 15 evaluable edits and 60% coverage;
- endpoint MAE at most 0.05;
- at least 20% improvement over the fixed-change model; and
- at least 75% of endpoints within 0.05 of their leave-one-out threshold.

The closest participant had endpoint MAE 0.040, but only 72.7% of endpoints
were within 0.05. Thus it would be misleading to claim a group of nearly exact
threshold users.

A broader **strong threshold-like** definition identified 13 participants. It
required at least 15 evaluable edits, at least 60% coverage, endpoint MAE at
most 0.075, at least 20% improvement over the fixed-change model, and an
absolute within-cell slope no greater than 0.25.

| Participant | Condition | Endpoint MAE | Fixed-change MAE | Relative improvement | Slope |
|---|---:|---:|---:|---:|---:|
| R_60Vourt3Kluat5u | Attribution | 0.040 | 0.115 | 65.4% | -0.002 |
| R_1TZbMIIo6RtFnsb | Attribution | 0.050 | 0.142 | 64.5% | 0.023 |
| R_6DD4Hjc7DAhjrhU | Attribution | 0.054 | 0.112 | 52.3% | 0.079 |
| R_61hCu6g2tcWoJo7 | Attribution | 0.066 | 0.137 | 51.9% | 0.234 |
| R_7PsMTUjq8Wa1btt | Counterfactual | 0.058 | 0.108 | 46.5% | 0.191 |
| R_6KDvslaxG4W04I9 | Counterfactual | 0.058 | 0.135 | 57.0% | 0.109 |
| R_1WAqgPTkRSdmI9W | Counterfactual | 0.068 | 0.090 | 24.8% | 0.157 |
| R_1hF6UnZQsvvwBCX | Counterfactual | 0.069 | 0.146 | 52.9% | 0.236 |
| R_6J90tRZsF1dbkUv | None | 0.050 | 0.148 | 66.3% | 0.064 |
| R_1bN1m4OYZIanJ5K | None | 0.054 | 0.087 | 37.7% | 0.060 |
| R_7ojrp9raD1YDwJj | None | 0.065 | 0.121 | 46.4% | 0.131 |
| R_3ENoszlL5q7JglI | None | 0.067 | 0.161 | 58.1% | 0.059 |
| R_7xWlSJ5Zdqog9gt | None | 0.071 | 0.139 | 48.9% | 0.093 |

### Spread across conditions

| Condition | Strong threshold-like | Total | Rate |
|---|---:|---:|---:|
| None | 5 | 16 | 31.3% |
| Attribution | 4 | 16 | 25.0% |
| Counterfactual | 4 | 20 | 20.0% |

There is no evidence that this subgroup is distributed differently across
conditions: omnibus chi-square p = **0.741**. The conclusion is unchanged over
reasonable sensitivity definitions. At an endpoint-MAE cutoff of 0.10, the
rates were 37.5%, 25.0%, and 30.0%, respectively, after retaining the 20%
fixed-change advantage and slope criteria (p = 0.742).

Likewise, continuous endpoint consistency did not differ by condition:
Kruskal-Wallis p = **0.485**. Median endpoint MAEs were 0.072 for none, 0.082
for attribution, and 0.082 for counterfactual.

Within-condition model comparisons are informative but should not be treated as
condition contrasts:

- None: endpoint MAE 0.072 versus fixed-change MAE 0.114; q = 0.006.
- Counterfactual: 0.082 versus 0.127; q = 0.003.
- Attribution: 0.082 versus 0.102; q = 0.174.

The weaker attribution result does not establish an attribution-specific
effect because the direct between-condition comparison was nonsignificant.

## Differences by target label

Endpoint behavior was somewhat more consistent when participants targeted
**No Diabetes**:

- Median MAE targeting Diabetes: **0.084**.
- Median MAE targeting No Diabetes: **0.069**.
- Median paired difference: **0.013**.
- Paired p = **0.016**, but q = **0.081** across the target-consistency test
  family.

The size of this target-label consistency difference did not vary by condition
(p = **0.723**). It should therefore be treated as a general task asymmetry,
not a counterfactual effect.

Across participant-feature pairs with enough data for both targets, 74.3% had
the expected ordering: the estimated Diabetes endpoint was higher than the No
Diabetes endpoint. Target separation did not differ by condition (p = 0.686),
nor did the rate of expected ordering (p = 0.597).

## Estimated endpoint levels across conditions

These are medians of participant-level endpoint estimates. Each participant
contributes one estimate only when they changed the feature at least twice for
that target.

| Feature | Target | None | Attribution | Counterfactual | Omnibus p |
|---|---|---:|---:|---:|---:|
| Glucose | Diabetes | 0.566 | 0.540 | 0.536 | 0.996 |
| Glucose | No Diabetes | 0.328 | 0.391 | 0.370 | 0.642 |
| Blood Pressure | Diabetes | 0.533 | 0.565 | 0.533 | 0.879 |
| Blood Pressure | No Diabetes | 0.370 | 0.304 | 0.391 | 0.152 |
| Insulin | Diabetes | 0.395 | 0.450 | 0.368 | 0.468 |
| Insulin | No Diabetes | 0.140 | 0.203 | 0.192 | 0.324 |
| BMI | Diabetes | 0.468 | 0.450 | 0.458 | 0.997 |
| BMI | No Diabetes | 0.235 | 0.275 | 0.320 | 0.126 |

No feature-target threshold differed significantly across conditions, before or
after multiplicity correction. Age estimates were too sparse for stable
condition comparisons.

The two closest exploratory patterns were both for No Diabetes:

- BMI: counterfactual median 0.320 versus none 0.235; unadjusted pairwise
  p = 0.057.
- Blood pressure: counterfactual median 0.391 versus attribution 0.304;
  unadjusted pairwise p = 0.068.

Both disappear after correction and should be hypotheses for replication, not
findings. They point in the direction of counterfactual participants choosing
somewhat **higher/less aggressive lowering endpoints** for these two features.

## Did counterfactual participants reproduce training endpoints?

Testing endpoints were compared with the nearest displayed counterfactual
training endpoint for the same feature and inferred target label. There was no
condition difference in mean nearest-endpoint distance (p = **0.550**) or in
the rate of coming within 0.025 of a displayed endpoint (p = **0.567**).

Thus the current data do not support the specific claim that counterfactual
participants memorized and reused the numerical endpoints shown during
training.

## Interpretation and limitations

The most revealing result is a general response regularity:

> Participants often move a selected feature toward a target-dependent region
> of its scale, rather than adding the same amount on every trial.

This regularity is present in all three conditions. Counterfactual explanations
did not measurably increase its precision, prevalence, target separation, or
alignment with displayed training endpoints.

“Threshold” here means a statistically stable endpoint. It does not establish
conscious threshold reasoning. Stable endpoints could also arise from slider
anchors, preferred round numbers, range clipping, or common beliefs about
healthy values. Threshold estimates are conditional on participants choosing
the feature, so cross-condition endpoint comparisons can also reflect
differences in which instances prompted that feature to be edited.

For a stronger test in the next iteration, manipulate the displayed training
endpoint while holding the case and feature constant across randomized groups.
If later endpoints move with the experimentally assigned training endpoint,
that would identify remembered numerical counterfactual information much more
cleanly than observational consistency alone.
