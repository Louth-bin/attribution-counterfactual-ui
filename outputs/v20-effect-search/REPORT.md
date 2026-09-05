# Exploratory v2.0 XAI effect search

## Bottom line

There is no defensible average-performance result that says either XAI condition
improves counterfactual success, boundary distance, confidence gain, edit cost,
plausibility, or SHAP-aligned movement across all participants and all 20 test
cases. The broad search produced zero overall, moderator-interaction, or
instance-specific findings at FDR q < .10.

The data do, however, support a useful next-iteration account:

1. Counterfactual XAI appears to change the *structure* of responses. Compared
   with attribution, participants spread edits across more attributes, especially
   Insulin, without increasing total L1 movement or improving outcomes.
2. A specific post-hoc strategy phenotype is much more common in the
   counterfactual arm: participants whose fitted feature-contribution model puts
   full weight on global relevance rather than local opposition. This phenotype
   is not simply low effort or confusion. It is a systematic but poorly
   conditional strategy: repeatedly using globally salient features while often
   missing the locally appropriate direction or amount.
3. Distance to a training case is not sufficient to predict benefit. Pair match,
   direction, and local applicability matter. Several case-level patterns are
   compatible with over-transferring a familiar edit template.

This suggests that the next experiment should target *conditional transfer* and
*edit diffusion*, rather than merely increase sample size for the same broad
success outcome.

## Data and inference

- Canonical source: a fresh export of `qualtrics_results_v2.0.jmp`, which was
  newer than the existing CSV exports.
- 62 complete participants and 1,240 test responses (20 per participant).
- Conditions: 27 counterfactual, 16 attribution, and 19 no-XAI participants.
- Waves: the original 36 and the new 26 participants were retained as strata.
- Treatment was assigned between participants. Participant-level averages and
  matched within-participant contrasts are therefore the inferential units; the
  1,240 rows were not treated as independent observations.
- Primary p-values are participant-label permutation tests within wave. Effects
  include participant bootstrap intervals and Hedges g. Broad searches include
  Benjamini-Hochberg FDR columns.

## 1. Average performance: no reliable condition benefit

Participant-level condition means were:

| Outcome | Attribution | Counterfactual | None |
|---|---:|---:|---:|
| Success rate | .538 | .474 | .547 |
| Boundary distance after edit (lower is better) | .117 | .128 | .137 |
| Boundary improvement | .006 | -.005 | -.014 |
| Target-confidence gain | .528 | .458 | .527 |
| L1 edit distance | .461 | .522 | .549 |
| Target-class centroid progress | .081 | .020 | .050 |
| Target-helpful SHAP change | .269 | .272 | .329 |
| Agreement with CF-taught directions | .812 | .707 | .841 |

None of the pairwise performance differences survived correction. There are
weak descriptive tendencies for attribution to finish nearer the boundary than
control, and for counterfactual to have lower success/confidence than both other
conditions, but the participant-level uncertainty is too large for an average
effect claim.

The broad scan included 27 outcomes across three comparisons. Only one test was
nominally below .05 and none had global q < .10. Ten key outcomes were then
tested across target, difficulty, training distance, seen/novel pair, nearest
demo-pair match, design cluster, prediction confidence, control-defined
difficulty, Age-pair, and exogenous geometry moderators. Four interactions were
nominally below .05 and none had global q < .10. Of 600 instance tests, nine
were nominally below .05 and none survived global correction.

## 2. Counterfactual XAI produces a more diffuse edit pattern

The clearest condition-level response difference is counterfactual versus
attribution:

| Metric | Counterfactual | Attribution | Difference | Hedges g | permutation p |
|---|---:|---:|---:|---:|---:|
| Mean attributes changed | 2.139 | 1.666 | +.473 | .73 | .0215 |
| Responses changing 3+ attributes | 33.0% | 12.2% | +20.8 pp | .83 | .0108 |
| Responses changing Insulin | 50.9% | 27.5% | +23.4 pp | .81 | .0161 |
| Feature-selection entropy | .814 | .740 | +.074 | .68 | .0354 |
| L1 edit distance | .522 | .461 | +.062 | .24 | .4833 |

The result is breadth rather than extremity: total L1 movement is similar, and
the mean movement per changed attribute is slightly smaller in the
counterfactual arm. A treatment-blind PCA of edit structure found the same
dimension (more features, more entropy, less modal-subset repetition); PC1
explained 38.4% of structural variance and differed between counterfactual and
attribution by 1.24 score units (g=.70, permutation p=.0318).

The direction is consistent across both collection waves:

- Mean-number-of-features difference: +.480 in the original wave and +.446 in
  the new wave.
- 3+-feature difference: +19.0 and +22.5 percentage points.
- Insulin-selection difference: +11.3 and +42.5 percentage points.

These focused structural results are hypothesis-generating rather than
confirmatory. Their FDR q within the 16-metric counterfactual-versus-attribution
family is .115 for the first three measures. The consistent wave directions and
large effects make them better next-study targets than the average success
outcomes.

### Interpretation

The v2.0 training set repeats two two-feature counterfactual templates six times
each: Glucose/BMI and Blood Pressure/Insulin. Age is never demonstrated. The
counterfactual display may therefore teach participants that several attributes
are jointly actionable, with particularly strong transfer to Insulin. It does
not appear to teach when a small subset is sufficient. This can create more
distributed editing without better decision-boundary movement.

## 3. A specific high-global-relevance counterfactual phenotype

The frozen cognitive model v0.1 defines alpha as the balance between global
learned relevance (alpha=1) and instance-local opposition (alpha=0) within the
feature-contribution family. Define the high-global phenotype as winning the
feature-contribution family with alpha=1.

Prevalence was:

- Counterfactual: 10/27 (37.0%).
- Attribution: 2/16 (12.5%).
- None: 1/19 (5.3%).

Counterfactual versus none gives OR=10.59, Fisher p=.0156 and q=.0468 across the
three condition contrasts. Stratifying by wave gives a common OR=12.65,
Mantel-Haenszel p=.0123, with no detected odds heterogeneity across waves
(p=.578). The direction appears in both waves: 7/15 versus 1/11 in the original
wave, and 3/12 versus 0/8 in the new wave. Counterfactual versus attribution is
directionally similar but underpowered (common OR=4.09, p=.0799).

Within the counterfactual arm, the 10 high-global participants differ from the
other 17 as follows:

| Outcome | High global | Other CF | Difference | FDR q across 14 outcomes |
|---|---:|---:|---:|---:|
| Cosine with nearest applicable CF demonstration | .053 | .444 | -.392 | .0007 |
| Agreement with CF-taught directions | .447 | .859 | -.412 | .0007 |
| Progress toward target-class training centroid | -.187 | .141 | -.328 | .0121 |
| Demo-feature Jaccard overlap | .289 | .415 | -.126 | .0133 |
| Target-helpful SHAP change | .181 | .325 | -.144 | .0347 |
| Confidence gain | .357 | .518 | -.162 | .0516 |
| Success | 36.5% | 53.8% | -17.3 pp | .0536 |

Every listed difference has the same direction in both waves. An unsupervised
edit-pattern clustering offers convergent characterization: 5/10 high-global CF
participants, but only 1/17 other CF participants, fall in the poorer-response
cluster (OR=16.0, Fisher p=.0152).

### Important limitation

This is not an independently validated subgroup. Model family and alpha were
fitted from the same test edit vectors, and the unsupervised cluster also uses
those edit patterns. The condition-prevalence result is worth pursuing, but the
within-subgroup outcome differences are post-hoc and partially endogenous. A
next study should classify strategy on one set of trials and test performance on
a held-out set.

### Mechanistic reading

This phenotype is more specific than “lost participants.” It is compatible
with learning *which attributes are generally important* from the repeated
counterfactuals, but failing to learn the local sign and amount required by a
new instance. It also explains why the overall CF direction-alignment mean is
low: excluding this phenotype, the remaining CF participants have direction
agreement .859, close to or above the other arms.

## 4. Instance and transfer clues

No instance effect survived the full instance search, but several nominal
effects were directionally replicated in both waves and can inform stimulus
selection:

- Instance 160210: attribution finished .068 nearer the boundary than no XAI
  (means .085 versus .153, permutation p=.0385). This is a No-Diabetes target
  whose Blood Pressure/Insulin ideal pair matches its nearest displayed demo.
- Instance 160216: attribution produced .237 more target-centroid progress than
  no XAI (p=.0325), with the same direction in both waves. The ideal BMI/Insulin
  pair differs from the nearest displayed Blood Pressure/Insulin demo.
- Instance 160203: counterfactual responses were much less aligned with the
  nearest matching BMI/Glucose demo than attribution responses (cosine .251
  versus .574, p=.0200), and gained .191 less target confidence (p=.0475).
- Instance 160217: counterfactual responses overlapped more with the nearest
  Blood Pressure/Insulin demo than attribution responses (Jaccard difference
  +.150, p=.0485), even though the test case's ideal pair is Age/Glucose. Success
  was descriptively lower in CF (.222) than attribution (.312) or none (.368).
  This is a concrete over-transfer candidate.
- Instance 160201: the test ideal is Blood Pressure/Insulin, while its nearest
  displayed same-class demo uses BMI/Glucose. Counterfactual responses had much
  less demo-feature overlap than controls (difference -.243, p=.0030), with the
  same direction in both waves. This illustrates that nearest-profile and
  relevant-feature similarity are separate dimensions.

The matched moderator analysis also gives a weak clue that both XAI conditions
reduce the change in target-centroid progress between originally closer- and
farther-from-boundary cases relative to no XAI (nominal interaction p=.043 for
attribution and .047 for counterfactual). This did not survive correction.

## 5. Clustering and distance results

- Participant edit-pattern clustering selected k=2 among solutions with at
  least six participants per cluster. The clusters were behaviorally distinct
  (success .418 versus .558; target-centroid progress -.048 versus .089), but
  separation was modest (silhouette .194; feature-bootstrap ARI .459).
- Overall cluster membership was not convincingly associated with condition
  (permutation p=.165).
- Median splits and participant-specific slopes based on distance to the nearest
  displayed same-class training instance did not show an FDR-robust condition
  effect.
- Seen versus novel ideal feature pair, nearest-demo-pair match, target label,
  original boundary distance, design cluster, prediction confidence, empirical
  control difficulty, Age-pair status, and an exogenous geometry cluster all
  failed broad correction.

The practical lesson is not that geometry is irrelevant. It is that a single
nearest-neighbor distance collapses too much: source-label match, feature-pair
match, sign match, and distance to the decision boundary should be modeled
separately.

## Recommended next experiment

### Highest priority

1. **Confirm edit diffusion.** Pre-register participant mean attributes changed
   and 3+-feature rate as co-primary behavioral endpoints, with success and
   boundary improvement as effectiveness endpoints. Target the CF-versus-
   attribution contrast directly.
2. **Break the repeated-feature confound.** Counterbalance which features appear
   in demonstrations. Give every actionable feature equal exposure, or randomly
   assign training templates independently of test relevance. The current 6+6
   two-pair design and absent Age make an Insulin-transfer effect difficult to
   separate from counterfactual format.
3. **Teach conditionality explicitly.** Add a short message such as: “This is one
   valid change for this case; a different case may require different features
   or directions.” Compare standard CF, conditionality-annotated CF, and
   attribution.
4. **Cross-fit the strategy phenotype.** Fit family/alpha on odd-numbered or first
   half trials; test direction agreement, centroid progress, confidence, and
   success on the held-out half. Reverse the split as a robustness check. Do not
   define and evaluate high-alpha on the same trials.
5. **Build balanced transfer cells.** Select enough cases in each preregistered
   cell: nearest-demo pair exact match, one-feature overlap, and no overlap;
   crossed with near/far training-profile distance and both target directions.
   The current exact-match group has only six cases and the Age-pair group only
   three.

### Measurement additions

6. After training, ask participants which statement they learned: a global rule
   (“these features are generally important”), an exemplar rule (“cases like
   this use these changes”), or a local rule (“direction depends on the current
   values”). This provides an independent manipulation check for the high-alpha
   interpretation.
7. Record the order in which sliders are changed, reversals, and time to first
   edit. Diffuse editing could arise from planned joint changes or from serial
   trial-and-error; final values cannot distinguish them.
8. Add confidence in the chosen edit and perceived similarity to the most
   relevant training case. This tests whether over-transfer is deliberate.

### Additional metrics and models to try

9. Compare each test origin and endpoint with both the 12 displayed cases and
   the full model-training set using range-scaled L1, standardized Euclidean,
   Mahalanobis/Gower distance, local density, and nearest-neighbor label purity.
10. Use distance to the nearest *same-label origin*, nearest *target-label
    origin*, nearest CF endpoint, and target-class centroid as separate variables.
11. Score feature-pair exact match, partial overlap, sign agreement, projection
    onto the demo vector, residual orthogonal movement, and Jensen-Shannon
    divergence from the demonstrated feature-frequency distribution.
12. Add edit-efficiency outcomes: target-confidence gain per L1 movement,
    centroid progress per L1, boundary progress divided by original boundary
    distance, and overshoot conditional on successful crossing. Winsorize ratios
    or use robust regression because near-zero denominators are unstable.
13. Fit a crossed participant/instance mixed model or GEE for each preregistered
    endpoint, with condition by transfer-cell interactions. Retain participant-
    level randomization inference as the primary robustness check.
14. Freeze the current treatment-blind diffuse-edit PC1 loadings and test that
    single composite prospectively. Do not re-estimate and test the loadings in
    the confirmatory sample.
15. Treat latent classes as descriptive unless they replicate. The current
    cluster stability is too weak for confirmatory subgroup labels.

## Sample-size implication

The observed mean-number-of-features contrast has g about .73. A simple balanced
two-group design requires about 31 participants per arm for 80% power at
two-sided alpha .05, or about 46 per arm at alpha .01. The raw 33% versus 12%
3+-feature proportions would require roughly 59 per arm if analyzed as one
binary observation per participant. A practical next iteration is therefore at
least 45 per key XAI arm, preferably about 60 if the binary response-structure
endpoint is primary. These are planning approximations based on exploratory
effects and should be inflated for exclusions and shrinkage.

## Output map

- `overall_condition_effects.csv`: broad participant-level effects.
- `response_structure_condition_effects.csv`: focused structural comparisons.
- `response_structure_wave_effects.csv`: wave replication.
- `high_global_strategy_prevalence.csv`: condition association and stratified
  odds ratios.
- `high_global_strategy_outcomes_within_counterfactual.csv`: post-hoc phenotype
  outcomes.
- `moderator_interactions.csv` and `moderator_subgroup_effects.csv`: geometry and
  transfer tests.
- `instance_specific_effects.csv`: case-level scan with wave-direction flags.
- `participant_cluster_*`: edit-pattern clustering and diagnostics.
- `diffuse_edit_pca_*`: treatment-blind structural composite.
- `trend_and_distance_slope_effects.csv`: trial, boundary, and training-distance
  slopes.
- `testing_rows_enriched.csv`, `participant_summary.csv`, and
  `instance_metadata_and_clusters.csv`: auditable analysis datasets.

