# Cognitive model v0.2

This version adds sequential full-state memory while preserving the v0.1 fitting
contract.

## Representation

After each ordered training exposure, the model stores a complete state:

- the current class midpoint and polarity;
- the complete ALCOVE attention-logit vector learned so far;
- the current training profile and its counterfactual endpoint.

It never reconstructs attention by summing remembered updates. Attention is
updated sequentially from the gradient of exemplar-category prediction error
plus an explanation-attention cross-entropy term. `attention rate` controls the
update size and `xai weight` controls the explanation term. One fitted
`specificity` parameter controls similarity both during ALCOVE learning and
during later exemplar/counterfactual retrieval; there are no separate `c` and
`lambda` parameters.

## Compared variants

- `perfect memory`: the feature model always uses the final complete state; the
  exemplar model can retrieve every eligible example without recency decay.
- `memory variation`: complete states or examples receive normalized retrieval
  weights from similarity and fitted power-law recency decay.

Retrieval produces one activation-weighted state or edit vector. Feature
selection and conditional edit magnitude are calculated only after this
blending operation; the model does not average errors from separate single
retrievals.

Retrieval refresh is deliberately disabled in this first comparison. This keeps
per-trial likelihoods and cross-validation exact. Adding sampled refresh would
require repeated Monte Carlo fits or a latent-history model.

## Preserved from v0.1

1. The participant's observed number of changed features, K, is retained.
2. Every eligible K-feature subset is enumerated exactly.
3. Selection parameters are fitted by subset negative log likelihood.
4. Additive rho is fitted afterward, conditional on the participant's observed
   feature subset.
5. The counterfactual exemplar strategy retains beta's mixture of copying the
   remembered endpoint and copying the demonstrated edit.
6. Model selection uses five-fold cross-validation.

Compared with v0.1, both perfect-memory families add fitted `attention rate`.
The feature family also adds `specificity`, which v0.1 used only for its
exemplar family. Explanation conditions replace v0.1's explanation-balance
parameter with fitted `xai weight`. The memory-variation variant additionally
fits decay.

## Run

```text
python -m src.cognitive_models.v0_2.fit
```

Outputs:

- `fits.csv`: both families under both memory variants;
- `memory_comparison.csv`: participant-level selected-family comparison;
- `summary.json`: aggregate comparison.
