# Cognitive model v0.2 fit report

## Analysis specification

The analysis includes 60 complete participants (12 ordered training exposures and 20 testing edits each). The previously screened cohort contains 57 participants. Base prototype and exemplar models are available in every condition; additive models are available after Attribution or Counterfactual XAI; counterfactual-edit models are available only after Counterfactual XAI.

Every feature is sampled independently and the resulting Bernoulli-product distribution is conditioned on at least one edit. Counterfactual memory is consolidated separately for each feature; edit frequency, conditional signed change, and conditional endpoint are retained rather than treating displayed feature pairs as atomic memories. The global feature-level lapse was selected by pooled five-fold cross-validated subset negative log likelihood. Each remaining parameter was fitted per participant and variant. Model selection uses cross-validated exact-subset NLL, with conditional amount MAE as a tie-breaker. Amount MAE is evaluated on the attributes the participant actually selected, keeping feature selection and edit magnitude diagnostically separate.

## Global lapse

Selected lapse: **0.400**. For each feature, lapse transforms its core inclusion probability as p'=(1-lapse)p+lapse/2 before the independent feature probabilities are multiplied and conditioned on a nonempty subset. The uniform baseline over the 31 nonempty feature subsets has NLL 3.434.

## Variant-level cross-validated fit

| Variant | Available N | Winners | Subset NLL | Expected F1 | MAP F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| additive_exemplar | 37 | 2 | 3.008 | 0.375 | 0.419 | 0.313 | 0.536 |
| additive_prototype | 37 | 12 | 2.916 | 0.380 | 0.429 | 0.313 | 0.534 |
| base_exemplar | 60 | 2 | 3.118 | 0.350 | 0.272 | 0.319 | 0.528 |
| base_prototype | 60 | 30 | 3.091 | 0.352 | 0.267 | 0.319 | 0.525 |
| counterfactual_exemplar | 19 | 2 | 2.816 | 0.417 | 0.555 | 0.230 | 0.680 |
| counterfactual_prototype | 19 | 12 | 2.773 | 0.423 | 0.581 | 0.217 | 0.709 |

## Selected-model performance

Across participants, selected models achieved mean subset NLL 2.907, expected selection F1 0.389, MAP selection F1 0.415, exact-subset accuracy 0.136, conditional amount MAE 0.297, and conditional direction accuracy 0.576.

Winner counts: additive_exemplar=2, additive_prototype=12, base_exemplar=2, base_prototype=30, counterfactual_exemplar=2, counterfactual_prototype=12

The selected models improve mean subset NLL over the uniform baseline by 0.527; 52/60 participants improve (Wilcoxon p=0.000). Conditional amount MAE improves over predicting zero movement by 0.013; 26/60 participants improve (Wilcoxon p=0.994).

## Fit by explanation condition

| XAI | N | Selected variants | Subset NLL | Expected F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| attribution | 18 | additive_exemplar=1, additive_prototype=9, base_exemplar=1, base_prototype=7 | 2.924 | 0.337 | 0.329 | 0.543 |
| counterfactual | 19 | additive_exemplar=1, additive_prototype=3, base_prototype=1, counterfactual_exemplar=2, counterfactual_prototype=12 | 2.751 | 0.430 | 0.245 | 0.679 |
| none | 23 | base_exemplar=1, base_prototype=22 | 3.022 | 0.396 | 0.314 | 0.518 |

## Prototype versus exemplar

Negative differences favor exemplars; positive differences favor prototypes.

| Representation | Paired N | Exemplar - prototype NLL | p | Exemplar - prototype amount MAE | p |
| --- | --- | --- | --- | --- | --- |
| base | 60 | +0.027 | 0.000 | -0.001 | 0.327 |
| additive | 37 | +0.091 | 0.000 | -0.000 | 0.695 |
| counterfactual | 19 | +0.043 | 0.028 | +0.013 | 0.114 |

## Interpretation limits

This is an exploratory model comparison. The global lapse is cross-validated over trials but selected using this cohort, so it should be frozen before confirmatory testing on a new cohort. Aggregate selected-model performance has some model-selection optimism because each participant's winning strategy is chosen from the same cross-validation estimates being summarized; the fixed-variant results are the cleaner descriptive comparison. Strategy assignments indicate which implemented process predicts held-out edits most closely; they do not establish that a participant exclusively used that strategy. Prototype and exemplar models overlap when distance sensitivity is zero, and the counterfactual-derived additive representation is an inverse-change approximation rather than a displayed attribution.
