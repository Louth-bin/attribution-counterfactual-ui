# Cognitive model v0.2 fit report

## Analysis specification

The analysis includes 42 complete participants (12 ordered training exposures and 20 testing edits each). The previously screened cohort contains 42 participants. Base prototype and exemplar models are available in every condition; additive models are available after Attribution or Counterfactual XAI; counterfactual-edit models are available only after Counterfactual XAI.

Every feature is sampled independently and the resulting Bernoulli-product distribution is conditioned on at least one edit. Counterfactual memory is consolidated separately for each feature; edit frequency, conditional signed change, and conditional endpoint are retained rather than treating displayed feature pairs as atomic memories. The global feature-level lapse was selected by pooled five-fold cross-validated subset negative log likelihood. Each remaining parameter was fitted per participant and variant. Model selection uses cross-validated exact-subset NLL, with conditional amount MAE as a tie-breaker. Amount MAE is evaluated on the attributes the participant actually selected, keeping feature selection and edit magnitude diagnostically separate.

## Global lapse

Selected lapse: **0.010**. For each feature, lapse transforms its core inclusion probability as p'=(1-lapse)p+lapse/2 before the independent feature probabilities are multiplied and conditioned on a nonempty subset. The uniform baseline over the 31 nonempty feature subsets has NLL 3.434.

## Variant-level cross-validated fit

| Variant | Available N | Winners | Subset NLL | Expected F1 | MAP F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| additive_exemplar | 27 | 1 | 1.824 | 0.221 | 0.219 | 0.193 | 0.713 |
| additive_prototype | 27 | 1 | 1.846 | 0.213 | 0.209 | 0.202 | 0.711 |
| base_exemplar | 42 | 10 | 1.422 | 0.297 | 0.389 | 0.192 | 0.773 |
| base_prototype | 42 | 28 | 1.423 | 0.293 | 0.387 | 0.197 | 0.768 |
| counterfactual_exemplar | 12 | 1 | 1.584 | 0.251 | 0.375 | 0.138 | 0.775 |
| counterfactual_prototype | 12 | 1 | 1.603 | 0.242 | 0.317 | 0.137 | 0.771 |

## Selected-model performance

Across participants, selected models achieved mean subset NLL 1.397, expected selection F1 0.302, MAP selection F1 0.402, exact-subset accuracy 0.402, conditional amount MAE 0.192, and conditional direction accuracy 0.776.

Winner counts: additive_exemplar=1, additive_prototype=1, base_exemplar=10, base_prototype=28, counterfactual_exemplar=1, counterfactual_prototype=1

The selected models improve mean subset NLL over the uniform baseline by 2.037; 42/42 participants improve (Wilcoxon p=0.000). Conditional amount MAE improves over predicting zero movement by 0.102; 34/42 participants improve (Wilcoxon p=0.000).

## Fit by explanation condition

| XAI | N | Selected variants | Subset NLL | Expected F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| attribution | 15 | additive_exemplar=1, base_exemplar=6, base_prototype=8 | 1.445 | 0.287 | 0.235 | 0.727 |
| counterfactual | 12 | additive_prototype=1, base_exemplar=4, base_prototype=5, counterfactual_exemplar=1, counterfactual_prototype=1 | 1.374 | 0.312 | 0.147 | 0.800 |
| none | 15 | base_prototype=15 | 1.369 | 0.308 | 0.186 | 0.807 |

## Prototype versus exemplar

Negative differences favor exemplars; positive differences favor prototypes.

| Representation | Paired N | Exemplar - prototype NLL | p | Exemplar - prototype amount MAE | p |
| --- | --- | --- | --- | --- | --- |
| base | 42 | -0.001 | 0.496 | -0.005 | 0.090 |
| additive | 27 | -0.022 | 0.454 | -0.010 | 0.022 |
| counterfactual | 12 | -0.019 | 1.000 | +0.002 | 0.575 |

## Interpretation limits

This is an exploratory model comparison. The global lapse is cross-validated over trials but selected using this cohort, so it should be frozen before confirmatory testing on a new cohort. Aggregate selected-model performance has some model-selection optimism because each participant's winning strategy is chosen from the same cross-validation estimates being summarized; the fixed-variant results are the cleaner descriptive comparison. Strategy assignments indicate which implemented process predicts held-out edits most closely; they do not establish that a participant exclusively used that strategy. Prototype and exemplar models overlap when distance sensitivity is zero, and the counterfactual-derived additive representation is an inverse-change approximation rather than a displayed attribution.
