# Cognitive model v0.2 fit report

## Analysis specification

The analysis includes 42 complete participants (12 ordered training exposures and 20 testing edits each). The selection distribution is conditioned on the interface-permitted set: [['blood_pressure'], ['insulin'], ['bmi'], ['age']]. The previously screened cohort contains 42 participants. Base prototype and exemplar models are available in every condition; additive models are available after Attribution or Counterfactual XAI; counterfactual-edit models are available only after Counterfactual XAI.

Feature probabilities are conditioned on the subsets actually permitted by the interface. Counterfactual memory is consolidated separately for each feature; edit frequency, conditional signed change, and conditional endpoint are retained rather than treating displayed feature pairs as atomic memories. The global feature-level lapse was selected by pooled five-fold cross-validated subset negative log likelihood. Each remaining parameter was fitted per participant and variant. Model selection uses cross-validated exact-subset NLL, with conditional amount MAE as a tie-breaker. Amount MAE is evaluated on the attributes the participant actually selected, keeping feature selection and edit magnitude diagnostically separate.

## Global lapse

Selected lapse: **0.400**. For each feature, lapse transforms its core inclusion probability as p'=(1-lapse)p+lapse/2 before the independent feature probabilities are multiplied and conditioned on the permitted selection space. Its uniform baseline has NLL 1.386.

## Variant-level cross-validated fit

| Variant | Available N | Winners | Subset NLL | Expected F1 | MAP F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| additive_exemplar | 27 | 3 | 1.329 | 0.297 | 0.269 | 0.194 | 0.696 |
| additive_prototype | 27 | 4 | 1.332 | 0.287 | 0.263 | 0.203 | 0.707 |
| base_exemplar | 42 | 10 | 1.312 | 0.302 | 0.327 | 0.196 | 0.767 |
| base_prototype | 42 | 20 | 1.300 | 0.294 | 0.342 | 0.198 | 0.769 |
| counterfactual_exemplar | 12 | 4 | 1.231 | 0.330 | 0.412 | 0.136 | 0.775 |
| counterfactual_prototype | 12 | 1 | 1.264 | 0.314 | 0.433 | 0.134 | 0.775 |

## Selected-model performance

Across participants, selected models achieved mean subset NLL 1.225, expected selection F1 0.324, MAP selection F1 0.361, exact-subset accuracy 0.361, conditional amount MAE 0.188, and conditional direction accuracy 0.776.

Winner counts: additive_exemplar=3, additive_prototype=4, base_exemplar=10, base_prototype=20, counterfactual_exemplar=4, counterfactual_prototype=1

The selected models improve mean subset NLL over the uniform baseline by 0.162; 32/42 participants improve (Wilcoxon p=0.000). Conditional amount MAE improves over predicting zero movement by 0.106; 36/42 participants improve (Wilcoxon p=0.000).

## Fit by explanation condition

| XAI | N | Selected variants | Subset NLL | Expected F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| attribution | 15 | additive_prototype=2, base_exemplar=6, base_prototype=7 | 1.293 | 0.294 | 0.228 | 0.740 |
| counterfactual | 12 | additive_exemplar=3, additive_prototype=2, base_exemplar=1, base_prototype=1, counterfactual_exemplar=4, counterfactual_prototype=1 | 1.106 | 0.374 | 0.138 | 0.788 |
| none | 15 | base_exemplar=3, base_prototype=12 | 1.252 | 0.313 | 0.189 | 0.803 |

## Prototype versus exemplar

Negative differences favor exemplars; positive differences favor prototypes.

| Representation | Paired N | Exemplar - prototype NLL | p | Exemplar - prototype amount MAE | p |
| --- | --- | --- | --- | --- | --- |
| base | 42 | +0.012 | 0.411 | -0.002 | 0.421 |
| additive | 27 | -0.003 | 0.989 | -0.009 | 0.030 |
| counterfactual | 12 | -0.033 | 0.169 | +0.002 | 0.721 |

## Interpretation limits

This is an exploratory model comparison. The global lapse is cross-validated over trials but selected using this cohort, so it should be frozen before confirmatory testing on a new cohort. Aggregate selected-model performance has some model-selection optimism because each participant's winning strategy is chosen from the same cross-validation estimates being summarized; the fixed-variant results are the cleaner descriptive comparison. Strategy assignments indicate which implemented process predicts held-out edits most closely; they do not establish that a participant exclusively used that strategy. Prototype and exemplar models overlap when distance sensitivity is zero, and the counterfactual-derived additive representation is an inverse-change approximation rather than a displayed attribution.
