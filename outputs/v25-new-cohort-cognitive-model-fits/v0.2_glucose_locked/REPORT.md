# Cognitive model v0.2 fit report

## Analysis specification

The analysis includes 21 complete participants (12 ordered training exposures and 20 testing edits each). The selection distribution is conditioned on the interface-permitted set: [['blood_pressure'], ['insulin'], ['bmi'], ['age']]. The previously screened cohort contains 21 participants. Base prototype and exemplar models are available in every condition; additive models are available after Attribution or Counterfactual XAI; counterfactual-edit models are available only after Counterfactual XAI.

Feature probabilities are conditioned on the subsets actually permitted by the interface. Counterfactual memory is consolidated separately for each feature; edit frequency, conditional signed change, and conditional endpoint are retained rather than treating displayed feature pairs as atomic memories. The global feature-level lapse was selected by pooled five-fold cross-validated subset negative log likelihood. Each remaining parameter was fitted per participant and variant. Model selection uses cross-validated exact-subset NLL, with conditional amount MAE as a tie-breaker. Amount MAE is evaluated on the attributes the participant actually selected, keeping feature selection and edit magnitude diagnostically separate.

## Global lapse

Selected lapse: **0.500**. For each feature, lapse transforms its core inclusion probability as p'=(1-lapse)p+lapse/2 before the independent feature probabilities are multiplied and conditioned on the permitted selection space. Its uniform baseline has NLL 1.386.

## Variant-level cross-validated fit

| Variant | Available N | Winners | Subset NLL | Expected F1 | MAP F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| additive_exemplar | 14 | 1 | 1.364 | 0.270 | 0.204 | 0.207 | 0.768 |
| additive_prototype | 14 | 3 | 1.358 | 0.271 | 0.189 | 0.204 | 0.757 |
| base_exemplar | 21 | 4 | 1.343 | 0.284 | 0.324 | 0.220 | 0.767 |
| base_prototype | 21 | 11 | 1.329 | 0.283 | 0.336 | 0.214 | 0.779 |
| counterfactual_exemplar | 5 | 0 | 1.279 | 0.312 | 0.360 | 0.167 | 0.710 |
| counterfactual_prototype | 5 | 2 | 1.277 | 0.310 | 0.400 | 0.165 | 0.700 |

## Selected-model performance

Across participants, selected models achieved mean subset NLL 1.279, expected selection F1 0.299, MAP selection F1 0.307, exact-subset accuracy 0.307, conditional amount MAE 0.215, and conditional direction accuracy 0.755.

Winner counts: additive_exemplar=1, additive_prototype=3, base_exemplar=4, base_prototype=11, counterfactual_prototype=2

The selected models improve mean subset NLL over the uniform baseline by 0.108; 15/21 participants improve (Wilcoxon p=0.001). Conditional amount MAE improves over predicting zero movement by 0.105; 19/21 participants improve (Wilcoxon p=0.000).

## Fit by explanation condition

| XAI | N | Selected variants | Subset NLL | Expected F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| attribution | 9 | additive_exemplar=1, additive_prototype=2, base_exemplar=2, base_prototype=4 | 1.303 | 0.285 | 0.216 | 0.744 |
| counterfactual | 5 | additive_prototype=1, base_exemplar=1, base_prototype=1, counterfactual_prototype=2 | 1.178 | 0.337 | 0.179 | 0.780 |
| none | 7 | base_exemplar=1, base_prototype=6 | 1.320 | 0.289 | 0.240 | 0.750 |

## Prototype versus exemplar

Negative differences favor exemplars; positive differences favor prototypes.

| Representation | Paired N | Exemplar - prototype NLL | p | Exemplar - prototype amount MAE | p |
| --- | --- | --- | --- | --- | --- |
| base | 21 | +0.014 | 0.211 | +0.006 | 0.069 |
| additive | 14 | +0.007 | 0.130 | +0.003 | 0.790 |
| counterfactual | 5 | +0.003 | 1.000 | +0.001 | 0.715 |

## Interpretation limits

This is an exploratory model comparison. The global lapse is cross-validated over trials but selected using this cohort, so it should be frozen before confirmatory testing on a new cohort. Aggregate selected-model performance has some model-selection optimism because each participant's winning strategy is chosen from the same cross-validation estimates being summarized; the fixed-variant results are the cleaner descriptive comparison. Strategy assignments indicate which implemented process predicts held-out edits most closely; they do not establish that a participant exclusively used that strategy. Prototype and exemplar models overlap when distance sensitivity is zero, and the counterfactual-derived additive representation is an inverse-change approximation rather than a displayed attribution.
