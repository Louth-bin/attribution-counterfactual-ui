# Cognitive model v0.2 fit report

## Analysis specification

The analysis includes 1 complete participants (12 ordered training exposures and 20 testing edits each). The previously screened cohort contains 1 participants. Base prototype and exemplar models are available in every condition; additive models are available after Attribution or Counterfactual XAI; counterfactual-edit models are available only after Counterfactual XAI.

Every feature is sampled independently and the resulting Bernoulli-product distribution is conditioned on at least one edit. Counterfactual memory is consolidated separately for each feature; edit frequency, conditional signed change, and conditional endpoint are retained rather than treating displayed feature pairs as atomic memories. The global feature-level lapse was selected by pooled five-fold cross-validated subset negative log likelihood. Each remaining parameter was fitted per participant and variant. Model selection uses cross-validated exact-subset NLL, with conditional amount MAE as a tie-breaker. Amount MAE is evaluated on the attributes the participant actually selected, keeping feature selection and edit magnitude diagnostically separate.

## Global lapse

Selected lapse: **0.010**. For each feature, lapse transforms its core inclusion probability as p'=(1-lapse)p+lapse/2 before the independent feature probabilities are multiplied and conditioned on a nonempty subset. The uniform baseline over the 31 nonempty feature subsets has NLL 1.386.

## Variant-level cross-validated fit

| Variant | Available N | Winners | Subset NLL | Expected F1 | MAP F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_exemplar | 1 | 0 | 1.301 | 0.304 | 0.350 | 0.165 | 0.800 |
| base_prototype | 1 | 1 | 1.284 | 0.307 | 0.400 | 0.167 | 0.800 |

## Selected-model performance

Across participants, selected models achieved mean subset NLL 1.284, expected selection F1 0.307, MAP selection F1 0.400, exact-subset accuracy 0.400, conditional amount MAE 0.167, and conditional direction accuracy 0.800.

Winner counts: base_prototype=1

The selected models improve mean subset NLL over the uniform baseline by 0.102; 1/1 participants improve (Wilcoxon p=1.000). Conditional amount MAE improves over predicting zero movement by 0.138; 1/1 participants improve (Wilcoxon p=1.000).

## Fit by explanation condition

| XAI | N | Selected variants | Subset NLL | Expected F1 | Amount MAE | Direction accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| none | 1 | base_prototype=1 | 1.284 | 0.307 | 0.167 | 0.800 |

## Prototype versus exemplar

Negative differences favor exemplars; positive differences favor prototypes.

| Representation | Paired N | Exemplar - prototype NLL | p | Exemplar - prototype amount MAE | p |
| --- | --- | --- | --- | --- | --- |
| base | 1 | +0.016 | 1.000 | -0.002 | 1.000 |

## Interpretation limits

This is an exploratory model comparison. The global lapse is cross-validated over trials but selected using this cohort, so it should be frozen before confirmatory testing on a new cohort. Aggregate selected-model performance has some model-selection optimism because each participant's winning strategy is chosen from the same cross-validation estimates being summarized; the fixed-variant results are the cleaner descriptive comparison. Strategy assignments indicate which implemented process predicts held-out edits most closely; they do not establish that a participant exclusively used that strategy. Prototype and exemplar models overlap when distance sensitivity is zero, and the counterfactual-derived additive representation is an inverse-change approximation rather than a displayed attribution.
