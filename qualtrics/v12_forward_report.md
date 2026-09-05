# v0.9 perfect-memory mental-model fit

## Method

Each participant's 10 forward-simulation responses was fitted prequentially. On each trial, exemplar, attribution, and prototype models predicted the participant's response using only earlier feedback. The current correct AI label and explanation were learned only after scoring that response.

This is the reviewed perfect-memory baseline: all encountered instances are retained, there is no decay or retrieval threshold, and no continuous parameter is estimated per participant. All three models use the same range-normalized feature relevance vector, learned from class diagnosticity (no XAI), displayed absolute attribution strength (attribution XAI), or changed-feature frequency (counterfactual XAI).

Primary fit is negative log likelihood (NLL) of the participant's chosen labels. Because every model has zero fitted continuous parameters, BIC = 2 x NLL. Models within delta BIC < 2 are reported as empirically indistinguishable; the strict minimum is retained for sorting but should not be treated as decisive in those cases.

## Overall results

| Mental model | Strict best | Supported (ΔBIC < 2) | Mean NLL | Mean accuracy |
| --- | --- | --- | --- | --- |
| exemplar | 8 | 23 | 8.222 | 0.552 |
| attribution | 13 | 23 | 8.103 | 0.608 |
| prototype | 3 | 23 | 8.263 | 0.566 |

Fit identifiability: 0 participants have one supported model, 3 have two, and 21 have all three within ΔBIC < 2.

## Strict best model by domain and explanation

| Domain | XAI | N | Exemplar | Attribution | Prototype | One supported |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | attribution | 9 | 2 | 4 | 3 | 0 |
| diabetes | counterfactual | 8 | 1 | 7 | 0 | 0 |
| diabetes | none | 7 | 5 | 2 | 0 | 0 |

## Participant-level interpretation

The participant CSV contains the strict best model, all models within ΔBIC < 2, fit strength, and the NLL/BIC, Brier score, and hard-choice accuracy for every candidate. The trial CSV contains the probability assigned to every observed response, allowing the fit to be audited trial by trial.

The first trial is necessarily an equal 0.5 prediction for all models because no feedback has yet been observed; this common term cancels in model comparison.

This comparison concerns forward-simulation responses only. It does not infer which counterfactual editing strategy generated participants' later edits.
