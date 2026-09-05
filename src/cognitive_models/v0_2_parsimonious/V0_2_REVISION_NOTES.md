# v0.2 independent-feature revision notes

## Why this revision was made

The initial v0.2 fit showed three concrete mismatches. It selected Age much
more often than participants, rarely selected Insulin, and its counterfactual
prototype collapsed the demonstrated Glucose/Insulin and Blood-pressure/Age
pairs into an artificial four-feature average edit. With zero fitted decay, the
counterfactual prototype was also effectively invariant across current test
instances of the same target class.

The archived v0.1 ablation found held-out support for the additive edit margin,
the balance between remembered endpoints and remembered changes, exemplar
locality, and participant-specific Age actionability. v0.2 already contained
the margin and locality mechanisms. This revision carries forward the two
missing mechanisms without restoring v0.1's observed feature count K.

## Changes to verify

1. Counterfactual memories are consolidated feature by feature. For feature
   `j`, the model stores its activation-weighted edit frequency, its signed
   change conditional on having been edited, and its endpoint conditional on
   having been edited. Demonstrated feature pairs are not atomic choices.
2. `edit_blend` interpolates between the current-to-remembered-endpoint change
   and the remembered change itself. This explicitly replaces the previously
   vague phrase "delta needed."
3. `age_actionable` is a fitted binary selection parameter. When false, the
   core strategy assigns Age zero selection score; feature-level lapse can
   still give an anomalous Age edit nonzero probability.
4. Lapse acts independently on each feature inclusion probability, rather than
   mixing the completed subset distribution with a uniform distribution.
5. Trial output now records `feature_scores`, `base_delta`, `edit_blend`, and
   `age_actionable` so every held-out decision can be inspected directly.

## Deliberately not restored from v0.1

The participant's observed number of edited features is not supplied to the
model. v0.2 continues to predict both cardinality and feature identity. The
condition-specific representation families also remain explicit rather than
being blended by v0.1's explanation-reliance parameter.
