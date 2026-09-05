# Perfect-memory cognitive-model baseline

This folder contains a minimal, testable account of how participants may learn
an AI's classifications and use that knowledge to construct counterfactual
edits. It is inspired by the ACT-R memory assumptions and the CoAX and CoXAM
models discussed in Chapters 3 and 4 of the attached thesis proposal, but this
first baseline deliberately assumes perfect memory.

The primary implementation is `PerfectMemoryCognitiveModel`. The older
`MinimalCognitiveModel` is retained as an experimental decay-based version; it
is not used in the v0.9 participant fits.

## Baseline commitments and parameters

Every labeled profile is retained and is always available. There is no memory
decay, retrieval threshold, retrieval noise, response-time process, learning
rate, similarity-sensitivity parameter, or lapse parameter.

Consequently, the participant fit has:

- no fitted continuous parameters;
- one discrete candidate mental model, \(M\in\{E,A,P\}\), where \(E\) is
  exemplar, \(A\) attribution, and \(P\) prototype;
- optionally, a separate discrete counterfactual strategy \(S\), if later edits
  are modeled.

`max_changes` and a fixed edit amount are task/action settings, not memory or
mental-model parameters. They should only be fitted as cognitive parameters if
the experimental design does not already determine them.

## Terms and encoding

- \(x_r\): the observed value of feature \(r\).
- \(z_r\): its internal normalized representation.
- \(R\): the number of user-visible features.
- \(y\in\{-1,+1\}\): a binary class label.
- \(q_r\geq0\): the relevance of feature \(r\), shared by all three mental
  models, with \(\sum_r q_r=1\).
- \(p_r\in\{-1,0,+1\}\): attribution polarity: whether increasing the feature
  supports the first or second class. For a one-hot categorical feature, each
  encoded level has its own polarity.
- \(m_r\): the attribution model's middle or normal value.
- \(d_r(a,b)\): a range-normalized feature distance.
- \(\phi_r\): a displayed signed attribution.
- \(\mathbb 1[\cdot]\): one when its condition is true and zero otherwise.

For a numerical feature with experiment range \([L_r,U_r]\),

\[
z_r=\operatorname{clip}\left(\frac{x_r-L_r}{U_r-L_r},0,1\right),
\qquad
d_r(z_r,z'_r)=|z_r-z'_r|.
\]

Categorical features are one-hot encoded and have distance zero for a match and
one for a mismatch. No arbitrary ordering is imposed on their levels.

## Shared relevance

Relevance is not derived from the attribution model's signed weight. It is an
independent quantity used symmetrically by all three mental models. Only its
source differs across explanation conditions.

### No explanation

After both classes have been observed, class diagnosticity for feature \(r\) is

\[
a_r^{\text{none}}=
\frac{\lVert\boldsymbol\mu_{r,+}-\boldsymbol\mu_{r,-}\rVert_1}
{\lVert\boldsymbol\sigma_{r,+}+\boldsymbol\sigma_{r,-}\rVert_1+\epsilon}.
\]

Before both classes are available, all features receive equal relevance.

### Attribution explanation

For one displayed attribution explanation,

\[
a_r^{\text{attr}}=|\phi_r|.
\]

Only displayed features have nonzero signal in the study bundle. The signal is
normalized within a trial and then averaged over observed explanations.

### Counterfactual explanation

For an original profile \(\mathbf z\) and displayed counterfactual
\(\mathbf z'\),

\[
a_r^{\text{CF}}=\mathbb 1[d_r(z_r,z'_r)>0].
\]

Thus relevance is learned from how often a feature is changed, not from inverse
change magnitude. The latter would require the stronger assumption that every
displayed counterfactual was perceived as minimal.

In every condition the final normalization is

\[
q_r=\frac{a_r}{\sum_j a_j}.
\]

## What is stored after feedback

All learning occurs after the participant makes the current forward prediction.

| Presented feedback | Exemplar representation | Attribution representation | Prototype representation | Change memory |
| --- | --- | --- | --- | --- |
| Correct label, no XAI | Store original labeled instance | Update middle, data polarity, and diagnostic relevance | Update class observations | None |
| Attribution XAI | Store original labeled instance | Learn relevance from \(|\phi|\) and polarity from signed attribution | Update class observations | None |
| Counterfactual XAI | Store original and edited target instances | Learn relevance from changed features and polarity from edit direction | Update source and target class observations | Store original, edited profile, and target |

This is how every mental model can learn in every explanation condition. The
counterfactual condition additionally provides a reusable change.

## Forward mental models

### 1. Exemplar

For query \(\mathbf z\) and remembered instance \(i\),

\[
D_E(\mathbf z,\mathbf z_i)=\sum_{r=1}^{R}q_r d_r(z_r,z_{ir}),
\qquad
s_i(\mathbf z)=e^{-D_E(\mathbf z,\mathbf z_i)}.
\]

Class evidence includes a fixed symmetric pseudo-count of one:

\[
V_c=1+\sum_i\mathbb 1[y_i=c]s_i(\mathbf z),
\qquad
P_E(y=c\mid\mathbf z)=\frac{V_c}{\sum_k V_k}.
\]

The model retains individual examples; it does not collapse them into a
summary.

### 2. Attribution

When both classes have been observed, the middle is halfway between their
feature means:

\[
\mathbf m=\frac{\boldsymbol\mu_++\boldsymbol\mu_-}{2}.
\]

Before that, it is the mean of all observed profiles. Attribution polarity is
learned in this order:

1. displayed attribution evidence, using the sign of
   \(\phi_r(z_r-m_r)\);
2. otherwise counterfactual direction evidence, using the sign of
   \(y'(z'_r-z_r)\);
3. otherwise the direction from the negative to the positive class center.

The signed weight is derived only after relevance and polarity are known:

\[
w_r=q_r p_r.
\]

For a categorical feature, \(q_r\) is divided equally across its encoded
levels. Local contribution, total evidence, and response probability are

\[
c_r(\mathbf z)=w_r(z_r-m_r),
\qquad
g_A(\mathbf z)=\sum_r c_r(\mathbf z),
\qquad
P_A(y=+1\mid\mathbf z)=\frac{1}{1+e^{-g_A(\mathbf z)}}.
\]

This keeps relevance symmetric across models: relevance is not a renamed
attribution weight, because the attribution model additionally needs direction.

### 3. Prototype

For numerical features, the prototype center is the remembered class median
and its typical range is the interquartile range
\([\ell_{r,c},u_{r,c}]=[Q_{.25},Q_{.75}]\). For categorical features, all tied
modal values form the typical set.

Numerical mismatch from class \(c\) is

\[
\delta_{r,c}(z_r)=
\max(\ell_{r,c}-z_r,\;z_r-u_{r,c},\;0).
\]

Categorical mismatch is zero for a modal value and one otherwise. Then

\[
D_P(\mathbf z,c)=\sum_{r=1}^{R}q_r\delta_{r,c}(z_r),
\qquad
P_P(y=c\mid\mathbf z)=
\frac{e^{-D_P(\mathbf z,c)}}{\sum_k e^{-D_P(\mathbf z,k)}}.
\]

## Directly applicable counterfactual strategies

The public method names match the study hypotheses.

| Method | Natural representation | Direct operation |
| --- | --- | --- |
| `change_most_contributing_attributes` | Attribution | Rank instance-specific contributions opposing the target and edit them toward the attribution boundary. |
| `change_most_influential_attributes_to_flip` | Attribution | Rank global relevance and edit those features until the attribution prediction flips. |
| `change_most_influential_attributes_by_fixed_amount` | Attribution | Apply the same normalized amount to globally relevant features in the target-supporting direction. |
| `set_attributes_to_target_profile_values` | Prototype | Set selected mismatches to the target class's median/mode values. |
| `change_attributes_not_matching_target_profile` | Prototype | Project every mismatch into the target prototype range. |
| `change_largest_target_profile_mismatches` | Prototype | Project only the largest \(k\) mismatches. |
| `change_toward_remembered_exemplar` | Exemplar | Copy selected values from the nearest remembered target-class example. |
| `copy_changes_from_remembered_example` | Shared change memory | Apply a stored edit from the most similar remembered source profile. |

The same strategy can sometimes be approximated by more than one
representation. The table identifies the representation from which the rule
follows directly. Reusing a stored change is deliberately treated as shared
memory and can coexist with any of the three forward models.

## Time in the baseline

Time is not a fitted quantity. Trials are processed in presentation order, and
all earlier memories are equally available regardless of elapsed seconds. This
is the clean perfect-memory comparison requested before adding ACT-R decay.

A later memory extension can assign each chunk \(i\) an ACT-R-style base-level
activation

\[
A_i(t)=\ln\sum_j(t-t_{ij})^{-d}
\]

and retrieve it only when \(A_i(t)\geq\tau\). The smallest useful extension
would add decay \(d\) and retrieval threshold \(\tau\); response-time or
drift-diffusion parameters are unnecessary unless response-time predictions
become a research target.

## v0.9 participant fitting

Run:

```text
python scripts/fit_v09_perfect_memory_models.py
```

For trial \(t\), every model predicts the participant's response using trials
\(1,\ldots,t-1\). Only then is trial \(t\)'s correct label and explanation
learned. Fit is the negative log likelihood of the participant's chosen label:

\[
\operatorname{NLL}_M=-\sum_{t=1}^{10}\log P_M(r_t\mid\mathbf x_t,H_{t-1}).
\]

There are zero fitted continuous parameters, so

\[
\operatorname{BIC}_M=2\operatorname{NLL}_M.
\]

The strict minimum is recorded, while all models within \(\Delta\text{BIC}<2\)
are marked as empirically indistinguishable. This avoids turning tiny
likelihood differences from only ten responses into confident assignments.

## v0.9 counterfactual-response fitting

The action-based fit is separate from the forward-response comparison:

```text
python scripts/fit_v09_counterfactual_strategies.py
```

Training trials are used only to construct the instances, relevance,
attribution direction, prototypes, and remembered changes that were available
under the participant's assigned explanation. The participant's training
answers are not used to select a mental model.

For testing instance \(t\), let \(\Delta z_{tr}^{obs}\) be the participant's
signed normalized change to feature \(r\), and let
\(\Delta z_{tr}^{S}\) be the edit predicted by strategy \(S\). The primary
per-instance action loss is

\[
L_t(S)=\frac{1}{R}\sum_{r=1}^{R}
\left|\Delta z_{tr}^{obs}-\Delta z_{tr}^{S}\right|.
\]

This measures feature choice, direction, and magnitude in one common unit.
Feature-selection F1, direction agreement, and sparsity difference are reported
as diagnostics rather than substituted for the primary loss.

The subset size \(k\) and, when applicable, one participant-level fixed amount
are selected using leave-one-instance-out cross-validation. Thus response
\(t\) is scored using settings chosen from all of that participant's other
testing responses. The lowest mean cross-validated loss is retained as the
minimum-loss strategy. Strategies within one paired standard error of that
minimum form a support set.

The reported assignment uses the conservative one-standard-error rule: choose
the supported strategy with the fewest participant-fitted parameters, then use
cross-validated loss to break complexity ties. Parameter count is zero for
`change_attributes_not_matching_target_profile`, one for strategies that fit
only \(k\), and two for
`change_most_influential_attributes_by_fixed_amount`, which fits both \(k\) and
one common normalized amount. Both the minimum-loss and conservative selections
are retained in the output.

Every participant has ten training exposure trials, whose responses are not
used for assignment. Participants with twenty counterfactual responses are
evaluated in twenty folds with nineteen fitting responses and one held-out
response per fold. Participants with ten counterfactual responses are evaluated
in ten folds with nine fitting responses and one held-out response per fold.

The implied mental model comes from the representation that directly supports
the strategy in the table above. `copy_changes_from_remembered_example` is an
exception: stored changes are shared episodic memory, so that strategy does not
uniquely identify exemplar, attribution, or prototype forward simulation.

## Literature relationship

- Anderson et al. (2004), *An Integrated Theory of the Mind*, motivates the
  optional ACT-R chunk activation extension.
- Nosofsky (1986), *Attention, Similarity, and the
  Identification-Categorization Relationship*, motivates relevance-weighted
  exemplar similarity and class aggregation.
- Gonzalez and Dutt (2011), *Instance-Based Learning: Integrating Sampling and
  Repeated Decisions from Experience*, motivates retaining decisions as
  reusable instances.
- CoAX (thesis proposal, Chapter 3) motivates combining exemplar similarity,
  memory, and explanation-dependent feature attention.
- CoXAM (thesis proposal, Chapter 4) motivates structured explanation chunks,
  profile/threshold editing, and reuse of remembered counterfactual changes.

The archive was used as conceptual source material only; draft instructions in
it were not treated as requirements.

## Tests

```text
python -m pytest -p no:cacheprovider src/cognitive_models/minimal_mental_model/tests -q
```
