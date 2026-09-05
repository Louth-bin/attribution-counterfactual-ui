# Cognitive model v0.2: parsimonious edit model

This implementation crosses three representations with two memory scopes:

| Representation | Prototype | Exemplar |
| --- | --- | --- |
| Base labeled instances | activation-weighted target prototype | CoAX/GCM-weighted opposite-class instances |
| Additive evidence | activation-weighted attribution schema | CoAX/GCM-weighted local attributions |
| Counterfactual edits | activation-weighted featurewise edit memory | CoAX/GCM-weighted featurewise edit memory |

Availability follows the experimental information: base models are available
in every condition, additive models after Attribution or Counterfactual XAI,
and counterfactual-edit models only after Counterfactual XAI.

The common fitted parameters are `(decay, distance, selection spread, margin,
age actionable)`, with distance omitted for prototype variants. Counterfactual
edit variants additionally fit `edit blend`, which interpolates between moving
toward a remembered counterfactual endpoint (`0`) and reusing its change amount
(`1`). One lapse is selected globally by pooled cross-validated subset NLL.

There is no retrieval threshold and no fixed or observed feature-count
parameter. Each feature is sampled independently. Counterfactual memory stores
three separate featurewise summaries: edit frequency, signed change conditional
on that feature having been edited, and counterfactual endpoint conditional on
that feature having been edited. It does not retrieve or reproduce feature
pairs as atomic units.

Core inclusion uses `p_j = 1 - exp(-spread * relative_score_j)`: small spread
concentrates the distribution on a single high-scoring feature, while larger
spread permits broader multi-feature edits. The global lapse is also applied
independently to every feature as `p'_j = (1-lapse)p_j + lapse/2`. The product
Bernoulli distribution is then conditioned on at least one edit.

Run from the parent of `implementation`:

```text
python -m implementation.v0_2_parsimonious.fit
```

The default inputs are the latest v2.3 participant export and matching diabetes v1.5
case bundle in the experiment repository. Use `--responses` and `--cases` to
override them.

Outputs in `results/`:

- `fits.csv`: participant-by-variant parameters and fit metrics;
- `trial_predictions.csv`: held-out trial predictions;
- `global_lapse.csv`: pooled lapse search;
- `summary.json`: all-participant and retained-cohort summaries;
- `REPORT.md`: readable fit report.

See `V0_2_REVISION_NOTES.md` for the independently sampled feature revision and
`results/REVISION_COMPARISON.md` for its held-out comparison with the initial
v0.2 implementation.
