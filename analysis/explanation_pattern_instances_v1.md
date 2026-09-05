# Explanation-pattern instance selection

Default boundary shortlist: closest 20% within each prediction over the full dataset. Overrides: {'diabetes': 0.5}.

Clustering input: absolute L1-normalized regularized-SHAP values concatenated with absolute normalized counterfactual changes. No original profile values, labels, directions, confidence, or boundary distance are included.

Selected 126 cases: 12 training and 30 testing per domain, balanced within three clusters and two predictions.
