# Explanation-pattern instance selection

Boundary shortlist: closest 50% within each prediction over the full dataset.

Clustering input: absolute L1-normalized regularized-SHAP values concatenated with absolute normalized counterfactual changes. No original profile values, labels, directions, confidence, or boundary distance are included.

Selected 42 cases: 12 training and 30 testing per domain, balanced within three clusters and two predictions.
