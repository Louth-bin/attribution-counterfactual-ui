# Explanation-pattern instance selection

Default boundary shortlist: closest 20% within each prediction over the full dataset. Overrides: none.

Clustering input: absolute L1-normalized regularized-SHAP values concatenated with absolute normalized counterfactual changes. No original profile values, labels, directions, confidence, or boundary distance are included.

Selected 30 cases: 12 training and 18 testing per domain, with the required number from each prediction in every cluster.
