# Clustered explanation instance review

## Method

The initial boundary parameter is the closest 20% of held-out profiles within each predicted class, measured by a box-constrained local linear approximation to normalized L1 distance from the profile to the model boundary. A configurable minimum candidate count can expand this fraction when the experimental quotas require more cases.

Clustering concatenates the regularized two-feature SHAP vector and the independently optimized counterfactual-change vector. Each block is oriented relative to the current prediction and L1-normalized. Prediction label, confidence, boundary margin, original values, and absolute counterfactual distance are excluded from clustering.

The number of clusters is the silhouette-maximizing K-means solution between 3 and 10, subject to a small-cluster safeguard. The three largest clusters are retained. A minimum-cost balanced assignment then chooses equal numbers from both predictions; natural-cluster matches are reported for auditing.

## Domain summary

| Domain | Chosen k | Silhouette | Selected | Training | Testing | Natural-cluster matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| housing | 10 | 0.619 | 42 | 12 | 30 | 37/42 |
| safelimit | 8 | 0.548 | 42 | 12 | 30 | 33/42 |
| diabetes | 5 | 0.378 | 42 | 12 | 30 | 33/42 |

## Boundary-filter diagnostics

| Domain | Prediction | Held out | Effective fraction | Candidates | Valid two-feature CFs | Max local boundary distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| housing | 0 | 3172 | 20.0% | 300 | 281 | 0.054 |
| housing | 1 | 3295 | 20.0% | 300 | 245 | 0.045 |
| safelimit | 0 | 146 | 20.5% | 30 | 30 | 0.067 |
| safelimit | 1 | 455 | 20.0% | 91 | 82 | 0.222 |
| diabetes | 0 | 44 | 54.5% | 24 | 24 | 0.093 |
| diabetes | 1 | 74 | 32.4% | 24 | 23 | 0.100 |

The CSV contains every selected profile, its two regularized attributions, the independently generated two-feature counterfactual, boundary margin, cluster assignment, and normalized vectors for review.
