# Independent-feature revision comparison

This comparison uses the same 60 complete participants and five-fold
cross-validation in both implementations. The revised global lapse is applied
to individual feature-inclusion probabilities, so its numerical value is not
directly interchangeable with the initial subset-level lapse.

| Selected-model metric | Initial v0.2 | Independent-feature revision | Change |
| --- | ---: | ---: | ---: |
| Global lapse | 0.550 | 0.400 | different mechanism |
| Subset NLL | 3.194 | 2.907 | -0.287 |
| Expected selection F1 | 0.355 | 0.389 | +0.034 |
| MAP selection F1 | 0.282 | 0.415 | +0.133 |
| Exact-subset accuracy | 0.097 | 0.136 | +0.039 |
| Cardinality MAE | 0.640 | 0.650 | +0.010 |
| Conditional amount MAE | 0.283 | 0.297 | +0.014 |
| Conditional direction accuracy | 0.591 | 0.576 | -0.015 |

The revised model improves selection but not edit amount. Mean MAP cardinality
moves from 1.397 to 1.564 edits, toward the observed 1.843. Age selection falls
from 49.2% to 6.8% (observed 12.2%). Glucose rises from 32.8% to 46.1%
(observed 79.5%) and Insulin from 6.3% to 18.0% (observed 47.7%), so important
feature-choice mismatch remains. Blood pressure and BMI are now overpredicted.

The result supports independent feature sampling and an actionability mechanism,
but not the revised amount rule as a sufficient account. In particular, the
model still lacks an explicit estimate of the current instance's distance to a
decision boundary. The endpoint/change blend makes the alternatives precise,
but neither necessarily equals the movement required to change the classifier's
decision.
