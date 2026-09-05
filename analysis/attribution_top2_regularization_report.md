# Two-feature Kernel SHAP regularization comparison

## Question

How much does feature selection change if the current procedure—compute all five Kernel SHAP values with `l1_reg=0`, then display the two largest absolute values—is replaced by Kernel SHAP's intrinsic `l1_reg="num_features(2)"` selection?

No participant data is used. The comparison covers the 90 static experiment profiles: 10 training and 20 testing profiles in each of three domains.

The classifier, 50-row training background, exact 32-coalition Kernel SHAP evaluation, positive-class output, and random seed are held fixed. Only `l1_reg` changes. This regularizes the local SHAP surrogate; it does not retrain or regularize the neural classifier.

## Result

Across all 90 profiles, the exact selected feature pair agrees on 85 (94.4%) and changes on 5 (5.6%). Across the 180 selected feature slots, 97.2% are retained.

| Domain | N | Exact pairs | Exact rate | One retained | None retained | Feature-slot retention |
| --- | --- | --- | --- | --- | --- | --- |
| housing | 30 | 27 | 90.0% | 3 | 0 | 95.0% |
| safelimit | 30 | 28 | 93.3% | 2 | 0 | 96.7% |
| diabetes | 30 | 30 | 100.0% | 0 | 0 | 100.0% |

The regularized solution has exactly two nonzero attributions for 90 of 90 profiles. Recomputed unregularized top-two pairs reproduce the stored experiment pair for 90 of 90 profiles.

## Feature selection counts

| Domain | Feature | Truncated top-2 | Regularized | Difference |
| --- | --- | --- | --- | --- |
| housing | sqft_living | 17 | 16 | -1 |
| housing | bedrooms | 4 | 3 | -1 |
| housing | bathrooms | 8 | 7 | -1 |
| housing | floors | 8 | 9 | 1 |
| housing | grade | 23 | 25 | 2 |
| safelimit | units | 20 | 20 | 0 |
| safelimit | weight | 15 | 16 | 1 |
| safelimit | duration | 13 | 11 | -2 |
| safelimit | gender | 6 | 6 | 0 |
| safelimit | stomach_fullness | 6 | 7 | 1 |
| diabetes | glucose | 18 | 18 | 0 |
| diabetes | blood_pressure | 11 | 11 | 0 |
| diabetes | insulin | 13 | 13 | 0 |
| diabetes | bmi | 6 | 6 | 0 |
| diabetes | age | 12 | 12 | 0 |

## Most frequent changed pair transitions

| Domain | Truncated pair | Regularized pair | Instances |
| --- | --- | --- | --- |
| housing | bathrooms | grade | floors | grade | 1 |
| housing | sqft_living | bedrooms | sqft_living | grade | 1 |
| housing | sqft_living | floors | floors | grade | 1 |
| safelimit | duration | gender | gender | stomach_fullness | 1 |
| safelimit | units | duration | units | weight | 1 |

## Interpretation

Top-two truncation ranks coefficients from the full five-feature local explanation. `num_features(2)` instead chooses two features while fitting the sparse local surrogate, then refits their values. Consequently, it can select a different pair when features are correlated or provide substitutable local effects.
