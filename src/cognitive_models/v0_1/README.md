# Cognitive model v0.1

This folder freezes the model used for the latest v1.9 participant fits. It is
the baseline immediately before adding sequential learning or forgetting.

The implementation is deliberately limited to three files:

- `model.py`: representation, prediction, probabilistic feature selection,
  conditional additive-rho fitting, and cross-validation.
- `fit.py`: data loading and the command-line fitting entry point.
- `__init__.py`: the public package interface.

Historical fitting, plotting, conversion, and sensitivity scripts elsewhere in
the repository are not part of this version.

## Baseline assumptions

1. All 12 training cases are combined into one fixed memory before testing.
2. The candidate families are `feature contribution` and `weighted examples`.
3. The observed number of edited features, K, is retained for each test trial.
4. Feature subsets are probabilistic:
   `P(S|x,K) proportional to exp(sum of standardized feature scores in S)`.
5. Feature-selection parameters are selected by selection NLL.
6. Rho is an additive normalized amount fitted afterward, conditional on the
   participant's observed feature subset.
7. Model family is selected by five-fold cross-validated selection NLL, then
   conditional amount MAE.
8. In the feature-contribution family, the local/global feature score controls
   both selection probability and the share of remaining feasible movement.

There is no sequential updating, forgetting, or retrieval refresh in v0.1.

## Run

From the repository root:

```text
python -m src.cognitive_models.v0_1.fit
```

To smoke-test one participant without replacing the archived fit file:

```text
python -m src.cognitive_models.v0_1.fit --participant PARTICIPANT_ID --output temporary.csv
```
