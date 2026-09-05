# Cognitive model v0.1 parameter-sweep simulation

Primary cohort: 46 participants with at least 70% of testing edits moving toward the target. All 62 participants are included as a sensitivity analysis.

Each sweep changes one parameter while preserving the participant's winning model family and fitted values for every other parameter. Predictions use the maximum-probability feature subset conditional on the observed number of edited features, matching the earlier v0.1 participant-versus-model analysis.

## Largest primary-cohort endpoint changes

The table reports the change from each parameter's lowest to highest grid value. Positive means the simulated outcome increases.

| XAI | Parameter | Desideratum | n | Mean change [95% CI] |
|---|---|---|---:|---:|
| No XAI | Age actionable | Actionability rate | 13 | -0.6769 [-0.7308, -0.6192] |
| No XAI | Additive change margin (ρ) | Normalized edit distance (L1) | 13 | 0.6046 [0.5033, 0.7049] |
| Counterfactual example | Additive change margin (ρ) | Normalized edit distance (L1) | 17 | 0.6014 [0.5286, 0.6742] |
| Feature attribution | Additive change margin (ρ) | Normalized edit distance (L1) | 16 | 0.4661 [0.3934, 0.5363] |
| Counterfactual example | Remembered-change reliance (β) | Actionability rate | 7 | 0.4357 [0.1786, 0.6786] |
| Counterfactual example | Additive change margin (ρ) | Counterfactual success rate | 17 | 0.3412 [0.2353, 0.4265] |
| Counterfactual example | Additive change margin (ρ) | Target-confidence gain | 17 | 0.3385 [0.2499, 0.4118] |
| Feature attribution | Additive change margin (ρ) | Counterfactual success rate | 16 | 0.2937 [0.2531, 0.3375] |
| Counterfactual example | Age actionable | Actionability rate | 17 | -0.2824 [-0.4324, -0.1441] |
| Feature attribution | Age actionable | Actionability rate | 16 | -0.2781 [-0.3875, -0.1781] |
| Feature attribution | Additive change margin (ρ) | Target-confidence gain | 16 | 0.2601 [0.2196, 0.2987] |
| No XAI | Additive change margin (ρ) | Counterfactual success rate | 13 | 0.2462 [0.1615, 0.3423] |
| No XAI | Additive change margin (ρ) | Target-confidence gain | 13 | 0.2346 [0.1770, 0.2977] |
| Counterfactual example | Remembered-change reliance (β) | Normalized edit distance (L1) | 7 | -0.1939 [-0.2201, -0.1655] |
| Counterfactual example | Explanation reliance (η) | Actionability rate | 17 | 0.1706 [0.0265, 0.3324] |
| Feature attribution | Explanation reliance (η) | Actionability rate | 16 | 0.1656 [0.0344, 0.3156] |
| Counterfactual example | Additive change margin (ρ) | Boundary-distance improvement | 17 | -0.1183 [-0.1379, -0.0999] |
| No XAI | Global relevance reliance (α) | Normalized edit distance (L1) | 6 | -0.1113 [-0.1460, -0.0810] |
| Counterfactual example | Global relevance reliance (α) | Actionability rate | 10 | 0.1100 [-0.0001, 0.2700] |
| No XAI | Additive change margin (ρ) | Boundary-distance improvement | 13 | -0.1079 [-0.1229, -0.0933] |
| Feature attribution | Global relevance reliance (α) | Normalized edit distance (L1) | 14 | -0.1068 [-0.1219, -0.0912] |
| Feature attribution | Additive change margin (ρ) | Boundary-distance improvement | 16 | -0.1052 [-0.1175, -0.0933] |
| Feature attribution | Explanation reliance (η) | Normalized edit distance (L1) | 16 | -0.0883 [-0.1108, -0.0654] |
| No XAI | Exemplar locality (λ) | Counterfactual success rate | 7 | -0.0786 [-0.1643, -0.0143] |

Success and actionability are rates. Confidence gain is a probability change. Boundary improvement and edit L1 use summed range-normalized L1 units. Plausibility is nearest-reference Gower similarity on a 0–1 scale.
