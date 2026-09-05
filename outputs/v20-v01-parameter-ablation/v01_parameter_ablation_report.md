# Cognitive model v0.1 parameter ablation

Primary analysis uses the 46 participants with at least 70% of edits moving toward the target. The all-62 analysis is a sensitivity check. Each participant's winning model family is held fixed. Positive deltas mean held-out performance worsened when the parameter was fixed to its null value.

| Parameter | Primary metric | Applicable n | Mean delta [95% bootstrap CI] | Holm p | Conclusion |
|---|---:|---:|---:|---:|---|
| Explanation reliance (η) | selection_nll | 33 | 0.2358 [0.1364, 0.3561] | 0.0002875 | useful |
| Global relevance reliance (α) | selection_nll | 30 | 0.0907 [-0.0268, 0.2503] | 0.587 | inconclusive |
| Additive change margin (ρ) | amount_mae | 46 | 0.0196 [0.0115, 0.0286] | 4.351e-05 | useful |
| Exemplar locality (λ) | selection_nll | 16 | 0.2385 [0.0274, 0.5785] | 0.05066 | limited evidence |
| Remembered-change reliance (β) | selection_nll | 7 | 0.8251 [0.3190, 1.4895] | 0.04156 | useful |
| Age-actionability flexibility | selection_nll | 46 | 0.3041 [0.1130, 0.4986] | 0.00392 | useful |

Selection parameters are evaluated with five-fold cross-validated selection NLL. The additive margin ρ is evaluated with five-fold cross-validated conditional amount MAE. One-sided paired Wilcoxon tests ask whether ablation worsens performance; Holm adjustment covers the six parameter tests within each cohort.
