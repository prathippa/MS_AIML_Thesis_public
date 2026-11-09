# Post-hoc analysis (test split, 95% bootstrap CIs)

| Metric | A: claude-sonnet-5-5 | B: gpt-5 |
|---|---|---|
| Retained / valid / failed | 169 / 169 / 0 | 169 / 169 / 0 |
| Exact-10, all cases (failed = wrong) | 0.201 [0.142, 0.260] | 0.314 [0.243, 0.385] |
| Exact-6, valid outputs | 0.610 [0.532, 0.680] | 0.604 [0.527, 0.680] |
| Major error rate (wrong chapter) | 0.130 [0.083, 0.183] | 0.130 [0.083, 0.183] |
| Wrong heading or chapter | 0.302 [0.237, 0.373] | 0.302 [0.237, 0.379] |
| Mean severity ordinal (0 = exact) | 2.219 [1.976, 2.474] | 2.047 [1.775, 2.331] |
| ECE raw | 0.188 [0.146, 0.242] | 0.236 [0.177, 0.304] |
| ECE calibrated | 0.074 [0.045, 0.131] | 0.140 [0.095, 0.215] |
| Brier raw | 0.152 [0.131, 0.174] | 0.249 [0.227, 0.273] |
| Brier calibrated | 0.118 [0.087, 0.152] | 0.212 [0.169, 0.256] |
| Selective risk @ 70%, raw | 0.737 [0.644, 0.814] | 0.602 [0.517, 0.703] |
| Selective risk @ 70%, calibrated | 0.737 [0.644, 0.814] | 0.602 [0.517, 0.703] |
| Distinct raw confidence values | 19 | 51 |

## Paired A - B (same test cases)

- Exact-10, all cases: -0.112 [-0.172, -0.053]
- Major error rate, wrong chapter (n = 169 cases valid for both): 0.000 [-0.035, 0.035]
- Wrong heading or chapter: 0.000 [-0.047, 0.047]
- Mean severity ordinal: 0.172 [-0.006, 0.343]

## Pooled routing (pipeline result)

- Selective risk raw 0.654, calibrated 0.688, raw - calibrated -0.034, 95% CI [-0.05485232067510548, 0.0]

Per-model raw and calibrated selective risk coincide by construction: a non-decreasing
calibrator preserves each model's ranking (except ties). Any routing difference comes from pooling.
Ties in confidence are broken by SHA-256 of the prediction ID, as in the pipeline's routing.py.

Figures: `posthoc/figures/`.
