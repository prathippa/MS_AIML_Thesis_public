# Post-hoc analysis (test split, 95% bootstrap CIs)

| Metric | A: claude-sonnet-5-5 | B: gpt-5 |
|---|---|---|
| Retained / valid / failed | 325 / 325 / 0 | 325 / 325 / 0 |
| Exact-10, all cases (failed = wrong) | 0.240 [0.194, 0.289] | 0.265 [0.215, 0.314] |
| Exact-6, valid outputs | 0.637 [0.588, 0.689] | 0.603 [0.551, 0.655] |
| Major error rate (wrong chapter) | 0.185 [0.145, 0.228] | 0.175 [0.135, 0.218] |
| Wrong heading or chapter | 0.268 [0.218, 0.317] | 0.283 [0.234, 0.335] |
| Mean severity ordinal (0 = exact) | 2.175 [1.982, 2.363] | 2.194 [2.000, 2.394] |
| ECE raw | 0.184 [0.150, 0.229] | 0.329 [0.285, 0.376] |
| ECE calibrated | 0.031 [0.028, 0.083] | 0.073 [0.049, 0.125] |
| Brier raw | 0.170 [0.156, 0.185] | 0.273 [0.258, 0.288] |
| Brier calibrated | 0.136 [0.113, 0.161] | 0.166 [0.141, 0.191] |
| Selective risk @ 70%, raw | 0.683 [0.626, 0.749] | 0.683 [0.617, 0.740] |
| Selective risk @ 70%, calibrated | 0.683 [0.626, 0.749] | 0.683 [0.617, 0.740] |
| Distinct raw confidence values | 20 | 49 |

## Paired A - B (same test cases)

- Exact-10, all cases: -0.025 [-0.074, 0.028]
- Major error rate, wrong chapter (n = 325 cases valid for both): 0.009 [-0.021, 0.040]
- Wrong heading or chapter: -0.015 [-0.049, 0.018]
- Mean severity ordinal: -0.018 [-0.172, 0.136]

## Pooled routing (pipeline result)

- Selective risk raw 0.681, calibrated 0.677, raw - calibrated 0.004, 95% CI [-0.00659340659340657, 0.01758241758241763]

Per-model raw and calibrated selective risk coincide by construction: a non-decreasing
calibrator preserves each model's ranking (except ties). Any routing difference comes from pooling.
Ties in confidence are broken by SHA-256 of the prediction ID, as in the pipeline's routing.py.

Figures: `posthoc/figures/`.
