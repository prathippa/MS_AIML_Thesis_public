# Post-hoc analysis (test split, 95% bootstrap CIs)

| Metric | A: claude-sonnet-5-5 | B: gpt-5 |
|---|---|---|
| Retained / valid / failed | 10 / 10 / 0 | 10 / 10 / 0 |
| Exact-10, all cases (failed = wrong) | 0.300 [0.000, 0.600] | 0.100 [0.000, 0.300] |
| Exact-6, valid outputs | 0.700 [0.400, 1.000] | 0.700 [0.400, 1.000] |
| Major error rate (wrong heading/chapter) | 0.000 [0.000, 0.000] | 0.100 [0.000, 0.300] |
| Mean severity ordinal (0 = exact) | 1.600 [0.900, 2.300] | 1.900 [1.200, 2.600] |
| ECE raw | 0.220 [0.130, 0.500] | 0.387 [0.236, 0.525] |
| ECE calibrated | 0.230 [0.087, 0.516] | 0.000 [0.000, 0.200] |
| Brier raw | 0.236 [0.116, 0.361] | 0.252 [0.210, 0.291] |
| Brier calibrated | 0.241 [0.036, 0.463] | 0.090 [0.010, 0.250] |
| Selective risk @ 70%, raw | 0.571 [0.286, 1.000] | 1.000 [0.714, 1.000] |
| Selective risk @ 70%, calibrated | 0.571 [0.286, 1.000] | 1.000 [0.714, 1.000] |
| Distinct raw confidence values | 8 | 7 |

## Paired A - B (same test cases)

- Exact-10, all cases: 0.200 [0.000, 0.500]
- Major error rate (n = 10 cases valid for both): -0.100 [-0.300, 0.000]
- Mean severity ordinal: -0.300 [-1.300, 0.700]

## Pooled routing (pipeline result)

- Selective risk raw 0.857, calibrated 0.786, raw - calibrated 0.071, 95% CI [0.0, 0.1428571428571429]

Per-model raw and calibrated selective risk coincide by construction: a non-decreasing
calibrator preserves each model's ranking (except ties). Any routing difference comes from pooling.

Figures: `posthoc/figures/`.
