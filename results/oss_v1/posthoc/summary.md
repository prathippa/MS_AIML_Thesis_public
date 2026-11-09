# Post-hoc analysis (test split, 95% bootstrap CIs)

| Metric | A: claude-sonnet-5-5 | B: openai/gpt-oss-120b |
|---|---|---|
| Retained / valid / failed | 169 / 169 / 0 | 169 / 168 / 1 |
| Exact-10, all cases (failed = wrong) | 0.254 [0.189, 0.320] | 0.018 [0.000, 0.041] |
| Exact-6, valid outputs | 0.610 [0.532, 0.680] | 0.066 [0.030, 0.101] |
| Major error rate (wrong chapter) | 0.136 [0.089, 0.189] | 0.256 [0.191, 0.327] |
| Wrong heading or chapter | 0.308 [0.243, 0.379] | 0.702 [0.625, 0.774] |
| Mean severity ordinal (0 = exact) | 2.148 [1.876, 2.408] | 3.845 [3.678, 3.994] |
| ECE raw | 0.142 [0.090, 0.200] | 0.684 [0.655, 0.711] |
| ECE calibrated | 0.078 [0.048, 0.146] | 0.011 [0.001, 0.035] |
| Brier raw | 0.172 [0.150, 0.195] | 0.502 [0.473, 0.532] |
| Brier calibrated | 0.157 [0.122, 0.196] | 0.018 [0.000, 0.041] |
| Selective risk @ 70%, raw | 0.695 [0.585, 0.754] | 0.983 [0.958, 1.000] |
| Selective risk @ 70%, calibrated | 0.695 [0.585, 0.754] | 0.975 [0.958, 1.000] |
| Distinct raw confidence values | 18 | 26 |

## Paired A - B (same test cases)

- Exact-10, all cases: 0.237 [0.177, 0.302]
- Major error rate, wrong chapter (n = 168 cases valid for both): -0.119 [-0.179, -0.066]
- Wrong heading or chapter: -0.393 [-0.470, -0.316]
- Mean severity ordinal: -1.696 [-1.940, -1.458]

## Pooled routing (pipeline result)

- Selective risk raw 0.864, calibrated 0.814, raw - calibrated 0.051, 95% CI [0.021186440677966156, 0.0805084745762712]

Per-model raw and calibrated selective risk coincide by construction: a non-decreasing
calibrator preserves each model's ranking (except ties). Any routing difference comes from pooling.
Ties in confidence are broken by SHA-256 of the prediction ID, as in the pipeline's routing.py.

Figures: `posthoc/figures/`.
