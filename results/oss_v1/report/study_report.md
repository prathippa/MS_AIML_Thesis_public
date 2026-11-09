# Persisted study outputs

**Not legal advice. No generated demo observations are used.**
Model/dataset/evidence provenance must still be verified by the researcher.

## Classification and reliability

| Model | Retained cases | Valid | All-case exact-10 | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|---:|
| A | 169 | 169 | 0.2544 | 0.1417 | 0.0777 |
| B | 169 | 168 | 0.0178 | 0.6837 | 0.0114 |

Failed/missing predictions count as wrong in all-case accuracy. Calibration conditions on valid outputs.
Structural hierarchy severity is not measured financial or legal loss.

## Routing interpretation

Unit: model_case_prediction. This is model–case recommendation coverage, NOT distinct-product workload.
Raw minus calibrated risk at requested coverage 70%: 0.050847.
Conditional product-cluster bootstrap interval: [0.021186440677966156, 0.0805084745762712].
Matched coverage is retrospective and conditions on valid outputs; all-prediction coverage is separately exported.
Failed/missing outputs are forced escalations. Ties and integer rounding can change achieved coverage.
Deployment threshold is selected only on validation, then frozen for test. It has NO finite-sample risk guarantee.
Increasing within-model calibration need not improve ranking; per-model invariance controls are in routing.json.
Intervals condition on fitted calibrators, not the uncertainty of calibration fitting.

## Evidence support

**RQ3 UNVALIDATED: independent human annotations have not been supplied.**

## Audit trail

Keep protocol, source hashes, environment, split audits, evidence manifests, raw generation/judge histories, calibrator states and requests.sqlite.
Do not upload API keys, proprietary inputs or confidential annotations to public repositories.
