# Persisted study outputs

**Not legal advice. No generated demo observations are used.**
Model/dataset/evidence provenance must still be verified by the researcher.

## Classification and reliability

| Model | Retained cases | Valid | All-case exact-10 | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|---:|
| A | 325 | 325 | 0.2400 | 0.1836 | 0.0308 |
| B | 325 | 325 | 0.2646 | 0.3289 | 0.0733 |

Failed/missing predictions count as wrong in all-case accuracy. Calibration conditions on valid outputs.
Structural hierarchy severity is not measured financial or legal loss.

## Routing interpretation

Unit: model_case_prediction. This is model–case recommendation coverage, NOT distinct-product workload.
Raw minus calibrated risk at requested coverage 70%: 0.004396.
Conditional product-cluster bootstrap interval: [-0.00659340659340657, 0.01758241758241763].
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
