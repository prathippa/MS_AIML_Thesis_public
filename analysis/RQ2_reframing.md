# RQ2 — reframing note for the dissertation (draft)

## The issue

The proposal states H2 as: *"At equivalent review coverage, calibrated confidence will achieve
lower selective risk than raw confidence when used to route cases for human review."*

For a single model this cannot happen. Platt scaling (and isotonic regression) is a
**non-decreasing** map: if case X had higher raw confidence than case Y, it still has
higher (or equal) calibrated confidence. Routing at fixed coverage streamlines the top-k cases
by confidence, so raw and calibrated routing select the **same cases** and have **identical
selective risk** (ties excepted). The pilot confirmed this exactly (per-model difference 0.000).

This is a property of the method, not a weakness of the study — but the thesis must say so
rather than present a null result as a failed hypothesis.

## Where calibration *can* change routing (what the code measures)

1. **Probability quality (reliability).** Calibration changes *what the number means*:
   ECE and Brier score measure whether "0.8" really means ~80% correct. Reported per model
   with bootstrap CIs and reliability diagrams.
2. **Pooled queues across models (ranking across sources).** When predictions from two models
   share one review queue, their raw scores are on different scales (e.g. one model always
   says 0.4–0.6, another 0.1–0.8). Calibration puts both on a common probability scale, which
   *can* change which model's cases are streamlined. This is the pipeline's primary RQ2
   comparison (`routing.json`, unit = model–case prediction, product-clustered bootstrap CI).
3. **Threshold transfer.** A policy such as "streamline when P(correct) ≥ t", with t chosen on
   validation to hit a target risk, only has a stable meaning on calibrated probabilities. The
   pipeline freezes the validation threshold and reports achieved test risk and coverage.

## Suggested revised wording

**RQ2.** Does post-hoc calibration of elicited confidence (i) improve the reliability of
confidence scores, and (ii) reduce selective risk of a human-review routing policy when
predictions from multiple models are routed through a common queue or by a fixed
validation-selected probability threshold, compared with routing on raw confidence?

**H2a.** Calibrated confidence will have lower ECE and Brier score than raw confidence on the
held-out test split.

**H2b.** At matched coverage of a pooled two-model queue, calibrated confidence will achieve
lower selective risk than raw confidence.

**Methods sentence.** "Because both calibrators are non-decreasing, they preserve each model's
internal ranking; within a single model, raw and calibrated routing at fixed coverage are
therefore identical except for ties. We report this invariance explicitly and evaluate the
routing benefit of calibration where it can arise: in a pooled multi-model queue and under a
probability threshold selected on validation data and frozen for test."

## Things to report alongside

- Per-model raw vs calibrated selective risk (identical — show it as the invariance check).
- Number of distinct raw confidence values per model (coarse confidences create ties).
- Cases where the Platt slope fits to 0 (confidence carries no information about correctness):
  the calibrated score becomes constant, so routing within that model is effectively arbitrary.
  This happened for gpt-5 in the 10-case pilot; check whether it persists in the full study.
