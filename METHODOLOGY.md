# Methodological safeguards and remaining research decisions

## RQ1: retrospective classification severity

Compare exact agreement at 2/4/6/8/10 digits and hierarchical partial credit. Digits 7–8 and 9–10 form complete prefix levels rather than independently rewarded digits. Weight schemes are sensitivity choices, not empirically validated costs. Structural severity is NOT measured duty loss, legal risk or certainty that a benchmark answer remains applicable under a different tariff version. Syntactically valid ten-digit codes are not checked for existence against a dated official HTS schedule; supply that additional validation before any operational use.

All retained cases contribute to all-case exact-10 accuracy. Missing/failed outputs count as wrong and must be escalated. Conditional valid-output severity/partial-credit metrics have separate denominators. Multi-code filtering and every other exclusion must be disclosed. The paired model accuracy interval resamples product cases.

## RQ2: calibration versus ranking

Confidence is elicited as the probability that the *full ten-digit answer* is correct. ECE and Brier measure reliability of valid-output probabilities. Calibration is fitted on validation only; the same fitted map is used for test.

The primary constrained logistic calibration in this enhancement differs from the original unconstrained fit. It fits a regularized logistic map to clipped confidence log-odds with non-negative slope. This must be documented as a change; do not present it as an unchanged standard implementation. The slope can be zero; clipping/extreme probabilities and flat maps can introduce ties. Isotonic regression can also create ties. Exact-coverage routing uses a deterministic label-independent hashed prediction ID to resolve tied scores.

A strictly increasing transform preserves within-model order, so improved ECE does not imply improved selective risk at equal coverage. Per-model controls are exported. Separately calibrated model scores *can* change cross-model ordering, but any benefit is an empirical question, not an expected result that the code should force.

The pooled primary unit is a **model–case recommendation**. If A and B both predict the same product, there are two recommendation units, not two independent products. This does not directly measure unique-product human workload. A product-level operational workflow requires an additional pre-specified model-selection/combination policy, which is not supplied here.

Exact-coverage comparisons are retrospective; test score order statistics set that coverage. Failed/missing outputs are forced escalations, and both valid-pool coverage and all-prediction coverage are exported. Integer rounding and ties must be reported.

The pooled risk-gap bootstrap clusters recommendations by case ID and conditions on the already fitted calibrators. It does **not** capture calibration-fit, evidence-selection or model-stochasticity uncertainty. Calibration ECE/Brier helper intervals are available but not automatically treated as evidence of a significant improvement. There is no guarantee that real results improve.

A second output selects the largest complete tied-score group whose **empirical validation** risk meets the target, then freezes its score threshold for test. This is not a statistical risk-control guarantee. It normally inherits single-model monotonic invariance; it is not a second cure for the original ranking hypothesis problem. No test labels are used to select its threshold.

## RQ3: evidence-support audit

Before real judging: freeze evidence sources, versions, retrieval settings, exact source exclusions and near-duplicate audit. The evidence corpus must not contain evaluated rulings or benchmark gold reasoning. Missing provenance cannot be cured by making up identifiers. The program's exact checks are conservative but incomplete: manual audit is still mandatory.

### Annotation rubric v2

- **supported:** supplied evidence materially supports the proposed classification and rationale, with no material contradiction in that supplied evidence.
- **unsupported:** supplied evidence materially contradicts the recommendation/rationale, or the rationale claims support not provided by the evidence.
- **uncertain:** insufficient, ambiguous or mixed supplied evidence prevents a supported/unsupported determination. Empty evidence is uncertain rather than automatically unsupported.

This is support *relative to supplied evidence*, not a definitive legal assessment, and not a guarantee of full-ten-digit correctness. Independent reviewers should apply the same rubric without benchmark gold codes or judge labels and document qualifications. Do not relabel support outcomes to force agreement with benchmark exact-match outcomes.

Repeated judging records evidence permutations, raw text, reasons, valid-run count and abstentions. A strict majority of all requested runs is required; malformed/API-error runs are not silently removed from the denominator. Repeated calls to one model, especially with deterministic settings, are not independent annotators. Same-model generation/judging is flagged rather than represented as independent validation.

Diagnostic stratified sampling is not a population-prevalence sample. Human subsets are sampled by distinct product, one recommendation per selected product, reducing but not eliminating dependencies. Agreement and kappa are descriptive checks; the exported human bootstrap conditions on this diagnostic sample. Undefined kappa is stored as null rather than incorrectly perfect agreement. A small convenient sample does not establish general judge validity. There is no pre-set declaration that kappa or agreement is 'good enough': agree on decision criteria, confidence intervals and appropriate expertise with the supervisor before final runs.

## Still outside code

Model choice and feasibility justification; literature-search databases and inclusion criteria; verification of citations and industry statistics; tariff-version alignment; independent expert recruitment; sample-size/power justification; ethics/confidentiality review; work plan and actual progress; thesis/video formatting and university requirements. These cannot be honestly marked completed by software changes.
