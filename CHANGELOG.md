# Enhancement notes

This release delivers actual files, not an assertion that a GitHub repository has been modified.

- Added `pipeline.py` and a real CLI for prepare, evidence, generate, metrics, calibrate, route, judge, human export/import and report.
- Added split-qualified identifiers, explicit multi-code exclusions, input manifests and cross-split checks.
- Kept gold labels/reasoning out of generation inputs.
- Added sourced TF-IDF evidence retrieval with fail-closed source-ruling checks.
- Added provider-specific settings, response metadata and a persistent SQLite call budget.
- Added configuration-safe caching, raw attempt histories and forced escalation for unusable predictions.
- Preserved 2/4/6/8/10 hierarchy scoring; tightened malformed-code and numeric validation.
- Added constrained non-decreasing logistic calibration; explicit method choice and fitted-state exports.
- Added deterministic ties, whole-tie-group deployment thresholds and product-cluster pooled bootstrap.
- Added per-model routing controls and separate valid-pool/all-prediction denominators.
- Added repeated-judge abstention and full valid-run accounting.
- Added blinded distinct-product human sampling, import completeness checks and agreement diagnostics.
- Added persisted-result Markdown/CSV reporting without synthetic fallback.
- Added regression and mocked integration tests. See the actual test log/status rather than previous chat test-count assertions.

No changes have been pushed to your repository. No provider requests or actual benchmark experiments were run in preparing this release. Model IDs remain user-configured and unverified.

## v16 (9 November 2026)

- Added `analysis/posthoc_v16.py`, `tests/test_posthoc_v16.py` and `results/posthoc_v16/results.json`: ECE bin-edge sensitivity, RQ3 verdicts by provision type, follow-up combined versus schedule-only interval. Offline; no model calls.
- No frozen top-level module and no earlier result file was changed, so every study's stored source hashes still match.
- Updated `TEST_STATUS.md` and `TEST_RESULT.txt` to the current 172-test run.
