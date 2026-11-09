# Current status (v16, 9 November 2026)

`python -m pytest -q` on this package: **172 tests passed**. `analysis/thesis_claims.py`: 29 of 29 selected checks pass. `analysis/verify_v10.py`: all checks pass. The 172 tests comprise the 167 of the v14 package plus five for `analysis/posthoc_v16.py`. The text below is the historical October release note and is kept for the record.

# Verified release checks

The included `TEST_RUN.txt` is the actual output of a completed local test run: **121 tests passed**. The suite includes the original 55 tests plus 66 regression/integration cases (including parameterized cases). `python run_study.py --help` and Python byte-compilation also completed successfully.

Tests use explicitly labelled offline canned fixtures. They are **not** real model predictions, actual independent annotations, benchmark findings or scientific-validity evidence. No paid API experiments were run.

Live API compatibility, dataset mapping, evidence provenance, tariff validity and independent human-review quality remain unverified. Read `README.md` and `METHODOLOGY.md` before use. The current test count is a verified release check; it should not be confused with earlier chat claims that did not deliver a test log or bundle.

The report path exports persisted-result Markdown and CSV. It does not automatically produce finished figures. Original prototype files in `original_snapshot/` are retained for comparison, not for execution.
