> **Local v8 extension (4 November 2026):** run `sh RUN_EXTENSION.sh` for offline product-level comparisons, verified RQ3 reanalysis and paid-run readiness. See `../REVISION_NOTES.md`. Original study outputs below remain unchanged; prospective generation has not started.

# HTS thesis evaluation toolkit

Code and results for the MS thesis *Beyond Exact Match: A Trustworthy Evaluation Framework for Generative AI in U.S. HTS Code Classification* (Prathippa Devi). It evaluates HTS predictions; it does not train a classifier. The stored runs are in `results/main_v1` (Claude Sonnet 5.5 vs GPT-5 on ATLAS), `results/oss_v1` (Claude vs gpt-oss-120b) and `results/hscc_v1` (Claude vs GPT-5 on HSCodeComp). **No human annotations are included yet**: the RQ3 blinded sheet awaits an annotator.

## Research questions and where the code answers them

| RQ | Question (thesis wording) | Code | Result files |
|---|---|---|---|
| RQ1 | How do hierarchy-aware measures alter the comparison of models relative to 10-digit exact match: at which levels do they differ, and in wrong-chapter errors? | `hts_hierarchy.py`, `metrics.py`, `analysis/posthoc.py`, `analysis/rq_alignment.py` (digit-level gaps), `analysis/supplementary.py` (version audit, national-suffix ceiling) | `metrics.json`, `posthoc/posthoc.json`, `rq_alignment.json`, `supplementary.json` |
| RQ2a | Does calibration improve the reliability of confidence? | `calibration.py`, `analysis/supplementary.py` (paired ECE/Brier change) | `calibration.json`, `supplementary.json` |
| RQ2b | Does calibrated confidence reduce routing risk at equal coverage, and what determines whether it can? | `routing.py`, `analysis/rq_alignment.py` (AUROC, schedule check), `analysis/supplementary.py` (model priority, tie-averaged risk) | `routing.json`, `rq_alignment.json`, `supplementary.json` |
| RQ3 | Can a tariff-grounded evaluator identify unsupported rationales? | `judge.py`, `analysis/rq3_judge.py`, `analysis/supplementary.py` (clustered AUC, gold-ruling withholding) | `main_v1/rq3/` |

The hypotheses H1–H3 are those of the research proposal and are tested as stated. Analyses in `analysis/` were added after the main runs and are reported as post hoc in the thesis.

`analysis/thesis_claims.py` recomputes every headline claim and hypothesis verdict in the thesis from the stored results and writes `results/thesis_claims.md`. It exits with an error if any claim and the data disagree; `tests/test_rq_alignment.py` runs it as part of the test suite. To regenerate everything from the stored outputs:

```sh
.venv/bin/python analysis/supplementary.py --hts data/hts_schedule/hts_full.json > results/supplementary.json
.venv/bin/python analysis/rq_alignment.py --hts data/hts_schedule/hts_full.json > results/rq_alignment.json
.venv/bin/python analysis/thesis_claims.py
```

The checker verifies that the thesis reports the data faithfully. It does not and cannot make a hypothesis hold: the pipeline files are frozen by hash in each run's `protocol.json`, and the analyses only read the sealed outputs.

## Start here

1. Extract this download into a **new folder**. Do not overwrite your only original copy.
2. Read `METHODOLOGY.md` and agree on the revised experiment with your supervisor.
3. Create a Python environment (Python 3.10+), install dependencies, and run tests.
4. Configure locally verified model IDs and API keys. Model IDs are intentionally not guessed.
5. Prepare the real benchmark inputs and independently sourced evidence before API calls.

```sh
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run_study.py --help
```

Copy `.env.example` to `.env` using your file manager, then edit `.env`. **Never commit API keys or private inputs.**

| Environment variable | Meaning |
|---|---|
| `HTS_MODEL_A` | Verified generator A model ID |
| `HTS_A_PROVIDER` | `anthropic` or `groq` |
| `HTS_MODEL_B` | Verified generator B model ID |
| `HTS_B_PROVIDER` | `anthropic` or `groq` |
| `HTS_JUDGE_MODEL` | Verified evidence-evaluator model ID, preferably independent |
| `HTS_J_PROVIDER` | `anthropic` or `groq` |
| `ANTHROPIC_API_KEY` / `GROQ_API_KEY` | Local credentials |

The SDKs are imported only when real clients are constructed. Request compatibility, account access, output-token limits, prices and model availability must be checked in your own account. Minimal provider-specific defaults are in `config.py`; configure them **before** experiment initialization. No deterministic API behaviour is promised.

## Input format

Use the original benchmark validation and test files separately. Supported primary format: JSONL (one object per line) or a JSON array. Each retained row must contain `messages` with one user description and one assistant gold answer:

```text
id: unique source-row identifier
source_ruling_id: original ruling identifier, if documented
messages:
  role=user, content=product description
  role=assistant, content=HTS US Code -> <10-digit gold>\nReasoning -> <gold reasoning>
```

This is a schema description, not sample benchmark data. Do not substitute invented records. Codes in assistant answers may be compact ten-digit strings or supported dotted formats. The loader strips the benchmark user prefix `What is the HTS US Code for ` when present. It retains gold fields only in private evaluation artifacts; generation is passed case ID, description and allowed evidence, not the gold answer/reasoning.

The default primary analysis excludes multi-code rows. Exclusions are logged. `--multi-policy first` is available for an explicitly justified secondary policy, **not** silently interchangeable with exclusion. No benchmark splitting or enlargement is performed. Dataset revision and source hashes are frozen.

## Run a no-evidence baseline

Root options go **before** the stage name. All commands are sequential; the CLI locks the experiment directory. Select a call limit appropriate to your budget: the cap is requests, not currency or tokens. Generation may use two requests per case (one malformed-output retry); judging defaults to three calls per sampled prediction.

```sh
python run_study.py --root results/baseline init --dataset-revision YOUR_ACTUAL_REVISION --evidence-mode none --max-calls 2000
python run_study.py --root results/baseline prepare --validation data/validation.jsonl --test data/test.jsonl
python run_study.py --root results/baseline evidence
python run_study.py --root results/baseline generate --model A --split validation
python run_study.py --root results/baseline generate --model A --split test
python run_study.py --root results/baseline generate --model B --split validation
python run_study.py --root results/baseline generate --model B --split test
python run_study.py --root results/baseline metrics
python run_study.py --root results/baseline calibrate
python run_study.py --root results/baseline route
python run_study.py --root results/baseline report
```

Replace `YOUR_ACTUAL_REVISION` and input paths with real values. `report` writes Markdown and CSV outputs only from saved predictions. **No synthetic fallback is run.** The no-evidence protocol cannot run evidence-support RQ3.

A conservative non-decreasing logistic map on confidence log-odds is the primary calibration default. Isotonic calibration is available at initialization with `--calibration isotonic`; pre-specify it rather than choosing whichever performs best on test data. Do not reuse a test set repeatedly to tune experimental choices.

## Evidence-enabled study and RQ3

Initialize a separate root with `--evidence-mode retrieved` (the default). The same preparation/generation/analysis sequence applies, except replace `evidence` with:

```sh
python run_study.py --root results/evidence_study evidence --corpus data/evidence.jsonl --top-k 3
```

Each corpus line must supply nonempty string fields:

```text
id, text, source_url, source_ruling_id, accessed_at, version
```

`source_url` must be an HTTP(S) URL. For this conservative retrieved-evidence protocol, **every held-out benchmark row must also have a documented `source_ruling_id`**. If your benchmark omits it, establish a traceable mapping without changing the predefined split, or use the no-evidence baseline while repairing provenance. Do not invent ruling identifiers or disable the guard to get a run through.

Retrieval uses reproducible TF-IDF unigram/bigram cosine ranking, fitted only to the supplied corpus, excluding zero-similarity results. Both generators receive identical packs. The corpus loader rejects matching held-out ruling IDs, exact normalized copied held-out descriptions/reasoning, duplicate IDs and duplicate normalized texts. **This is not proof against near-duplicates, source contamination or historical tariff-version mismatch.** Audit the corpus independently; source URLs and version strings are recorded, not externally verified by the program.

After generators have run:

```sh
python run_study.py --root results/evidence_study prepare-judge
python run_study.py --root results/evidence_study judge --model J
python run_study.py --root results/evidence_study export-human
```

The diagnostic judge sample is stratified over correctness/severity/confidence, then a distinct-product human subset is randomly drawn. The selected sample and evidence are frozen. The annotation file is `results/evidence_study/annotations/blinded.csv`. Send **only this blinded file**, not `key.json`, gold labels or judge outputs, to the independent annotator. The file hides model identity, gold code and judge decision, but the candidate answer itself can still reveal stylistic cues; blinding is not guaranteed perfect.

The annotator fills `label` (`supported`, `unsupported` or `uncertain`), `reason` and `annotator_id` without editing the supplied content. The rubric is in `METHODOLOGY.md`. After annotation, create a local attestation JSON containing:

```text
annotator_id: the actual annotator ID
expertise: accurate description of actual qualifications
independent_of_generation: true
blinded_to_gold_and_judge: true
rubric_version: "v2"
```

Then import the annotations:

```sh
python run_study.py --root results/evidence_study validate-human --annotations results/evidence_study/annotations/blinded.csv --attestation data/attestation.json
python run_study.py --root results/evidence_study report --require-human
```

Annotation size is frozen during `init` (`--human-size`, default 30). If the frozen judge sample contains too few distinct products, export fails rather than silently reducing the requirement. Plan sample sizes before final runs. `--require-human` requires completed independent annotations, **not** proof that the judge is reliable or the classifications are legally correct. Expertise is self-attested and must be verified by the researcher.

## Output and audit trail

- `protocol.json` and hashes: pre-specified method, sampling, model configuration and code fingerprints
- `environment.json`: local Python/package versions
- `validation/`, `test/`: separate records, load audits, evidence packs, raw generation histories and scored records
- `requests.sqlite`: persistent provider-request attempts across generation/judging/restarts
- `metrics.json`, `paired_accuracy.json`: all-case accuracy and paired-product comparison
- `calibrator_A.json`, `calibrator_B.json`, `calibration.json`: fitted states, valid-output ECE/Brier
- `routing.json`: raw/calibrated risk, achieved coverage, forced failures, per-model controls, cluster intervals and frozen-threshold evaluation
- `judge_cases.jsonl`, `judge_results.jsonl`: diagnostic sample, permutations, reasons and abstentions
- `annotations/`: frozen blinded sample and private link key
- `human_validation.json`: completeness, agreement/kappa, abstentions, confusion matrix and explicit limitations
- `report/`: Markdown report, model-metric CSV, routing-curve CSV and input hashes

Do not manually edit results. Source-code changes after initialization require a **new experiment directory**. Failed predictions remain failed cached outcomes; they are not automatically retried on the next invocation. For a new failure-retry policy use a separately specified experiment rather than deleting inconvenient failures. An interrupted generation stage leaves an unsealed cache: rerun the same `generate` command before analysis. Inspect any stale `.active.lock` only after confirming no process is running. Request reservations are conservative: a crash may consume an attempt slot even when no response was obtained.

Hashes detect accidental artifact changes; they are not tamper-proof security or a substitute for provenance review. The cap bounds attempted provider calls, not token expenditure or the cost of any individual call.

## Compatibility and limitations

This release is **not a drop-in replacement** for the old `run_study.py` flags or result paths. Core hierarchy/metric functions and the original tests are retained, while the orchestrator and data schemas have changed. Your uploaded sources are preserved in `original_snapshot/`; original test files are included in `tests/`.

The old low-level `reporting.py` plotting helpers remain for compatibility, but the new report path deliberately does not claim to reproduce the uploaded JPGs or generate finished thesis figures. There are no fabricated observations or real-result example files in this package. See `TEST_STATUS.md` for executed checks and unverified aspects.
