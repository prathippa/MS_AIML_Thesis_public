# Sprint Runbook — real experiment, step by step

This is the exact sequence to produce real results, verified end-to-end on sample
data with mock models (see `examples/rehearsal.py`; it runs the whole flow offline).
Every command is `python run_study.py --root <dir> <stage> [options]`. Stages are
sealed and resumable — rerunning a completed stage is safe.

## Pre-sprint (finish before 28 Oct)

1. **API keys** — put `ANTHROPIC_API_KEY` and `GROQ_API_KEY` in `.env`.
2. **Model IDs** — export the exact, verified snapshot IDs (these are frozen at `init`):
   `export HTS_MODEL_A=<claude sonnet 5.5 id>  HTS_MODEL_B=<llama 3.3 70b id>  HTS_JUDGE_MODEL=<a third, independent model id>`
   The judge must be a *different* model from the generators.
3. **Dataset** — download ATLAS; make a `validation.jsonl` and `test.jsonl` from the
   predefined splits. Confirm the real column names against `data_loader.load_atlas_messages`.
4. **Ruling IDs (decides RQ3)** — each row needs a top-level `source_ruling_id` (the CROSS
   ruling number) for `--evidence-mode retrieved`. If your rows don't have them, either add
   them, or run `--evidence-mode none` (RQ1+RQ2 only; `prepare-judge` will refuse).
5. **Evidence corpus** (only if retrieved) — a `corpus.jsonl` where every row has
   `id, text, source_url (http/https), source_ruling_id, accessed_at, version`; no duplicates;
   no row may be a case's own ruling or copy a case description/reasoning. The guard is fail-closed.
6. **Pilot** — run the whole sequence below on ~10 real rows first. Confirm requests succeed,
   parsing holds, costs match expectation, and `requests.sqlite` logs them. Treat the pilot as
   a dry run, not findings; then start fresh with a new `--root`.

## The command sequence

```bash
R=results/study_v2
python run_study.py --root $R init --dataset-revision atlas-2026-10 \
    --evidence-mode retrieved --multi-policy exclude \
    --calibration platt --human-size 30 --max-calls 2000
python run_study.py --root $R prepare --validation validation.jsonl --test test.jsonl
python run_study.py --root $R evidence --corpus corpus.jsonl --top-k 3
python run_study.py --root $R generate --model A --split validation
python run_study.py --root $R generate --model B --split validation
python run_study.py --root $R generate --model A --split test
python run_study.py --root $R generate --model B --split test
python run_study.py --root $R metrics        # RQ1: exact + hierarchy severity, all-case vs valid
python run_study.py --root $R calibrate       # ECE/Brier, fits + freezes calibrators
python run_study.py --root $R route           # RQ2: pooled raw vs calibrated, risk-coverage
python run_study.py --root $R prepare-judge    # RQ3 sample (needs retrieved evidence)
python run_study.py --root $R judge --model J
python run_study.py --root $R export-human     # writes annotations/blinded.csv + key.json
#   --> annotate blinded.csv yourself (second annotator): fill ONLY label/reason/annotator_id.
#       Do not touch description/predicted_code/rationale/evidence (content is hash-checked).
#       Write an attestation.json: {annotator_id, expertise, independent_of_generation:true,
#       blinded_to_gold_and_judge:true, rubric_version:"v2"}
python run_study.py --root $R validate-human --annotations filled.csv --attestation attestation.json
python run_study.py --root $R report --out $R/report --require-human
```

## Gotchas the rehearsal caught

- **Code freeze.** `init` hashes every `*.py`. If you edit any module afterward, the run
  aborts ("Code changed after freeze"). Finalise code first; to change it, start a new `--root`.
- **Protocol freeze.** `--multi-policy`, `--evidence-mode`, `--calibration`, coverage and
  target-risk are set once at `init` and cannot drift. Choose deliberately.
- **Failed/missing predictions count as wrong** in all-case accuracy and are force-escalated
  in routing; valid-only metrics are reported separately. Both appear in `metrics.json`.
- **Routing coverage is per model-case prediction, not per product.** The report states this;
  keep it in the dissertation wording.
- **Budget** is global and persistent in `requests.sqlite`, capped by `--max-calls`.

## Five-day plan (28 Oct – 1 Nov)

| Day | Date | Stages | Output |
|---|---|---|---|
| 1 | 28 Oct | init, prepare, evidence, generate ×4 | cached predictions; verify cost/parsing |
| 2 | 29 Oct | metrics, calibrate, route | RQ1 + RQ2 results + severity figures |
| 3 | 30 Oct | prepare-judge, judge, export-human, **annotate 30 cases**, validate-human | RQ3 + Cohen's κ |
| 4 | 31 Oct | HSCodeComp robustness (optional), report, figures/tables | study_report.md + CSVs |
| 5 | 1 Nov | error diagnosis, interpretation | results chapter draft started |

Write-up and video run 2 Nov – 2 Dec. The pipeline produces the numbers; the dissertation
makes the argument around them.
