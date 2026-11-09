# HTS Trust-Eval

Evaluation and human-review-routing framework for AI-generated U.S. 10-digit HTS
classifications. This repo is the code for the dissertation *"Beyond Exact Match"*.
It does **not** build a classifier — it evaluates trustworthiness: hierarchy-aware
error severity, confidence calibration, calibrated routing, and evidence support.

## Pre-sprint checklist (do before 28 Oct)
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your **Anthropic** and **Groq** API keys.
3. Download the ATLAS/CROSS dataset into `data/` (Hugging Face, CC BY 4.0).
4. Verify the build: `python run_study.py --self-test` (runs the pipeline on synthetic data),
   and run the tests in `tests/` (51 passing across 7 modules).

## Repo layout
```
src/
  hts_hierarchy.py   # Decision 2: 6-tier severity + paired partial credit   [DONE, tested]
  generation.py      # Stage 2: query Model A & Model B, cache structured output   [DONE, tested]
  metrics.py         # Stages 3-4: exact + hierarchy-aware metrics             [DONE, tested]
  calibration.py     # Stage 6: Platt / isotonic, ECE, Brier, reliability      [DONE, tested]
  routing.py         # Stage 7 + Decision 1: pooled raw-vs-calibrated routing  [DONE, tested]
  judge.py           # Stage 5: tariff-grounded evidence-support evaluator     [DONE, tested]
  reporting.py       # Figures + tables for the write-up (RQ1/RQ2/RQ3)        [DONE, tested]
  data_loader.py     # Stage 1: parse REAL ATLAS messages format + sampler     [DONE, tested]
tests/               # unit tests (run before the sprint)
data/                # datasets (gitignored)
results/             # cached model outputs, metrics, figures (gitignored)
run_study.py         # one orchestrator entry point (try: python run_study.py --self-test)
make_report.py       # build thesis figures/tables (try: python make_report.py --demo)
config.py            # models, splits, thresholds, cost caps
```

## Sprint run order (28 Oct - 1 Nov)
Day 1 `generation.py` -> Day 2 `metrics.py` -> Day 3 `calibration.py` + `routing.py`
-> Day 4 `judge.py` + human validation + HSCodeComp -> Day 5 analysis & figures.

All model outputs are cached and timestamped, so re-running analysis never re-bills the API.

## Dataset format (confirmed Oct 2026)
ATLAS (`flexifyai/cross_rulings_hts_dataset_for_tariffs`) is a **chat/SFT dataset**: each row is
`{"messages":[user, assistant]}` and the gold code(s) + reasoning are embedded in the assistant
text as `HTS US Code -> <code(s)>` / `Reasoning -> ...`. `data_loader.load_atlas_messages` parses this.
**Some rows carry multiple gold codes** -- the loader records them all (`gold_all`) and flags the row
(`multi_code`); the primary single-code gold is the first. Decide before the sprint whether to
filter multi-code rows or score against the primary code. Chapter 98/99 special provisions
(e.g. `9803.00.50`) are not 10-digit and are skipped. HSCodeComp (secondary) lives on Hugging Face;
confirm its field names on download.

Try the pipeline on real ATLAS rows now: `python examples/dry_run_demo.py`
