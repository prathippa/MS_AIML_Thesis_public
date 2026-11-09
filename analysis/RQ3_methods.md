# RQ3 — methods note for the dissertation (draft)

Script: `analysis/rq3_judge.py`. Outputs: `results/main_v1/rq3/`.

## What was run

- **Sample.** The pipeline's own `stratified_judge_subset` on the main-study test predictions
  (both generators pooled), 15 per stratum, seed 7: correct–high confidence, correct–low
  confidence (median split among correct predictions), near-miss (wrong statistical suffix or
  rate line) and major error (wrong subheading, heading or chapter). n = 60 predictions.
- **Evaluator.** `google/gemini-3.8-flash` via OpenRouter, the rubric prompt in `judge.py`
  unchanged (Table 6), 3 runs per prediction with the evidence order shuffled each run.
  It belongs to neither generator family (Anthropic, OpenAI), which avoids self-preference bias.
  `gpt-5-mini` was not used, because it shares a family with generator B (gpt-5).
- **Evidence pack** (per prediction):
  1. the official HTSUS provision text for the **predicted** code (USITC export, 2026 HTS
     Revision 22), shown as the hierarchy path from heading to statistical line. If the
     10-digit line no longer exists, the nearest existing 8/6/4-digit provision is shown and
     labelled as such;
  2. up to k = 3 CROSS-ruling passages retrieved by TF-IDF on the product description from the
     `orlandowhite/hts-cross-rulings` dump (54,558 rulings, revision `050ee21…`).

## Deviations from the proposal (state these)

1. **Evaluator-only evidence.** The proposal gives the same evidence to the generators and the
   evaluator. The ATLAS mirror has no source-ruling IDs, so the generators ran without
   evidence (`--evidence-mode none`) and the evidence was used by the evaluator only.
2. **Gold-ruling exclusion by code, not by ruling ID.** The case's own CROSS ruling could not
   be identified reliably: matching descriptions to the dump was unreliable, because ATLAS
   descriptions are one short line. So **every ruling that carries the case's gold 10-digit
   code** is excluded from that case's retrieval pool. Rulings whose text contains the
   held-out description or gold reasoning are also excluded. This is leak-proof by
   construction and conservative: correct predictions cannot be "supported" by a ruling for
   the identical code, so support for correct predictions must come from the tariff text and
   neighbouring rulings. Expect a shift towards *uncertain*, not towards *supported*.
3. **Current tariff text, historical rulings.** The provisions come from the 2026 schedule.
   Some older statistical lines have changed, and those cases fall back to the nearest
   existing provision. The share is reported (`provision_exact_10_rate`).

## How RQ3 is measured (proposal: consistency + agreement)

- **Consistency:** mean share of runs agreeing with the majority label; unanimous rate;
  abstentions (no majority).
- **Validity against the gold CROSS code** (the proposal's primary reference without an
  expert):
  - AUC of the evaluator's support score (supported = 1, uncertain = 0.5, unsupported = 0,
    averaged over runs) for separating correct from wrong predictions, with a bootstrap CI;
  - P(supported | correct) and P(unsupported | major error).
- **Human agreement:** `export-human` writes a blinded 30-case CSV
  (`rq3/annotations/blinded.csv`). Cohen's κ can be computed once an independent annotator,
  ideally a trade-compliance professional, labels it. The proposal treats this as optional
  ("where available").
