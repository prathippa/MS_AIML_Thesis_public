# Prospective follow-up amendment v8a — 4 November 2026

This is a follow-up under Objective 2 / RQ2b. Original aims, research questions, hypotheses, completed studies and stopped technical pilot are preserved. This amendment is frozen before any evaluation generation. It is a local timestamped, hashed protocol, not an independent public preregistration.

## Why amend

The 4,000-token technical pilot stopped after GPT-5 exhausted its allowance. There were 136 Claude development records and 28 GPT-5 records (two failed), totalling 165 recorded API attempts. Neither model generated evaluation predictions. GPT-5 now receives 8,000 output tokens, including reasoning; Claude remains at 4,000. Provider defaults and prompts are unchanged. This is a separate model setting and study; no retroactive replacement of original results.

## Cohort and analysis

Use 100 development and 700 evaluation products. Retain the unused 400 evaluation products and append 300 seeded, screened products. All 200 original development products remain excluded from evaluation. The first 100 original development products are regenerated; the two known token-cap failures are ordered first solely to check the remedy. No model correctness determines cohort membership or ordering.

The single primary contrast is GPT-5-designated combined routing minus confidence-only selective exact-code risk at 50% coverage, with a paired product-bootstrap 95% interval (2,000 resamples, seed7). Negative values favour combined routing. Confirmatory support requires the interval entirely below zero. Other policies, coverages and Claude designation are secondary and descriptive; no best-performing result replaces the primary contrast. One prediction per product forms the queue. Designated failures are always referred; failed secondary predictions cannot agree. SHA256 product-ID tie-breaking is fixed across policies.

50% coverage was chosen in light of earlier observed routing behaviour at70% and represents a larger review allocation; it was fixed before new evaluation outputs. The lexicographic combined rule prioritises agreement, then predicted-code membership in the frozen schedule, then confidence. Gold codes never enter these signals. Small-sample development calibration is descriptive and does not change the primary ordering.

## Precision and interpretation

Conditional paired resampling of earlier validation products (5,000 resamples, seed20261104) suggests approximate interval half-widths of3.5–3.8 percentage points at400 products and2.9–3.0 points at700. These estimates reuse finite empirical support and do not guarantee power under a new distribution. This design does not claim80% power for a three-point gain. Report the observed difference and interval. Non-significance does not imply equivalence, and does not by itself exclude a four-point benefit.

## Cost and stopping

The original EUR100 ceiling includes the stopped pilot. A shared ledger retains every original reservation and appends usage reconciliations. Verified token usage is charged conservatively at the documented provider rates times1.5 for currency/fees contingency, rounded upward; this is not an invoice total. Unknown usage keeps its full reservation. Reserve EUR0.075 before each Claude call andEUR0.13 before each GPT-5 call, including retries. The pipeline stops before issuing a request that would exceed the shared ceiling. Official rates checked4November2026: https://platform.claude.com/docs/en/models/sonnet-5-5/overview and https://developers.openai.com/api/docs/models/gpt-5.

There are1,600 base requests and an additional request-count ceiling of1,760. Worst-case full-token usage cannot all be purchased withinEUR100; continuation depends on verified usage settling below maximum reservations. If the cap or technical circuit stops evaluation, preserve all outputs and report incomplete evaluation, without a confirmatory finding. Never select a smaller completed subset according to its results or silently increase the budget. Three consecutive technical failures or more than10% failed outputs in a completed split trigger review. No model, prompt, policy or token-cap tuning after evaluation begins.

## Remaining limitations

Public training-split provenance does not establish absence of model-training exposure. Exact and TF-IDF near-duplicate filtering cannot establish independent rulings without source identifiers. Historical golds may differ from the2026 tariff snapshot. Agreement requires paying for a second model. No expert validation of RQ3, real reviewer correction accuracy, review time, or legal/financial benefit is supplied by this follow-up.

Machine-readable authority: prospective_plan_v8a.json; cohort selection: data/prospective_candidate_v8a/selection_protocol.json; outputs: results/prospective_v8a. The final results must be integrated and checked before this thesis is treated as submission-ready.
