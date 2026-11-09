# Public release for thesis version 16

Code, frozen protocols, cached model responses and results for the MS thesis
*Beyond Exact Match: A Trustworthy Evaluation Framework for Generative AI in U.S. HTS Code Classification*
(Prathippa Devi, November 2026). Start with `README.md` and `RUNBOOK.md`.

This repository is the `code/` folder of the thesis archive `MS_thesis_code_results_v16.zip`, with two kinds of file left out:

- `data/`: downloaded datasets and the 2026 HTSUS Revision 22 export. Sources, revisions and checksums are in Appendix A.5 of the thesis. Scripts that need them (`analysis/posthoc_v16.py`, `analysis/verify_v10.py`, `analysis/supplementary.py`, the schedule policies in `analysis/product_routing.py`) will not run until the files are restored under `data/`.
- `*.sqlite`: local request-count and budget ledgers.

The unit tests need neither: `python -m pytest -q` (172 tests).

No API keys are included, and none are needed to recompute results from the stored responses.
Benchmark text in `results/` comes from ATLAS (CROSS rulings mirror, Apache-2.0) and HSCodeComp (Apache-2.0); CROSS rulings are published by U.S. Customs and Border Protection.
