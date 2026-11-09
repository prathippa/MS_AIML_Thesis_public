#!/bin/sh
set -eu
cd "$(dirname "$0")"
python -m pytest -q
python analysis/product_routing.py --out results/extension_v8/product_routing_full --hts data/hts_2026_revision_22.json
python analysis/rq3_judge.py analyse --root results/main_v1 --out results/extension_v8/rq3_verified
python analysis/prospective_preflight.py
