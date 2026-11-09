#!/usr/bin/env python3
"""
run_study.py -- sprint orchestrator
-----------------------------------
One entry point that sequences the study. Each stage reads and writes JSON/JSONL
in results/, so stages are independent and resumable.

    python run_study.py --self-test                 # offline wiring check (no API/data)
    python run_study.py --stage load   --data data/atlas_test.jsonl
    python run_study.py --stage generate --model A  --split test
    python run_study.py --stage metrics
    python run_study.py --stage calibrate
    python run_study.py --stage route
    python run_study.py --stage judge

The load/generate/judge stages need the real dataset and API keys; metrics,
calibrate and route run purely on cached predictions.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
from hts_hierarchy import evaluate
import metrics as M
import calibration as C
import routing as R


# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #
def _save(name, obj):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def join_predictions(gold_records, generations):
    """Merge gold codes with cached model predictions into eval-ready records."""
    gold = {r["case_id"]: r for r in gold_records}
    out = []
    for g in generations:
        if g.get("status") != "ok" or g["case_id"] not in gold:
            continue
        gold_code = gold[g["case_id"]]["gold"]
        out.append({"case_id": g["case_id"], "model": g["model"], "gold": gold_code,
                    "pred": g["hts_code"], "confidence": g["confidence"],
                    "correct": int(evaluate(g["hts_code"], gold_code)["exact_10"])})
    return out


# --------------------------------------------------------------------------- #
# Offline wiring self-test (synthetic data)                                    #
# --------------------------------------------------------------------------- #
def self_test():
    import numpy as np
    rng = np.random.default_rng(0)

    def synth(n, offset, acc):
        """n cases; reported confidence shifted by `offset`; true accuracy `acc`."""
        true_p = rng.uniform(0.3, 0.95, n)
        reported = np.clip(true_p + offset, 0, 1)
        correct = (rng.uniform(0, 1, n) < acc).astype(int)
        return reported, correct

    # two models with different confidence behaviour, same cases
    va, ya = synth(200, 0.00, 0.6)      # Model A: well-behaved
    vb, yb = synth(200, 0.25, 0.4)      # Model B: overconfident, less accurate
    ta, tya = synth(200, 0.00, 0.6)
    tb, tyb = synth(200, 0.25, 0.4)

    # calibrate each model on its validation split
    resA = C.calibrate_and_eval(va, ya, ta, tya, method="auto")
    resB = C.calibrate_and_eval(vb, yb, tb, tyb, method="auto")
    print("Calibration (test ECE raw -> calibrated):")
    print(f"  Model A: {resA['raw']['ece']:.3f} -> {resA['calibrated']['ece']:.3f}  ({resA['method']})")
    print(f"  Model B: {resB['raw']['ece']:.3f} -> {resB['calibrated']['ece']:.3f}  ({resB['method']})")

    # assemble the pool with raw and calibrated scores, then route
    items = ([{"raw": float(r), "cal": float(c), "correct": int(y)}
              for r, c, y in zip(ta, resA["calibrated_test_confidence"], tya)] +
             [{"raw": float(r), "cal": float(c), "correct": int(y)}
              for r, c, y in zip(tb, resB["calibrated_test_confidence"], tyb)])
    route = R.compare_pooled_routing(items)
    gap = R.bootstrap_routing_gap(items, n_boot=300)
    print(f"\nPooled routing @ {route['primary_coverage']:.0%} coverage:")
    print(f"  selective risk  raw={route['selective_risk_raw']:.3f}  "
          f"calibrated={route['selective_risk_calibrated']:.3f}")
    print(f"  improvement={route['improvement']:+.3f}  "
          f"bootstrap 95% CI=[{gap[1]:+.3f}, {gap[2]:+.3f}]")
    print("\nSelf-test OK: generation->calibration->routing wiring runs end to end.")


# --------------------------------------------------------------------------- #
# Real stages (need data / API)                                                #
# --------------------------------------------------------------------------- #
def stage_load(data_path):
    """Parse the real ATLAS messages format into results/records.jsonl."""
    from data_loader import load_atlas_messages
    recs = load_atlas_messages(data_path)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "records.jsonl")
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    multi = sum(r["multi_code"] for r in recs)
    print(f"Loaded {len(recs)} ATLAS rows ({multi} multi-code) -> {out}")
    return recs


def stage_metrics():
    gold = [json.loads(l) for l in open(os.path.join(config.RESULTS_DIR, "records.jsonl"))]
    summary = {}
    for lbl in ("A", "B"):
        gen_path = os.path.join(config.RESULTS_DIR, f"generations_{lbl}.jsonl")
        if not os.path.exists(gen_path):
            continue
        gens = [json.loads(l) for l in open(gen_path)]
        recs = join_predictions(gold, gens)
        summary[lbl] = M.aggregate(recs)
    print(_save("metrics.json", summary))
    return summary


def main():
    ap = argparse.ArgumentParser(description="HTS trust-eval study orchestrator")
    ap.add_argument("--self-test", action="store_true", help="offline wiring check")
    ap.add_argument("--stage", choices=["load", "generate", "metrics", "calibrate", "route", "judge"])
    ap.add_argument("--model", choices=["A", "B"])
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--data")
    args = ap.parse_args()

    if args.self_test:
        self_test(); return
    if args.stage == "load":
        stage_load(args.data); return
    if args.stage == "metrics":
        stage_metrics(); return
    print(f"Stage '{args.stage}' runs on your machine with the dataset/API keys in place. "
          f"See README for the per-stage inputs.")


if __name__ == "__main__":
    main()
