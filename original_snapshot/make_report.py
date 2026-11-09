#!/usr/bin/env python3
"""
make_report.py -- produce all thesis figures and tables from results.

    python make_report.py --demo     # synthetic data, proves the figures render
    python make_report.py            # real: reads results/ written by run_study.py

Outputs land in results/report/ (figures as PNG, tables as Markdown).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import config
import reporting as RP
import calibration as C
import routing as R
from hts_hierarchy import normalize_hts

OUT = os.path.join(config.RESULTS_DIR, "report")


def _synth_records(n, acc, overconf, seed):
    """Synthetic gold/pred/confidence records for one model (demo only)."""
    rng = np.random.default_rng(seed)
    base = ["0101210010", "6109100012", "8471300000", "2922410010", "8536908085",
            "7318156060", "3904100000", "6102100000", "8501104040", "9015404000"]
    recs = []
    for i in range(n):
        gold = rng.choice(base)
        roll = rng.random()
        if roll < acc:
            pred, conf = gold, 0.90
        elif roll < acc + 0.2:
            pred, conf = gold[:6] + "99" + gold[8:], 0.72     # wrong rate line
        else:
            pred, conf = "9999" + gold[4:], 0.58               # wrong chapter
        conf = min(0.98, conf + (0.22 if overconf else 0.0)) + rng.normal(0, 0.02)
        conf = float(np.clip(conf, 0.01, 0.99))
        recs.append({"case_id": f"{seed}_{i}", "model": f"seed{seed}", "gold": normalize_hts(gold),
                     "pred": normalize_hts(pred), "confidence": conf,
                     "correct": int(normalize_hts(pred) == normalize_hts(gold))})
    return recs


def demo():
    os.makedirs(OUT, exist_ok=True)
    a = _synth_records(200, 0.58, overconf=False, seed=1)
    b = _synth_records(200, 0.60, overconf=True, seed=2)

    # RQ1
    RP.plot_severity_distribution({"Model A": a, "Model B": b}, os.path.join(OUT, "fig_severity.png"))
    open(os.path.join(OUT, "table_severity.md"), "w").write(
        RP.severity_table_md({"Model A": a, "Model B": b}))

    # RQ2 reliability (Model B, overconfident -> calibration helps visibly)
    half = len(b) // 2
    val, test = b[:half], b[half:]
    res = C.calibrate_and_eval([r["confidence"] for r in val], [r["correct"] for r in val],
                               [r["confidence"] for r in test], [r["correct"] for r in test], method="platt")
    RP.plot_reliability([r["confidence"] for r in test], [r["correct"] for r in test],
                        os.path.join(OUT, "fig_reliability.png"),
                        calibrated=res["calibrated_test_confidence"])

    # RQ2 risk-coverage (pooled raw vs calibrated)
    items = []
    for recs, over in ((a, False), (b, True)):
        h = len(recs) // 2
        v, t = recs[:h], recs[h:]
        rr = C.calibrate_and_eval([r["confidence"] for r in v], [r["correct"] for r in v],
                                  [r["confidence"] for r in t], [r["correct"] for r in t], method="platt")
        items += [{"raw": r["confidence"], "cal": float(c), "correct": r["correct"]}
                  for r, c in zip(t, rr["calibrated_test_confidence"])]
    raw = np.array([it["raw"] for it in items]); cal = np.array([it["cal"] for it in items])
    cor = np.array([it["correct"] for it in items])
    cov = [0.5, 0.6, 0.7, 0.8, 0.9]
    RP.plot_risk_coverage(R.risk_coverage_curve(raw, cor, cov),
                          R.risk_coverage_curve(cal, cor, cov),
                          os.path.join(OUT, "fig_risk_coverage.png"))
    print("Wrote figures + tables to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
    else:
        print("Real mode reads results/ from run_study.py. Use --demo for a synthetic render.")
