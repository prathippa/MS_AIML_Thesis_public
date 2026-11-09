"""
rq_alignment.py
---------------
Post-hoc analyses added when the research questions were aligned with what the pipeline measures
(thesis v3). All read the sealed, scored outputs through `pipeline.Study`; none calls a model.

  1. Where the gap lives: paired accuracy difference at each HTS digit level (RQ1).
  2. Confidence ranking: AUROC of raw and calibrated confidence for exact correctness (RQ2b).
  3. Schedule check: escalate predicted codes absent from the 2026 HTSUS before ranking by
     confidence, compared with confidence-only routing at the same coverage (RQ2b, exploratory).

    python analysis/rq_alignment.py --hts data/hts_schedule/hts_full.json > results/rq_alignment.json
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supplementary import STUDIES, N_BOOT, SEED, auc, by_model, hkey, hts_codes, scored  # noqa: E402

LEVELS = ("2", "4", "6", "8", "10")


def ci(xs):
    return [round(float(x), 3) for x in np.percentile(xs, [2.5, 97.5])]


def case_bootstrap(case_ids, rng):
    pick = rng.integers(0, len(case_ids), len(case_ids))
    return [case_ids[i] for i in pick]


# 1 ------------------------------------------------------------------------------------------
def digit_gap(root):
    """Accuracy at each digit level over all cases (failed generations count as wrong), and the
    paired difference B - A with a case bootstrap."""
    _, rows = scored(root, "test")
    hit = defaultdict(dict)
    for r in rows:
        hit[r["case_id"]][r["model"]] = {d: bool(r["status"] == "ok" and r["prefix_correct"][d]) for d in LEVELS}
    cases = sorted(c for c, v in hit.items() if set(v) == {"A", "B"})
    acc = {m: {d: float(np.mean([hit[c][m][d] for c in cases])) for d in LEVELS} for m in ("A", "B")}
    diff = {d: np.array([hit[c]["B"][d] - hit[c]["A"][d] for c in cases], dtype=float) for d in LEVELS}
    rng = np.random.default_rng(SEED)
    boots = {d: [] for d in LEVELS}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(cases), len(cases))
        for d in LEVELS:
            boots[d].append(diff[d][i].mean())
    out = {"n_cases": len(cases), "levels": {}}
    for d in LEVELS:
        lo, hi = ci(boots[d])
        out["levels"][d] = {"acc_A": round(acc["A"][d], 3), "acc_B": round(acc["B"][d], 3),
                            "diff_B_minus_A": round(float(diff[d].mean()), 3), "diff_ci95": [lo, hi],
                            "detectable": not (lo <= 0 <= hi)}
    # share of the 10-digit gap that opens after the 6-digit level
    g10, g6 = diff["10"].mean(), diff["6"].mean()
    out["gap_beyond_6_digits"] = round(float(g10 - g6), 3)
    return out


# 2 ------------------------------------------------------------------------------------------
def confidence_ranking(root):
    """How well confidence ranks correct above incorrect predictions. A non-decreasing calibrator
    leaves within-model AUROC unchanged except where it merges distinct scores into ties."""
    ok, _ = scored(root, "test")
    rng = np.random.default_rng(SEED)
    out = {}
    groups = dict(sorted(by_model(ok).items()))
    groups["pooled"] = ok
    for m, rs in groups.items():
        y = [bool(r["exact_10"]) for r in rs]
        raw = [r["confidence"] for r in rs]
        cal = [r["calibrated_confidence"] for r in rs]
        b_raw, b_cal = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(rs), len(rs))
            yy = [y[j] for j in i]
            a1, a2 = auc([raw[j] for j in i], yy), auc([cal[j] for j in i], yy)
            if not (math.isnan(a1) or math.isnan(a2)):
                b_raw.append(a1)
                b_cal.append(a2)
        out[m] = {"n": len(rs), "n_correct": int(sum(y)),
                  "auroc_raw": round(auc(raw, y), 3), "auroc_raw_ci95": ci(b_raw),
                  "auroc_calibrated": round(auc(cal, y), 3), "auroc_calibrated_ci95": ci(b_cal),
                  "distinct_raw": len(set(raw)), "distinct_calibrated": len(set(cal))}
    return out


# 3 ------------------------------------------------------------------------------------------
def _select(rs, coverage, key):
    k = max(1, int(math.floor(len(rs) * coverage + 0.5)))
    return sorted(rs, key=key)[:k]


def _risk(sel):
    return 1 - float(np.mean([r["exact_10"] for r in sel]))


def _policies(rs, ten, coverage):
    conf = _select(rs, coverage, lambda r: (-r["confidence"], hkey(r["prediction_id"])))
    sched = _select(rs, coverage, lambda r: (r["pred"] not in ten, -r["confidence"], hkey(r["prediction_id"])))
    return conf, sched


def schedule_check(root, codes, coverage=0.7):
    """Policy S: predictions whose 10-digit code is not in the 2026 HTSUS go to review first; the
    remaining slots are filled by confidence. Compared with confidence-only routing at the same
    coverage, pooled and within each model. Exploratory: specified after the results were known."""
    ten = codes[0]
    ok, _ = scored(root, "test")
    groups = dict(sorted(by_model(ok).items()))
    groups["pooled"] = ok
    rng = np.random.default_rng(SEED)
    out = {}
    for m, rs in groups.items():
        invalid = [r for r in rs if r["pred"] not in ten]
        errors = [r for r in rs if not r["exact_10"]]
        conf, sched = _policies(rs, ten, coverage)
        by_case = defaultdict(list)
        for r in rs:
            by_case[r["case_id"]].append(r)
        cases = sorted(by_case)
        gaps = []
        for _ in range(N_BOOT):
            boot = [r for c in case_bootstrap(cases, rng) for r in by_case[c]]
            c2, s2 = _policies(boot, ten, coverage)
            gaps.append(_risk(c2) - _risk(s2))
        out[m] = {
            "n": len(rs), "coverage": coverage, "k": len(conf),
            "invalid_codes": len(invalid),
            "invalid_codes_wrong": sum(not r["exact_10"] for r in invalid),
            "share_of_errors_flagged": round(sum(not r["exact_10"] for r in invalid) / len(errors), 3) if errors else None,
            "risk_confidence_only": round(_risk(conf), 3),
            "risk_schedule_check": round(_risk(sched), 3),
            "risk_reduction": round(_risk(conf) - _risk(sched), 3),
            "risk_reduction_ci95_case_clustered": ci(gaps),
            "invalid_streamlined_confidence_only": sum(r["pred"] not in ten for r in conf),
            "invalid_streamlined_schedule_check": sum(r["pred"] not in ten for r in sched),
            "risk_valid_codes_only": round(_risk([r for r in rs if r["pred"] in ten]), 3) if len(invalid) < len(rs) else None,
            "coverage_valid_codes_only": round(1 - len(invalid) / len(rs), 3),
        }
    # the confidence-only policy must reproduce the pipeline's own routing numbers
    routing = json.load(open(Path(root) / "routing.json"))
    assert abs(out["pooled"]["risk_confidence_only"] - round(routing["selective_risk_raw"], 3)) < 1e-9, root
    for m in ("A", "B"):
        assert abs(out[m]["risk_confidence_only"] - round(routing["within_model"][m]["selective_risk_raw"], 3)) < 1e-9, (root, m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hts", required=True)
    args = ap.parse_args()
    codes = hts_codes(args.hts)
    out = {}
    for s in STUDIES:
        root = f"results/{s}"
        out[s] = {"digit_gap": digit_gap(root),
                  "confidence_ranking": confidence_ranking(root),
                  "schedule_check": schedule_check(root, codes)}
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
