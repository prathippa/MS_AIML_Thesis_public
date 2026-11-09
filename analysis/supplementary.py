"""
supplementary.py
----------------
Post-hoc analyses added after examiner-style review of the thesis draft. All read the sealed,
scored outputs through `pipeline.Study`; none calls a model.

  1. Paired bootstrap intervals for the change in ECE and Brier score (raw - calibrated).
  2. Wilson intervals for threshold-transfer risk (small accepted sets).
  3. A validation-based model-priority baseline for the pooled routing queue.
  4. Tariff-version and code-validity audit against the 2026 HTSUS (Revision 22).
  5. Diagnostic audit of the open-weight (gpt-oss-120b) outputs.
  6. RQ3 AUC with a product-clustered bootstrap (the 60 predictions cover fewer products).
  7. Worked cases for the discussion.
  8. Constant validation-prevalence Brier baseline for calibration (does it beat the base rate?).
  9. Tie-averaged selective risk: routing with ties at the cut resolved in expectation, not by hash.
 10. National-suffix ceiling: invalid predicted codes whose 8-digit rate line matches the gold.
 11. RQ3 gold-ruling withholding: verdicts on correct codes with and without a ruling withheld.

    python analysis/supplementary.py --hts data/hts_schedule/hts_full.json > results/supplementary.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import ece, brier  # noqa: E402
from pipeline import Study  # noqa: E402

STUDIES = ("main_v1", "oss_v1", "hscc_v1")
N_BOOT, SEED = 2000, 7


def scored(root, split):
    study = Study(root)
    rows = study.scored(split)
    return [r for r in rows if r["status"] == "ok"], rows


def by_model(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["model"]].append(r)
    return out


def hkey(pid):
    return hashlib.sha256(pid.encode()).hexdigest()


# 1 ------------------------------------------------------------------------------------------
def paired_calibration_change(root):
    ok, _ = scored(root, "test")
    rng = np.random.default_rng(SEED)
    res = {}
    for m, rs in sorted(by_model(ok).items()):
        raw = np.array([r["confidence"] for r in rs])
        cal = np.array([r["calibrated_confidence"] for r in rs])
        y = np.array([1 if r["exact_10"] else 0 for r in rs])
        d_ece, d_brier = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(rs), len(rs))
            d_ece.append(ece(raw[i], y[i]) - ece(cal[i], y[i]))
            d_brier.append(brier(raw[i], y[i]) - brier(cal[i], y[i]))
        res[m] = {
            "n": len(rs),
            "mean_raw_conf": round(float(raw.mean()), 3), "mean_cal_conf": round(float(cal.mean()), 3),
            "accuracy": round(float(y.mean()), 3),
            "ece_reduction": round(ece(raw, y) - ece(cal, y), 3),
            "ece_reduction_ci95": [round(float(x), 3) for x in np.percentile(d_ece, [2.5, 97.5])],
            "brier_reduction": round(brier(raw, y) - brier(cal, y), 3),
            "brier_reduction_ci95": [round(float(x), 3) for x in np.percentile(d_brier, [2.5, 97.5])],
        }
    return res


# 1b -----------------------------------------------------------------------------------------
def constant_prevalence_baseline(root):
    """Does calibration do more than push every confidence toward the base rate? Assign every test
    prediction the *validation* correctness rate (a single constant) and score it with the Brier
    score; compare with the calibrated model on the same test cases. A calibrated Brier below the
    constant one means the calibrated probabilities carry case-specific information, not merely a
    corrected average. Paired case bootstrap on (constant - calibrated)."""
    vok, _ = scored(root, "validation")
    tok, _ = scored(root, "test")
    vby, tby = by_model(vok), by_model(tok)
    rng = np.random.default_rng(SEED)
    res = {}
    for m in sorted(tby):
        p0 = float(np.mean([1 if r["exact_10"] else 0 for r in vby[m]]))  # validation prevalence
        y = np.array([1 if r["exact_10"] else 0 for r in tby[m]], dtype=float)
        cal = np.array([r["calibrated_confidence"] for r in tby[m]], dtype=float)
        b_const = (p0 - y) ** 2
        b_cal = (cal - y) ** 2
        diffs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            diffs.append(float(b_const[i].mean() - b_cal[i].mean()))
        res[m] = {
            "n_test": len(y),
            "validation_prevalence": round(p0, 3),
            "brier_constant_prevalence": round(float(b_const.mean()), 3),
            "brier_calibrated": round(float(b_cal.mean()), 3),
            "brier_gain_over_constant": round(float(b_const.mean() - b_cal.mean()), 3),
            "brier_gain_ci95": [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])],
            "calibrated_beats_constant": bool(np.percentile(diffs, 2.5) > 0),
        }
    return res


# 2 ------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def threshold_transfer(root):
    r = json.load(open(Path(root) / "routing.json"))
    out = {}
    for k in ("threshold_raw", "threshold_cal"):
        t, v = r[k]["test"], r[k]["validation"]
        n = t["n_streamlined"]
        err = round(t["selective_risk"] * n) if n else 0
        out[k] = {"threshold": v["threshold"], "validation_n": v["n_streamlined"],
                  "test_n": n, "test_errors": err, "test_risk": t.get("selective_risk"),
                  "test_risk_wilson95": wilson(err, n)}
    return out


# 3 ------------------------------------------------------------------------------------------
def model_priority_baseline(root):
    """Streamline the model with higher validation accuracy first (by its own confidence),
    then the other model; compare with raw and calibrated pooling at the same k."""
    val_ok, _ = scored(root, "validation")
    acc = {m: np.mean([r["exact_10"] for r in rs]) for m, rs in by_model(val_ok).items()}
    first = max(acc, key=acc.get)
    ok, all_rows = scored(root, "test")
    routing = json.load(open(Path(root) / "routing.json"))
    k = routing["raw"]["n_streamlined"]
    order = sorted(ok, key=lambda r: (r["model"] != first, -r["confidence"], hkey(r["prediction_id"])))
    top = order[:k]
    return {
        "validation_accuracy": {m: round(float(a), 3) for m, a in acc.items()}, "priority_model": first,
        "k": k, "n_pooled": len(all_rows),
        "risk_model_priority": round(1 - float(np.mean([r["exact_10"] for r in top])), 3),
        "streamlined_by_model": dict(Counter(r["model"] for r in top)),
        "risk_raw": round(routing["selective_risk_raw"], 3),
        "risk_calibrated": round(routing["selective_risk_calibrated"], 3),
    }


# 4 ------------------------------------------------------------------------------------------
def hts_codes(path):
    rows = json.load(open(path, encoding="utf-8"))
    ten, eight, six = set(), set(), set()
    for r in rows:
        d = re.sub(r"\D", "", r.get("htsno") or "")
        if len(d) == 10:
            ten.add(d)
        if len(d) >= 8:
            eight.add(d[:8])
        if len(d) >= 6:
            six.add(d[:6])
    return ten, eight, six


def version_audit(root, codes):
    ten, eight, _ = codes
    ok, _ = scored(root, "test")
    out = {}
    for m, rs in sorted(by_model(ok).items()):
        gold_valid = [r["gold"] in ten for r in rs]
        pred_valid = [r["pred"] in ten for r in rs]
        nat = [r for r in rs if r["severity_tier"] in ("wrong_stat_suffix", "wrong_rate_line")]
        cats = Counter()
        for r in nat:
            if r["gold"] not in ten:
                cats["gold_not_in_2026"] += 1
            elif r["pred"] not in ten:
                cats["pred_not_in_2026" + ("_8digit_exists" if r["pred"][:8] in eight else "_8digit_missing")] += 1
            else:
                cats["both_valid_2026"] += 1
        exact = [r for r in rs if r["exact_10"]]
        out[m] = {
            "n": len(rs),
            "gold_valid_2026": round(float(np.mean(gold_valid)), 3),
            "pred_valid_2026": round(float(np.mean(pred_valid)), 3),
            "pred_ending_0000": round(float(np.mean([r["pred"].endswith("0000") for r in rs])), 3),
            "national_digit_errors": len(nat),
            "national_digit_error_breakdown": dict(cats),
            "exact_matches_on_obsolete_gold": sum(r["gold"] not in ten for r in exact),
            "accuracy_10_when_gold_valid": round(float(np.mean([r["exact_10"] for r in rs if r["gold"] in ten])), 3),
            "n_gold_valid": int(sum(gold_valid)),
        }
    return out


# 5 ------------------------------------------------------------------------------------------
def oss_audit(root, codes):
    ten = codes[0]
    gen = [json.loads(l) for l in open(Path(root) / "test" / "generations_B.jsonl")]
    ok, _ = scored(root, "test")
    b = [r for r in ok if r["model"] == "B"]
    a = [r for r in ok if r["model"] == "A"]
    meta = [h.get("metadata") or {} for g in gen for h in g["history"][-1:]]
    reasoning = [((m.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") for m in meta]
    reasoning = [x for x in reasoning if x is not None]
    return {
        "resolved_models": dict(Counter(m.get("resolved_model") for m in meta)),
        "stop_reasons": dict(Counter(m.get("stop_reason") for m in meta)),
        "median_reasoning_tokens": float(np.median(reasoning)) if reasoning else None,
        "pred_valid_2026": {"gpt_oss": round(float(np.mean([r["pred"] in ten for r in b])), 3),
                            "claude": round(float(np.mean([r["pred"] in ten for r in a])), 3)},
        "pred_ending_0000": {"gpt_oss": round(float(np.mean([r["pred"].endswith("0000") for r in b])), 3),
                             "claude": round(float(np.mean([r["pred"].endswith("0000") for r in a])), 3)},
        "pred_ending_000000": round(float(np.mean([r["pred"].endswith("000000") for r in b])), 3),
        "prefix_accuracy_gpt_oss": {d: round(float(np.mean([r["prefix_correct"][d] for r in b])), 3)
                                    for d in ("2", "4", "6", "8", "10")},
        "distinct_confidences": len({r["confidence"] for r in b}),
        "examples": [{"case": r["case_id"], "gold": r["gold"], "pred": r["pred"], "conf": r["confidence"],
                      "tier": r["severity_tier"], "rationale": r["rationale"][:220]} for r in b[:6]],
    }


# 6 ------------------------------------------------------------------------------------------
def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rq3_cluster(root):
    from analysis.rq3_judge import validated_rows, clustered_auc
    rows, _ = validated_rows(Path(root))
    result = clustered_auc(rows, N_BOOT, SEED)
    products = Counter(r["prediction_id"].rsplit("|", 1)[0] for r in rows)
    return {"n_predictions": len(rows), "n_products": len(products),
            "products_in_two_predictions": sum(n > 1 for n in products.values()),
            "auc": round(result["auc"], 3) if result["auc"] is not None else None,
            "auc_ci95_product_clustered": [round(x, 3) for x in result["ci95"]] if result["ci95"] else None,
            "correct_by_majority_label": dict(Counter(r["majority"] for r in rows if r["correct"]))}


# 7 ------------------------------------------------------------------------------------------
def worked_cases(root, codes):
    ten = codes[0]
    ok, _ = scored(root, "test")
    rq3 = {json.loads(l)["prediction_id"]: json.loads(l)
           for l in open(Path(root) / "rq3" / "judge_case_scored.jsonl")}
    by_pid = {r["prediction_id"]: r for r in ok}

    def card(r, why):
        j = rq3.get(r["prediction_id"], {})
        return {"why": why, "prediction_id": r["prediction_id"], "description": r["description"][:400],
                "gold": r["gold"], "pred": r["pred"], "tier": r["severity_tier"],
                "confidence": r["confidence"], "calibrated": round(r["calibrated_confidence"], 3),
                "gold_in_2026": r["gold"] in ten, "pred_in_2026": r["pred"] in ten,
                "evaluator": j.get("majority"), "rationale": r["rationale"][:400]}

    cases = []
    sub = sorted([r for r in ok if r["severity_tier"] == "wrong_stat_suffix" and r["pred"] in ten and r["gold"] in ten],
                 key=lambda r: (-r["confidence"], hkey(r["prediction_id"])))
    cases.append(card(sub[0], "correct to 8 digits, wrong statistical suffix, both codes valid in 2026"))
    wc = sorted([r for r in ok if r["severity_tier"] == "wrong_chapter"],
                key=lambda r: (-r["confidence"], hkey(r["prediction_id"])))
    cases.append(card(wc[0], "most confident wrong-chapter prediction"))
    unc = [by_pid[p] for p, j in sorted(rq3.items()) if str(j["correct"]) == "True" and j["majority"] == "uncertain"]
    cases.append(card(sorted(unc, key=lambda r: -r["confidence"])[0], "benchmark-correct, evaluator uncertain"))
    obs = sorted([r for r in ok if r["gold"] not in ten and not r["exact_10"] and r["deepest_correct"] >= 6],
                 key=lambda r: hkey(r["prediction_id"]))
    if obs:
        cases.append(card(obs[0], "gold code no longer in the 2026 schedule"))
    nm = [by_pid[p] for p, j in sorted(rq3.items()) if j["stratum"] == "near_miss" and j["majority"] == "supported"]
    if nm:
        cases.append(card(nm[0], "near miss judged supported by the evaluator"))
    return cases


# 9 ------------------------------------------------------------------------------------------
def tie_averaged_risk(conf, correct, coverage):
    """Selective risk at `coverage` with ties at the cut resolved in expectation.

    The frozen pipeline breaks ties by a hash of the prediction ID. Verbalised confidence
    takes few distinct values, so many cases can tie at the cut; this gives the expected
    risk if the tied cases were drawn at random instead.
    """
    x, y = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    k = max(1, int(math.floor(len(x) * coverage + 0.5)))
    taken, hits = 0, 0.0
    for s in sorted(set(x), reverse=True):
        m = x == s
        n = int(m.sum())
        share = min(1.0, (k - taken) / n)
        hits += share * y[m].sum()
        taken += min(n, k - taken)
        if taken >= k:
            break
    return 1 - hits / k


def tie_averaged_routing(root, coverage=0.7):
    ok, _ = scored(root, "test")
    y = np.array([1.0 if r["correct"] else 0.0 for r in ok])
    raw = np.array([r["confidence"] for r in ok])
    cal = np.array([r["calibrated_confidence"] for r in ok])
    groups = defaultdict(list)
    for i, r in enumerate(ok):
        groups[r["case_id"]].append(i)
    keys = list(groups)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = [i for j in rng.integers(0, len(keys), len(keys)) for i in groups[keys[j]]]
        diffs.append(tie_averaged_risk(raw[idx], y[idx], coverage) - tie_averaged_risk(cal[idx], y[idx], coverage))
    r_raw, r_cal = tie_averaged_risk(raw, y, coverage), tie_averaged_risk(cal, y, coverage)
    out = {"pooled": {"risk_raw": round(r_raw, 3), "risk_calibrated": round(r_cal, 3),
                      "raw_minus_calibrated": round(r_raw - r_cal, 3),
                      "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])]}}
    for m, rs in sorted(by_model(ok).items()):
        c = [r["confidence"] for r in rs]
        k = max(1, int(math.floor(len(c) * coverage + 0.5)))
        cut = sorted(c, reverse=True)[k - 1]
        out[m] = {"risk_raw": round(tie_averaged_risk(c, [r["correct"] for r in rs], coverage), 3),
                  "risk_calibrated": round(tie_averaged_risk([r["calibrated_confidence"] for r in rs],
                                                             [r["correct"] for r in rs], coverage), 3),
                  "tied_at_cut": sum(1 for v in c if v == cut),
                  "needed_from_tie": k - sum(1 for v in c if v > cut)}
    return out


# 10 -----------------------------------------------------------------------------------------
def national_suffix_ceiling(root, codes):
    """Candidate discrepancies and a separate gold-release-aligned recovery bound.
    Neither quantity is an observed improvement from a repair model.
    """
    ten = codes[0]
    lines_under = defaultdict(list)
    for t in ten:
        lines_under[t[:8]].append(t)
    out = {}
    for split in ("validation", "test"):
        ok, _ = scored(root, split)
        for m, rs in sorted(by_model(ok).items()):
            inv = [r for r in rs if r["pred"] not in ten]
            same = sum(r["pred"][:8] == r["gold"][:8] for r in inv)
            snap = sum(len(lines_under[r["pred"][:8]]) == 1 and lines_under[r["pred"][:8]][0] == r["gold"]
                       for r in inv)
            out.setdefault(split, {})[m] = {
                "n": len(rs), "exact_10": round(sum(r["exact_10"] for r in rs) / len(rs), 3),
                "invalid_pred": len(inv), "same_rate_line": same,
                "candidate_discrepancy_pts": round(100 * same / len(rs), 1),
                "release_aligned_ceiling_pts": round(100 * sum(r["pred"][:8] == r["gold"][:8] and r["gold"] in ten for r in inv) / len(rs), 1),
                "single_line_snap": snap, "snap_gain_pts": round(100 * snap / len(rs), 1)}
    return out


# 11 -----------------------------------------------------------------------------------------
def rq3_withholding(root):
    read = lambda f: [json.loads(l) for l in open(Path(root) / "rq3" / f)]
    truth = {t["prediction_id"]: t for t in read("judge_case_truth.jsonl")}
    res = {r["prediction_id"]: r for r in read("judge_results.jsonl")}
    out = {}
    for key, flag in (("withheld", True), ("none_withheld", False)):
        rs = [res[p] for p, t in truth.items()
              if t["correct"] and (t["same_gold_code_rulings_skipped"] > 0) == flag]
        sup = sum(r["majority"] == "supported" for r in rs)
        out[key] = {"n_correct": len(rs), "supported": sup, "not_supported": len(rs) - sup}
    out["mean_rulings_withheld"] = round(
        float(np.mean([t["same_gold_code_rulings_skipped"] for t in truth.values()])), 2)
    out["mean_rulings_in_pack"] = round(float(np.mean([t["rulings_in_pack"] for t in truth.values()])), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hts", required=True)
    args = ap.parse_args()
    codes = hts_codes(args.hts)
    out = {"hts_codes_10digit": len(codes[0])}
    for s in STUDIES:
        root = f"results/{s}"
        out[s] = {"calibration_change": paired_calibration_change(root),
                  "constant_prevalence_baseline": constant_prevalence_baseline(root),
                  "threshold_transfer": threshold_transfer(root),
                  "model_priority_baseline": model_priority_baseline(root),
                  "version_audit": version_audit(root, codes),
                  "tie_averaged_routing": tie_averaged_routing(root),
                  "national_suffix_ceiling": national_suffix_ceiling(root, codes)}
    out["oss_v1"]["oss_audit"] = oss_audit("results/oss_v1", codes)
    out["main_v1"]["rq3_cluster"] = rq3_cluster("results/main_v1")
    out["main_v1"]["worked_cases"] = worked_cases("results/main_v1", codes)
    out["main_v1"]["rq3_withholding"] = rq3_withholding("results/main_v1")
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
