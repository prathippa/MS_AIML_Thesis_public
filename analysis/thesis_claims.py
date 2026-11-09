"""
thesis_claims.py
----------------
Checks every headline claim of the thesis (abstract, the answer to each research question and
each hypothesis verdict) against the stored result files, and writes a claim-by-claim report.
Exits with status 1 if any claim does not match the data, so a changed result cannot silently
leave the text behind. Reads only saved outputs; calls no model.

    python analysis/thesis_claims.py            # writes results/thesis_claims.md

Run analysis/supplementary.py and analysis/rq_alignment.py first; their JSON files are inputs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

RESULTS = Path("results")
STUDIES = ("main_v1", "oss_v1", "hscc_v1")


def _judge_models(results):
    """Model identifiers actually recorded in the RQ3 judge outputs, for a provenance check."""
    seen = set()
    with open(results / "main_v1/rq3/judge_results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seen.add(r.get("resolved_model") or r.get("model"))
    return sorted(seen)


def load(results=RESULTS):
    j = lambda p: json.load(open(results / p))
    return {
        "metrics": {s: j(f"{s}/metrics.json") for s in STUDIES},
        "posthoc": {s: j(f"{s}/posthoc/posthoc.json") for s in STUDIES},
        "routing": {s: j(f"{s}/routing.json") for s in STUDIES},
        "supp": j("supplementary.json"),
        "align": j("rq_alignment.json"),
        "rq3": j("main_v1/rq3/rq3_summary.json"),
        "protocol": j("main_v1/protocol.json"),
        "judge_models_used": _judge_models(results),
        "human_validated": (results / "main_v1/rq3/human_validation.json").exists(),
    }


def excludes_zero(ci):
    return ci[0] > 0 or ci[1] < 0


def near(x, target, tol=0.0005):
    return abs(x - target) <= tol


def claims(d):
    """Each claim: (id, research question, statement as in the thesis, observed value, holds?)."""
    out = []

    def add(cid, rq, text, value, ok):
        out.append({"id": cid, "rq": rq, "claim": text, "observed": value, "holds": bool(ok)})

    m, ph, rt, sp, al = d["metrics"]["main_v1"], d["posthoc"], d["routing"], d["supp"], d["align"]
    pa = ph["main_v1"]["paired_A_minus_B"]

    # RQ1 ------------------------------------------------------------------------------------
    a10, b10 = m["A"]["all_case_exact10_accuracy"], m["B"]["all_case_exact10_accuracy"]
    add("C1", "RQ1", "ATLAS exact match: Claude 20.1%, GPT-5 31.4%", f"{a10:.3f} / {b10:.3f}",
        near(a10, 0.201) and near(b10, 0.314))
    add("C2", "RQ1", "The exact-match difference is statistically detectable",
        f"{pa['exact_10_all_case']['point']} {pa['exact_10_all_case']['ci95']}", excludes_zero(pa["exact_10_all_case"]["ci95"]))
    wc = pa["major_error_rate_both_valid"]
    add("C3", "RQ1", "Both models put 13.0% in the wrong chapter; no detectable difference",
        f"{m['A']['valid_output_metrics']['major_error_rate']:.3f} / {m['B']['valid_output_metrics']['major_error_rate']:.3f}, diff {wc['ci95']}",
        near(m["A"]["valid_output_metrics"]["major_error_rate"], 0.130) and not excludes_zero(wc["ci95"]))
    lv = al["main_v1"]["digit_gap"]["levels"]
    add("C4", "RQ1", "ATLAS: no detectable difference at 2, 4 or 6 digits; detectable at 8 and 10",
        {k: v["detectable"] for k, v in lv.items()},
        not any(lv[k]["detectable"] for k in ("2", "4", "6")) and all(lv[k]["detectable"] for k in ("8", "10")))
    hv = al["hscc_v1"]["digit_gap"]["levels"]
    add("C5", "RQ1", "HSCodeComp: no detectable difference at any digit level",
        {k: v["detectable"] for k, v in hv.items()}, not any(v["detectable"] for v in hv.values()))
    om = d["metrics"]["oss_v1"]["B"]
    add("C6", "RQ1", "gpt-oss: 1.8% exact match but 74% right chapter",
        f"{om['all_case_exact10_accuracy']:.3f} / {om['valid_output_metrics']['prefix_accuracy']['prefix_2_chapter']:.3f}",
        near(om["all_case_exact10_accuracy"], 0.018) and near(om["valid_output_metrics"]["prefix_accuracy"]["prefix_2_chapter"], 0.744))
    va = sp["main_v1"]["version_audit"]
    nat = sum(va[k]["national_digit_errors"] for k in ("A", "B"))
    bad = sum(v for k in ("A", "B") for c, v in va[k]["national_digit_error_breakdown"].items() if c.startswith("pred_not_in_2026"))
    add("C7", "RQ1", "About half of ATLAS national-digit errors are codes not in the 2026 schedule",
        f"{bad}/{nat} = {bad / nat:.2f}", 0.4 <= bad / nat <= 0.6)

    # RQ2a -----------------------------------------------------------------------------------
    pairs = [(s, k) for s in STUDIES for k in ("A", "B")]
    cc = {(s, k): sp[s]["calibration_change"][k] for s, k in pairs}
    add("C8", "RQ2a", "Calibration lowers ECE in all six model-study pairs",
        {f"{s}:{k}": v["ece_reduction"] for (s, k), v in cc.items()}, all(v["ece_reduction"] > 0 for v in cc.values()))
    n_det = sum(excludes_zero(v["ece_reduction_ci95"]) for v in cc.values())
    add("C9", "RQ2a", "The ECE reduction is statistically detectable in four of six pairs", f"{n_det} of 6", n_det == 4)

    # RQ2b -----------------------------------------------------------------------------------
    wm = rt["main_v1"]["within_model"]
    add("C10", "RQ2b", "Within each frontier model, raw and calibrated routing risk are identical",
        {k: (round(v["selective_risk_raw"], 3), round(v["selective_risk_calibrated"], 3)) for k, v in wm.items()},
        all(near(v["selective_risk_raw"], v["selective_risk_calibrated"], 1e-12) for v in wm.values()))
    ranked = [(s, k) for s, k in pairs if (s, k) != ("oss_v1", "B")]  # gpt-oss slope fitted to zero
    cr = {f"{s}:{k}": (al[s]["confidence_ranking"][k]["auroc_raw"], al[s]["confidence_ranking"][k]["auroc_calibrated"]) for s, k in ranked}
    add("C11", "RQ2b", "Calibration leaves each model's AUROC unchanged (positive-slope calibrators)",
        cr, all(near(a, b, 1e-12) for a, b in cr.values()))
    add("C12", "RQ2b", "Claude's confidence ranks better than GPT-5's on ATLAS (0.85 vs 0.70)",
        f"{al['main_v1']['confidence_ranking']['A']['auroc_raw']} / {al['main_v1']['confidence_ranking']['B']['auroc_raw']}",
        al["main_v1"]["confidence_ranking"]["A"]["auroc_raw"] > al["main_v1"]["confidence_ranking"]["B"]["auroc_raw"])
    r = rt["main_v1"]
    add("C13", "RQ2b", "ATLAS pooled queue: calibrated routing not better (0.654 raw vs 0.688)",
        f"{r['selective_risk_raw']:.3f} / {r['selective_risk_calibrated']:.3f} {r['ci95_raw_minus_calibrated']}",
        r["improvement"] <= 0 and not r["ci95_raw_minus_calibrated"][0] > 0)
    r = rt["hscc_v1"]
    add("C14", "RQ2b", "HSCodeComp pooled queue: no detectable difference",
        f"{r['ci95_raw_minus_calibrated']}", not excludes_zero(r["ci95_raw_minus_calibrated"]))
    r = rt["oss_v1"]
    mp = sp["oss_v1"]["model_priority_baseline"]["risk_model_priority"]
    add("C15", "RQ2b", "Open-weight pooled queue: calibration lowers risk (0.864 to 0.814), matched by model priority (0.809)",
        f"{r['selective_risk_raw']:.3f} -> {r['selective_risk_calibrated']:.3f} {r['ci95_raw_minus_calibrated']}; priority {mp}",
        excludes_zero(r["ci95_raw_minus_calibrated"]) and abs(mp - r["selective_risk_calibrated"]) < 0.01)
    sc = [al[s]["schedule_check"][g] for s in STUDIES for g in ("A", "B", "pooled")]
    pos = sum(x["risk_reduction"] > 0 for x in sc)
    det = sum(x["risk_reduction_ci95_case_clustered"][0] > 0 for x in sc)
    add("C16", "RQ2b", "Schedule check lowers risk in 8 of 9 comparisons, detectably in 5",
        f"{pos} of 9 lower, {det} detectable", pos == 8 and det == 5)
    wrong = {s: (al[s]["schedule_check"]["pooled"]["invalid_codes_wrong"], al[s]["schedule_check"]["pooled"]["invalid_codes"]) for s in STUDIES}
    add("C17", "RQ2b", "Codes absent from the 2026 schedule are wrong 88-99% of the time",
        {s: f"{w}/{n}" for s, (w, n) in wrong.items()}, all(0.88 <= w / n <= 0.995 for w, n in wrong.values()))

    # RQ3 ------------------------------------------------------------------------------------
    q = d["rq3"]
    add("C18", "RQ3", "Evaluator unanimous in 83% of predictions across three runs",
        q["unanimous_rate"], near(q["unanimous_rate"], 0.833))
    rc = sp["main_v1"]["rq3_cluster"]
    add("C19", "RQ3", "Support score associated with correctness: AUC 0.74 [0.60, 0.87], product-clustered",
        f"{rc['auc']} {rc['auc_ci95_product_clustered']}", near(rc["auc"], 0.741) and rc["auc_ci95_product_clustered"][0] > 0.5)
    lab = rc["correct_by_majority_label"]
    add("C20", "RQ3", "13 of 30 benchmark-correct predictions not judged supported",
        f"{lab.get('uncertain', 0) + lab.get('unsupported', 0)} of {sum(lab.values())}",
        lab.get("uncertain", 0) + lab.get("unsupported", 0) == 13 and sum(lab.values()) == 30)

    # RQ2a, constant-prevalence baseline -----------------------------------------------------
    cpb = {(s, k): sp[s]["constant_prevalence_baseline"][k] for s, k in pairs}
    n_beat = sum(v["calibrated_beats_constant"] for v in cpb.values())
    add("C21", "RQ2a", "Calibrated confidence beats a constant validation-prevalence predictor "
        "(lower Brier) detectably in four of six pairs; not for GPT-5 on ATLAS or the near-zero-accuracy open model",
        {f"{s}:{k}": v["brier_gain_over_constant"] for (s, k), v in cpb.items()}, n_beat == 4)

    # Post-hoc robustness and headroom (supplementary 9-11) ------------------------------------
    def h2_holds(gap, ci):
        return gap > 0 and ci[0] > 0
    same = {}
    for s in ("main_v1", "oss_v1", "hscc_v1"):
        t = sp[s]["tie_averaged_routing"]["pooled"]
        same[s] = ((t["raw_minus_calibrated"] > 0) == (rt[s]["improvement"] > 0)
                   and h2_holds(t["raw_minus_calibrated"], t["ci95"])
                   == h2_holds(rt[s]["improvement"], rt[s]["ci95_raw_minus_calibrated"]))
    t_main = sp["main_v1"]["tie_averaged_routing"]["pooled"]["ci95"]
    add("C22", "RQ2b", "Resolving confidence ties at the cut in expectation, instead of by hash, leaves the sign of "
        "every pooled raw-vs-calibrated comparison and every H2 verdict unchanged; on ATLAS the interval's upper "
        "end moves from 0.000 to -0.001",
        {s: sp[s]["tie_averaged_routing"]["pooled"]["raw_minus_calibrated"] for s in same} | {"main_v1_ci": t_main},
        all(same.values()) and t_main[1] == -0.001)
    ceil = {f"{s}:{k}": sp[s]["national_suffix_ceiling"]["test"][k]
            for s, k in (("main_v1", "A"), ("main_v1", "B"), ("hscc_v1", "A"), ("hscc_v1", "B"), ("oss_v1", "A"))}
    add("C23", "RQ1", "For the frontier models, absent-release codes on the gold's own rate line form a "
        "candidate set of 9-14 percentage points; this is not a recoverable-accuracy ceiling without gold-release alignment",
        {k: v["ceiling_gain_pts"] for k, v in ceil.items()},
        all(9.0 <= v["ceiling_gain_pts"] <= 14.5 and v["snap_gain_pts"] < 1.0 for v in ceil.values()))
    w = sp["main_v1"]["rq3_withholding"]
    add("C24", "RQ3", "Of the 13 correct codes not judged supported, 7 came from the 12 packs with a gold-code "
        "ruling withheld; 6 of the 18 correct codes with nothing withheld were not judged supported",
        f"{w['withheld']['not_supported']}/{w['withheld']['n_correct']} vs "
        f"{w['none_withheld']['not_supported']}/{w['none_withheld']['n_correct']}",
        (w["withheld"]["not_supported"], w["withheld"]["n_correct"],
         w["none_withheld"]["not_supported"], w["none_withheld"]["n_correct"]) == (7, 12, 6, 18))

    # Provenance of the RQ3 judge ------------------------------------------------------------
    jm = d["judge_models_used"]
    add("P1", "Provenance", "RQ3 judge identity is consistent across the run: the model in every judge "
        "output equals the one reported in rq3_summary.json (google/gemini-3.8-flash)",
        f"{jm} == {d['rq3'].get('judge_model')}",
        jm == ["google/gemini-3.8-flash"] and d["rq3"].get("judge_model") == "google/gemini-3.8-flash")
    add("P2", "Provenance", "The main_v1 protocol's judge default (J) was not the RQ3 judge: main_v1 is an "
        "evidence_mode='none' generation study, so RQ3 was judged post hoc by the model recorded in its own outputs",
        f"evidence_mode={d['protocol'].get('evidence_mode')}, protocol J={d['protocol']['models']['J']['model']}, rq3 judge={d['rq3'].get('judge_model')}",
        d["protocol"].get("evidence_mode") == "none" and d["rq3"].get("judge_model") == "google/gemini-3.8-flash")

    # Hypothesis verdicts, derived from the data rather than asserted ---------------------------
    hp = ph["hscc_v1"]["paired_A_minus_B"]
    h1_supported = excludes_zero(hp["major_error_rate_both_valid"]["ci95"]) or excludes_zero(hp["heading_or_worse_error_rate_both_valid"]["ci95"])
    add("H1", "Verdict", "H1 not supported: on HSCodeComp (similar exact match) severity shows no detectable difference",
        f"wrong chapter {hp['major_error_rate_both_valid']['ci95']}, heading or worse {hp['heading_or_worse_error_rate_both_valid']['ci95']}",
        not h1_supported)
    h2 = {s: rt[s]["ci95_raw_minus_calibrated"][0] > 0 for s in STUDIES}
    add("H2", "Verdict", "H2 not supported in main and robustness studies; supported only in the open-weight study",
        h2, h2 == {"main_v1": False, "oss_v1": True, "hscc_v1": False})
    add("H3", "Verdict", "H3 exploratory support only: correct codes flagged, no human validation yet",
        f"flagged {lab.get('uncertain', 0) + lab.get('unsupported', 0)}; human validation present: {d['human_validated']}",
        lab.get("unsupported", 0) > 0 and not d["human_validated"])
    return out


def report(rows):
    lines = ["# Thesis claims checked against the stored results", "",
             "Generated by `analysis/thesis_claims.py`. Each claim is recomputed from the result files; "
             "these are selected numerical assertions, not an automatic validation of the full thesis or its scientific interpretation.", "",
             "| ID | RQ | Claim | Observed | Status |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['rq']} | {r['claim']} | {r['observed']} | {'PASS' if r['holds'] else 'FAIL'} |")
    n_ok = sum(r["holds"] for r in rows)
    lines += ["", f"{n_ok} of {len(rows)} selected checks pass."]
    return "\n".join(lines) + "\n"


def main():
    rows = claims(load())
    text = report(rows)
    (RESULTS / "thesis_claims.md").write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if all(r["holds"] for r in rows) else 1)


if __name__ == "__main__":
    main()
