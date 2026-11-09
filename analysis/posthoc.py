"""
posthoc.py
----------
Post-hoc analysis of a COMPLETED study root (after `report`). Reads the scored
outputs through pipeline.Study, which re-verifies the protocol hash, the code
freeze and the generation/calibration seals; the routing seal is checked too.
Makes no API calls and writes nothing outside <root>/posthoc/.

Conventions shared with the pipeline:
  * ties in confidence are broken exactly as routing.py does (SHA-256 of the
    prediction_id), so per-model selective risk equals routing.json within_model;
  * "major error" = wrong chapter (metrics.major_error_rate, proposal RQ1);
    wrong heading or chapter is reported separately as heading_or_worse.

Adds the proposal outputs the pipeline does not produce itself:
  * bootstrap 95% CIs for accuracy, severity, ECE, Brier and selective risk (Sec. 7.4);
  * paired bootstrap A-B differences (RQ1 contrast);
  * reliability diagrams, raw vs calibrated, per model (RQ2, Sec. 7.3 Stage 6);
  * risk-coverage curves per model and pooled (RQ2, Stage 7);
  * error-severity distribution figure (RQ1, Table 5).

Lives in analysis/ so it is outside the code freeze (study_io hashes top-level *.py only).

    python analysis/posthoc.py --root results/main_v1
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import bootstrap_ci, bootstrap_difference, eval_records, major_error_rate  # noqa: E402
from calibration import bootstrap_metric_ci, ece, brier  # noqa: E402
from routing import routing_at_coverage, risk_coverage_curve  # noqa: E402
from hts_hierarchy import SEVERITY_ORDER  # noqa: E402
from pipeline import Study  # noqa: E402
import reporting  # noqa: E402

N_BOOT = 2000
SEED = 7
TIERS = [name for name, _ in SEVERITY_ORDER]
HEADING_OR_WORSE = ("wrong_heading", "wrong_chapter")


def heading_or_worse_rate(evals):
    return float(np.mean([e["severity_tier"] in HEADING_OR_WORSE for e in evals]))


def ci(point_lo_hi):
    p, lo, hi = point_lo_hi
    return {"point": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def fmt(d):
    return f'{d["point"]:.3f} [{d["ci95"][0]:.3f}, {d["ci95"][1]:.3f}]'


def selective_risk_ci(conf, correct, ids, coverage):
    """Case-resampling bootstrap of selective risk at fixed coverage (one model).
    Resampled copies get draw-prefixed IDs, as in routing.bootstrap_routing_gap."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, int)
    rng = np.random.default_rng(SEED)
    point = routing_at_coverage(conf, correct, coverage, ids)["selective_risk"]
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(conf), len(conf))
        boot_ids = [f"{draw}|{ids[j]}" for draw, j in enumerate(i)]
        boots.append(routing_at_coverage(conf[i], correct[i], coverage, boot_ids)["selective_risk"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def analyse(root: Path):
    study = Study(root)  # protocol hash + code-freeze check
    scored = {s: study.scored(s) for s in ("validation", "test")}  # generation + calibration seals
    study.verify("routing", scored)
    protocol = study.protocol
    routing = json.loads((root / "routing.json").read_text())
    out_dir = root / "posthoc"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cov = protocol["coverage"]

    test = {m: [r for r in scored["test"] if r["model"] == m] for m in ("A", "B")}
    names = {m: protocol["models"][m]["model"] for m in ("A", "B")}
    res = {"root": str(root), "models": names, "n_boot": N_BOOT, "per_model": {}}

    # ---- RQ1: accuracy and severity, per model --------------------------------
    for m, rows in test.items():
        valid = [r for r in rows if r["status"] == "ok"]
        ev = eval_records(valid)
        allcase = [{"correct": int(r["correct"])} for r in rows]
        res["per_model"][m] = {
            "model": names[m],
            "n_all": len(rows),
            "n_valid": len(valid),
            "n_failed": len(rows) - len(valid),
            "exact_10_all_case": ci(bootstrap_ci(allcase, lambda e: np.mean([x["correct"] for x in e]), N_BOOT, seed=SEED)),
            "exact_10_valid": ci(bootstrap_ci(ev, lambda e: np.mean([x["exact_10"] for x in e]), N_BOOT, seed=SEED)),
            "exact_6_valid": ci(bootstrap_ci(ev, lambda e: np.mean([x["exact_6"] for x in e]), N_BOOT, seed=SEED)),
            "major_error_rate_valid": ci(bootstrap_ci(ev, major_error_rate, N_BOOT, seed=SEED)),
            "heading_or_worse_error_rate_valid": ci(bootstrap_ci(ev, heading_or_worse_rate, N_BOOT, seed=SEED)),
            "mean_severity_ordinal_valid": ci(bootstrap_ci(ev, lambda e: np.mean([x["severity_ordinal"] for x in e]), N_BOOT, seed=SEED)),
        }

    # ---- RQ1: paired A-B differences ------------------------------------------
    a_all = sorted(test["A"], key=lambda r: r["case_id"])
    b_all = sorted(test["B"], key=lambda r: r["case_id"])
    both_ok = {r["case_id"] for r in a_all if r["status"] == "ok"} & {r["case_id"] for r in b_all if r["status"] == "ok"}
    a_ok = eval_records([r for r in a_all if r["case_id"] in both_ok])
    b_ok = eval_records([r for r in b_all if r["case_id"] in both_ok])
    for e, r in zip(a_ok, [r for r in a_all if r["case_id"] in both_ok]):
        e["case_id"] = r["case_id"]
    for e, r in zip(b_ok, [r for r in b_all if r["case_id"] in both_ok]):
        e["case_id"] = r["case_id"]
    acc = lambda e: np.mean([x["correct"] for x in e])
    res["paired_A_minus_B"] = {
        "exact_10_all_case": ci(bootstrap_difference(
            [{"case_id": r["case_id"], "correct": int(r["correct"])} for r in a_all],
            [{"case_id": r["case_id"], "correct": int(r["correct"])} for r in b_all],
            acc, N_BOOT, seed=SEED)),
        "n_both_valid": len(both_ok),
        "major_error_rate_both_valid": ci(bootstrap_difference(
            a_ok, b_ok, major_error_rate, N_BOOT, seed=SEED)),
        "heading_or_worse_error_rate_both_valid": ci(bootstrap_difference(
            a_ok, b_ok, heading_or_worse_rate, N_BOOT, seed=SEED)),
        "mean_severity_ordinal_both_valid": ci(bootstrap_difference(
            a_ok, b_ok, lambda e: np.mean([x["severity_ordinal"] for x in e]), N_BOOT, seed=SEED)),
    }
    reporting.plot_severity_distribution(
        {names[m]: [r for r in test[m] if r["status"] == "ok"] for m in test},
        str(fig_dir / "rq1_severity_distribution.png"))

    # ---- RQ2: calibration and routing, per model ------------------------------
    for m, rows in test.items():
        valid = [r for r in rows if r["status"] == "ok"]
        raw = [r["confidence"] for r in valid]
        cal = [r["calibrated_confidence"] for r in valid]
        y = [int(r["correct"]) for r in valid]
        ids = [r["prediction_id"] for r in valid]
        pm = res["per_model"][m]
        pm["ece_raw"] = ci(bootstrap_metric_ci(raw, y, "ece", N_BOOT, seed=SEED))
        pm["ece_calibrated"] = ci(bootstrap_metric_ci(cal, y, "ece", N_BOOT, seed=SEED))
        pm["brier_raw"] = ci(bootstrap_metric_ci(raw, y, "brier", N_BOOT, seed=SEED))
        pm["brier_calibrated"] = ci(bootstrap_metric_ci(cal, y, "brier", N_BOOT, seed=SEED))
        pm[f"selective_risk_raw_at_{cov:.0%}"] = ci(selective_risk_ci(raw, y, ids, cov))
        pm[f"selective_risk_calibrated_at_{cov:.0%}"] = ci(selective_risk_ci(cal, y, ids, cov))
        for key, name in (("raw", "selective_risk_raw"), ("calibrated", "selective_risk_calibrated")):
            if not np.isclose(pm[f"{name}_at_{cov:.0%}"]["point"], routing["within_model"][m][name], atol=5e-5):
                raise ValueError(f"Model {m} {key} selective risk disagrees with routing.json")
        pm["distinct_raw_confidence_values"] = len(set(raw))
        reporting.plot_reliability(raw, y, str(fig_dir / f"rq2_reliability_{m}.png"), calibrated=cal)
        reporting.plot_risk_coverage(
            risk_coverage_curve(raw, y, protocol["coverages"], ids),
            risk_coverage_curve(cal, y, protocol["coverages"], ids),
            str(fig_dir / f"rq2_risk_coverage_{m}.png"), cov)
    reporting.plot_risk_coverage(routing["curve_raw"], routing["curve_cal"],
                                 str(fig_dir / "rq2_risk_coverage_pooled.png"), cov)
    res["pooled_routing"] = {
        "selective_risk_raw": routing["selective_risk_raw"],
        "selective_risk_calibrated": routing["selective_risk_calibrated"],
        "improvement": routing["improvement"],
        "ci95_raw_minus_calibrated": routing["ci95_raw_minus_calibrated"],
    }

    (out_dir / "posthoc.json").write_text(json.dumps(res, indent=2))
    (out_dir / "summary.md").write_text(summary_md(res, cov))
    return res


def summary_md(res, cov):
    pm = res["per_model"]
    rows = [
        ("Retained / valid / failed", lambda d: f'{d["n_all"]} / {d["n_valid"]} / {d["n_failed"]}'),
        ("Exact-10, all cases (failed = wrong)", lambda d: fmt(d["exact_10_all_case"])),
        ("Exact-6, valid outputs", lambda d: fmt(d["exact_6_valid"])),
        ("Major error rate (wrong chapter)", lambda d: fmt(d["major_error_rate_valid"])),
        ("Wrong heading or chapter", lambda d: fmt(d["heading_or_worse_error_rate_valid"])),
        ("Mean severity ordinal (0 = exact)", lambda d: fmt(d["mean_severity_ordinal_valid"])),
        ("ECE raw", lambda d: fmt(d["ece_raw"])),
        ("ECE calibrated", lambda d: fmt(d["ece_calibrated"])),
        ("Brier raw", lambda d: fmt(d["brier_raw"])),
        ("Brier calibrated", lambda d: fmt(d["brier_calibrated"])),
        (f"Selective risk @ {cov:.0%}, raw", lambda d: fmt(d[f"selective_risk_raw_at_{cov:.0%}"])),
        (f"Selective risk @ {cov:.0%}, calibrated", lambda d: fmt(d[f"selective_risk_calibrated_at_{cov:.0%}"])),
        ("Distinct raw confidence values", lambda d: str(d["distinct_raw_confidence_values"])),
    ]
    out = ["# Post-hoc analysis (test split, 95% bootstrap CIs)", "",
           f"| Metric | A: {res['models']['A']} | B: {res['models']['B']} |", "|---|---|---|"]
    out += [f"| {label} | {f(pm['A'])} | {f(pm['B'])} |" for label, f in rows]
    p = res["paired_A_minus_B"]
    po = res["pooled_routing"]
    out += ["", "## Paired A - B (same test cases)", "",
            f"- Exact-10, all cases: {fmt(p['exact_10_all_case'])}",
            f"- Major error rate, wrong chapter (n = {p['n_both_valid']} cases valid for both): {fmt(p['major_error_rate_both_valid'])}",
            f"- Wrong heading or chapter: {fmt(p['heading_or_worse_error_rate_both_valid'])}",
            f"- Mean severity ordinal: {fmt(p['mean_severity_ordinal_both_valid'])}",
            "", "## Pooled routing (pipeline result)", "",
            f"- Selective risk raw {po['selective_risk_raw']:.3f}, calibrated {po['selective_risk_calibrated']:.3f}, "
            f"raw - calibrated {po['improvement']:.3f}, 95% CI {po['ci95_raw_minus_calibrated']}",
            "", "Per-model raw and calibrated selective risk coincide by construction: a non-decreasing",
            "calibrator preserves each model's ranking (except ties). Any routing difference comes from pooling.",
            "Ties in confidence are broken by SHA-256 of the prediction ID, as in the pipeline's routing.py.",
            "", "Figures: `posthoc/figures/`."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    analyse(Path(a.root))
    print((Path(a.root) / "posthoc" / "summary.md").read_text())
