"""Only persisted outputs. No synthetic/demo fallback; RQ3 can be gated."""

import csv
from pathlib import Path
from study_io import *
from pipeline import Study


def build_report(root, out=None, require_human=False):
    s = Study(root)
    root = Path(root)
    out = Path(out) if out else root / "report"
    joined = {
        sp: {k: s.joined(sp, k) for k in ("A", "B")} for sp in ("validation", "test")
    }
    s.verify("metrics", joined["test"])
    s.verify("calibration", joined)
    scored = {sp: s.scored(sp) for sp in ("validation", "test")}
    s.verify("routing", scored)
    hv = (
        read(root / "human_validation.json")
        if (root / "human_validation.json").exists()
        else None
    )
    if hv:
        s.verify("judge", s.judge_cases())
        s.verify(
            "human",
            {
                "judge": lines(root / "judge_results.jsonl"),
                "key": read(root / "annotations" / "key.json"),
            },
        )
    if require_human and not hv:
        raise ValueError("Independent human annotations required for final RQ3 report")
    out.mkdir(parents=True, exist_ok=True)
    m = read(root / "metrics.json")
    c = read(root / "calibration.json")
    r = read(root / "routing.json")
    table = [
        {
            "model": k,
            "n_all": m[k]["n_all"],
            "n_valid": m[k]["n_valid"],
            "n_failed": m[k]["n_failed"],
            "n_missing": m[k]["n_missing"],
            "all_case_exact10": m[k]["all_case_exact10_accuracy"],
            "raw_ece": c[k]["raw"]["ece"],
            "calibrated_ece": c[k]["calibrated"]["ece"],
            "raw_brier": c[k]["raw"]["brier"],
            "calibrated_brier": c[k]["calibrated"]["brier"],
        }
        for k in ("A", "B")
    ]
    for name, rows in [
        ("model_metrics", table),
        (
            "routing_curve",
            [{"score": key, **p} for key in ("raw", "cal") for p in r["curve_" + key]],
        ),
    ]:
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    text = [
        "# Persisted study outputs",
        "",
        "**Not legal advice. No generated demo observations are used.**",
        "Model/dataset/evidence provenance must still be verified by the researcher.",
        "",
        "## Classification and reliability",
        "",
        "| Model | Retained cases | Valid | All-case exact-10 | Raw ECE | Calibrated ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in table:
        text.append(
            f'| {x["model"]} | {x["n_all"]} | {x["n_valid"]} | {x["all_case_exact10"]:.4f} | {x["raw_ece"]:.4f} | {x["calibrated_ece"]:.4f} |'
        )
    text += [
        "",
        "Failed/missing predictions count as wrong in all-case accuracy. Calibration conditions on valid outputs.",
        "Structural hierarchy severity is not measured financial or legal loss.",
        "",
        "## Routing interpretation",
        "",
        f'Unit: {r["unit"]}. This is model–case recommendation coverage, NOT distinct-product workload.',
        f'Raw minus calibrated risk at requested coverage {r["primary_coverage"]:.0%}: {r["improvement"]:.6f}.',
        f'Conditional product-cluster bootstrap interval: {r["ci95_raw_minus_calibrated"]}.',
        "Matched coverage is retrospective and conditions on valid outputs; all-prediction coverage is separately exported.",
        "Failed/missing outputs are forced escalations. Ties and integer rounding can change achieved coverage.",
        "Deployment threshold is selected only on validation, then frozen for test. It has NO finite-sample risk guarantee.",
        "Increasing within-model calibration need not improve ranking; per-model invariance controls are in routing.json.",
        "Intervals condition on fitted calibrators, not the uncertainty of calibration fitting.",
        "",
        "## Evidence support",
        "",
    ]
    if hv:
        text += [
            f'Independent annotations: {hv["n"]}; comparable decisions: {hv["n_comparable"]}; judge abstentions: {hv["n_judge_abstained"]}.',
            f'Agreement: {hv["agreement"]}; kappa: {hv["kappa"]}; kappa CI: {hv["kappa_ci95"]}.',
            "Annotation completeness is NOT a declaration that the judge is valid or legally correct.",
        ]
    else:
        text += [
            "**RQ3 UNVALIDATED: independent human annotations have not been supplied.**"
        ]
    jr = lines(root / "judge_results.jsonl")
    if jr:
        text += [
            f'Same-model generator/evaluator units: {sum(bool(x.get("judge_is_generator")) for x in jr)}. Disclose this dependency.'
        ]
    text += [
        "",
        "## Audit trail",
        "",
        "Keep protocol, source hashes, environment, split audits, evidence manifests, raw generation/judge histories, calibrator states and requests.sqlite.",
        "Do not upload API keys, proprietary inputs or confidential annotations to public repositories.",
    ]
    atomic(out / "study_report.md", "\n".join(text) + "\n")
    needed = [
        "protocol.json",
        "environment.json",
        "metrics.json",
        "calibration.json",
        "routing.json",
        "paired_accuracy.json",
    ]
    if hv:
        needed += ["human_validation.json", "judge_results.jsonl"]
    write(
        out / "report_inputs.json",
        {
            "sha256": {f: file_hash(root / f) for f in needed},
            "human_annotations_complete": bool(hv),
            "judge_validity_established": False,
        },
    )
    return {
        "report": str(out / "study_report.md"),
        "human_annotations_complete": bool(hv),
        "judge_validity_established": False,
    }
