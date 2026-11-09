"""Blinded independent annotations: completeness is NOT validation of legal correctness."""

from pathlib import Path
import csv, random
import numpy as np
from judge import LABELS, cohens_kappa
from study_io import read, write, digest

CONTENT = ("description", "predicted_code", "rationale", "evidence")


def export(cases, out, size, seed=7):
    out = Path(out)
    if (out / "key.json").exists():
        raise ValueError("Human sample already frozen")
    groups = {}
    for r in cases:
        groups.setdefault(r["case_id"], []).append(r)
    if size < 1 or len(groups) < size:
        raise ValueError("Not enough distinct products for human sample")
    rng = random.Random(seed)
    eligible = [
        rng.choice(sorted(v, key=lambda r: r["prediction_id"]))
        for _, v in sorted(groups.items())
    ]
    sample = rng.sample(eligible, size)
    out.mkdir(parents=True, exist_ok=True)
    key = []
    with (out / "blinded.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["annotation_id", *CONTENT, "label", "reason", "annotator_id"]
        )
        writer.writeheader()
        for i, r in enumerate(sample):
            aid = f"H{i+1:03d}"
            ev = "\n\n".join(
                f'[{e["id"]}] {e["text"]}\nSource: {e["source_url"]}'
                for e in r["evidence"]
            )
            content = {
                "description": r["description"],
                "predicted_code": r["predicted_code"],
                "rationale": r["rationale"],
                "evidence": ev,
            }
            writer.writerow({"annotation_id": aid, **content})
            key.append(
                {
                    "annotation_id": aid,
                    "prediction_id": r["prediction_id"],
                    "content_hash": digest(content),
                }
            )
    write(
        out / "key.json",
        {
            "mapping": key,
            "hash": digest(key),
            "sampling": "random distinct-product subset of diagnostic judge sample",
        },
    )
    return str(out / "blinded.csv")


def validate(path, key_path, judge_results, attestation, n_boot=2000, seed=7):
    for k in (
        "annotator_id",
        "expertise",
        "independent_of_generation",
        "blinded_to_gold_and_judge",
        "rubric_version",
    ):
        if k not in attestation:
            raise ValueError(f"Missing attestation: {k}")
    if (
        not str(attestation["annotator_id"]).strip()
        or not str(attestation["expertise"]).strip()
    ):
        raise ValueError("Annotator ID and expertise required")
    if (
        attestation["independent_of_generation"] is not True
        or attestation["blinded_to_gold_and_judge"] is not True
        or attestation["rubric_version"] != "v2"
    ):
        raise ValueError("Independent blinded v2 annotation required")
    key = read(key_path)
    if digest(key["mapping"]) != key["hash"]:
        raise ValueError("Human sample key changed")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by = {r["annotation_id"]: r for r in rows}
    j = {r["prediction_id"]: r for r in judge_results}
    if len(j) != len(judge_results):
        raise ValueError("Duplicate judge units")
    if len(by) != len(rows) or set(by) != {r["annotation_id"] for r in key["mapping"]}:
        raise ValueError("Annotations must match complete frozen sample")
    a = []
    b = []
    for k in key["mapping"]:
        r = by[k["annotation_id"]]
        if digest({f: r.get(f, "") for f in CONTENT}) != k["content_hash"]:
            raise ValueError(
                "Annotation content changed; edit labels/reasons/annotator IDs only"
            )
        if (
            r.get("label") not in LABELS
            or not r.get("reason", "").strip()
            or r.get("annotator_id") != attestation["annotator_id"]
        ):
            raise ValueError("Incomplete annotation")
        if k["prediction_id"] not in j:
            raise ValueError("Missing judge unit")
        lab = j[k["prediction_id"]]["majority"]
        if lab is not None:
            a.append(lab)
            b.append(r["label"])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot if a else 0):
        ix = rng.integers(0, len(a), len(a))
        value = cohens_kappa([a[i] for i in ix], [b[i] for i in ix])
        if np.isfinite(value):
            boots.append(value)
    matrix = {x: {y: 0 for y in LABELS} for x in LABELS}
    for x, y in zip(a, b):
        matrix[x][y] += 1
    return {
        "human_annotations_complete": True,
        "judge_validity_established": False,
        "n": len(rows),
        "n_comparable": len(a),
        "n_judge_abstained": len(rows) - len(a),
        "agreement": float(np.mean(np.array(a) == np.array(b))) if a else None,
        "kappa": cohens_kappa(a, b) if a else None,
        "kappa_ci95": np.percentile(boots, [2.5, 97.5]).tolist() if boots else None,
        "bootstrap_finite": len(boots),
        "confusion_judge_rows_human_columns": matrix,
        "attestation": attestation,
        "limitations": [
            "Diagnostic small sample, not prevalence representative",
            "Expertise attested, not independently verified",
            "Human agreement does not establish legal correctness",
        ],
    }
