"""
rq3_judge.py
------------
Post-hoc RQ3 (tariff-grounded evidence-support evaluator) on a COMPLETED study
root that was generated with --evidence-mode none. Lives in analysis/ so it is
outside the code freeze; it reads the sealed scored outputs and writes only to
<root>/rq3/.

Why post hoc: the ATLAS mirror carries no source_ruling_id, so the frozen
pipeline's retrieved-evidence stage (and prepare_judge) cannot run. The evidence
here is EVALUATOR-ONLY: the generators never saw it.

Evidence pack per sampled prediction (proposal Sec. 7.2/7.3):
  * the official HTSUS provision text for the PREDICTED code (USITC export,
    hierarchy path chapter heading > ... > statistical line; longest existing
    prefix if the 10-digit line is not in the current schedule);
  * up to k = 3 CROSS-ruling passages retrieved by TF-IDF on the product
    description from the orlandowhite/hts-cross-rulings dump.
Gold-ruling exclusion: the case's own ruling cannot be identified reliably, so
EVERY ruling carrying the case's gold 10-digit code is excluded from its pool
(leak-proof by construction, conservative), as is any ruling whose text contains
the held-out description or gold reasoning.

Sampling, judging and the blinded human export reuse the pipeline's own
stratified_judge_subset, judge_one and annotations.export.

    python analysis/rq3_judge.py prepare --root results/main_v1 \
        --hts data/hts_schedule/hts_full.json --rulings data/cross_dump/cross_rulings.json
    python analysis/rq3_judge.py judge   --root results/main_v1 --model google/gemini-3.8-flash [--limit 4]
    python analysis/rq3_judge.py analyse --root results/main_v1
    python analysis/rq3_judge.py export-human --root results/main_v1

The judge call expects OPENAI_API_KEY / OPENAI_BASE_URL to point at OpenRouter.
"""

from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import stratified_judge_subset  # noqa: E402
from evidence import normalized  # noqa: E402
from judge import judge_one, LABELS  # noqa: E402
from study_io import digest, file_hash, lines, read, write, write_lines  # noqa: E402

K = 3
DECODING = {"max_completion_tokens": 8000}
HTS_RELEASE = "2026HTSRev22"
SCORE = {"supported": 1.0, "uncertain": 0.5, "unsupported": 0.0}


def digits(s):
    return re.sub(r"\D", "", s or "")


# --------------------------------------------------------------------------- #
# Evidence sources                                                            #
# --------------------------------------------------------------------------- #
def hts_paths(path):
    """Map every numbered HTS line (4-10 digits) to its hierarchy path text."""
    rows = json.load(open(path, encoding="utf-8"))
    stack, out = [], {}
    for r in rows:
        indent = int(r.get("indent") or 0)
        no = r.get("htsno") or ""
        desc = re.sub(r"<[^>]+>", "", r.get("description") or "").strip()
        if len(digits(no)) == 4:
            stack = []
        stack = [s for s in stack if s[0] < indent] + [(indent, no, desc)]
        if no:
            out[digits(no)] = " > ".join(f"{n} {d}".strip() for _, n, d in stack)
    return out


def provision(code, paths):
    for n in (10, 8, 6, 4):
        if code[:n] in paths:
            note = "" if n == 10 else f" (10-digit line {code} not in the current schedule; nearest {n}-digit provision shown)"
            return {
                "id": f"HTSUS:{code}",
                "text": f"HTSUS {HTS_RELEASE} provision{note}: {paths[code[:n]]}",
                "source_url": f"https://hts.usitc.gov/search?query={code[:4]}",
                "kind": "tariff_provision",
            }
    return None


def ruling_corpus(path):
    rows, seen = [], set()
    for r in json.load(open(path, encoding="utf-8")):
        rid = (r.get("ruling_id") or "").strip()
        if not rid or rid in seen or not r.get("full_description"):
            continue
        seen.add(rid)
        rows.append({
            "id": rid,
            "code": digits(r.get("hts_code")),
            "text": f"CBP ruling {rid} ({r.get('date')}), classified under HTSUS {r.get('hts_code')}. "
                    f"Product: {r['full_description']} Reasoning: {r.get('reasoning') or ''}".strip(),
            "source_url": f"https://rulings.cbp.gov/ruling/{rid.split()[-1]}",
            "kind": "cross_ruling",
        })
    return rows


# --------------------------------------------------------------------------- #
# Stages                                                                      #
# --------------------------------------------------------------------------- #
def prepare(root, hts_path, rulings_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    out = root / "rq3"
    if (out / "judge_cases.jsonl").exists():
        raise ValueError("RQ3 sample already frozen")
    protocol = read(root / "protocol.json")
    rows = []
    for m in ("A", "B"):
        for r in lines(root / "test" / f"scored_{m}.jsonl"):
            if r["status"] == "ok":
                rows.append({**r, "model": m})
    chosen = stratified_judge_subset(rows, protocol["judge_per_stratum"], seed=protocol["seed"])

    paths = hts_paths(hts_path)
    corpus = ruling_corpus(rulings_path)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    matrix = vec.fit_transform([c["text"] for c in corpus])
    norm_text = [normalized(c["text"]) for c in corpus]

    cases, audit = [], []
    for r in chosen:
        blocked_code = r["gold"]
        held = [normalized(r["description"])] + ([normalized(r["reasoning"])] if r.get("reasoning") else [])
        sims = cosine_similarity(vec.transform([r["description"]]), matrix)[0]
        picked, n_code_blocked = [], 0
        for i in np.argsort(-sims, kind="stable"):
            if sims[i] <= 0 or len(picked) == K:
                break
            if corpus[i]["code"] == blocked_code:
                n_code_blocked += 1
                continue
            if any(h and h in norm_text[i] for h in held):
                continue
            picked.append({k: corpus[i][k] for k in ("id", "text", "source_url", "kind")} | {"retrieval_score": round(float(sims[i]), 4)})
        prov = provision(digits(r["pred"]), paths)
        evidence = ([prov] if prov else []) + picked
        cases.append({
            "case_id": r["case_id"],
            "prediction_id": r["prediction_id"],
            "description": r["description"],
            "predicted_code": r["pred"],
            "rationale": r["rationale"],
            "evidence": evidence,
            "generator_model": protocol["models"][r["model"]]["model"],
            "generator_label": r["model"],
            "stratum": r["stratum"],
        })
        audit.append({
            "prediction_id": r["prediction_id"], "stratum": r["stratum"],
            "gold": r["gold"], "pred": r["pred"], "correct": bool(r["correct"]),
            "severity_tier": r["severity_tier"], "confidence": r["confidence"],
            "provision_found": prov is not None,
            "provision_exact_10": bool(prov) and "not in the current schedule" not in prov["text"],
            "same_gold_code_rulings_skipped": n_code_blocked,
            "rulings_in_pack": len(picked),
        })
    out.mkdir(parents=True, exist_ok=True)
    write_lines(out / "judge_cases.jsonl", cases)
    write_lines(out / "judge_case_truth.jsonl", audit)  # gold side, never shown to the judge
    write(out / "rq3_manifest.json", {
        "source_root": str(root),
        "scored_test_hashes": {m: file_hash(root / "test" / f"scored_{m}.jsonl") for m in ("A", "B")},
        "hts_export": {"file": file_hash(hts_path), "release": HTS_RELEASE},
        "rulings_dump": {"file": file_hash(rulings_path), "dataset": "orlandowhite/hts-cross-rulings",
                         "revision": "050ee2144e757db476a6c81cc54e160afd5586d5", "n_rulings": len(corpus)},
        "k": K, "exclusion": "all rulings with the case's gold 10-digit code + held-out text containment",
        "evidence_visible_to_generators": False,
        "sample_hash": digest(cases),
        "n_cases": len(cases),
        "strata": Counter(c["stratum"] for c in cases),
    })
    return {"n": len(cases), "strata": Counter(c["stratum"] for c in cases)}


def judge(root, model, limit=None, runs=3, seed=7):
    from generation import OpenAIClient

    out = root / "rq3"
    cases = lines(out / "judge_cases.jsonl")
    if digest(cases) != read(out / "rq3_manifest.json")["sample_hash"]:
        raise ValueError("RQ3 judge sample changed after freeze")
    done = lines(out / "judge_results.jsonl") if (out / "judge_results.jsonl").exists() else []
    if len({r["prediction_id"] for r in done}) != len(done):
        raise ValueError("Duplicate judge prediction IDs")
    if any(r.get("model") != model or len(r.get("labels", [])) != runs for r in done):
        raise ValueError("Judge configuration differs from cached responses")
    ids = {r["prediction_id"] for r in done}
    if ids - {r["prediction_id"] for r in cases}:
        raise ValueError("Foreign judge prediction IDs")
    from study_io import bind
    run_config = out / "judge_run_config.json"
    if done and not run_config.exists():
        raise ValueError("Legacy judge output has no frozen run configuration; analyse it, but start a new run for new calls")
    bind(run_config, {"model": model, "runs": runs, "seed": seed, "decoding": DECODING,
                      "sample_hash": digest(cases), "judge_source_hash": file_hash(Path(__file__).parents[1] / "judge.py")})
    client = OpenAIClient(model, os.environ["OPENAI_API_KEY"])
    n_new = 0
    for i, c in enumerate(cases):
        if c["prediction_id"] in ids:
            continue
        if limit is not None and n_new >= limit:
            break
        res = judge_one(client, c, DECODING, runs, seed=seed + i)
        done.append({**res, "stratum": c["stratum"], "generator_label": c["generator_label"],
                     "judge_is_generator": model == c["generator_model"]})
        write_lines(out / "judge_results.jsonl", done)
        n_new += 1
        print(f"{len(done)}/{len(cases)} {c['prediction_id']} {res['labels']}", flush=True)
    return {"n": len(done), "abstained": sum(r["majority"] is None for r in done)}


def auc(pos, neg):
    """P(score of a random positive > random negative), ties = 1/2."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def verify_inputs(root):
    """The scored files the sample was drawn from must still match their seals
    (pipeline.Study re-checks protocol, code freeze, generation and calibration)
    and the hashes recorded when the RQ3 sample was frozen."""
    from pipeline import Study

    Study(root).scored("test")
    recorded = read(root / "rq3" / "rq3_manifest.json")["scored_test_hashes"]
    if any(file_hash(root / "test" / f"scored_{m}.jsonl") != h for m, h in recorded.items()):
        raise ValueError("Scored test files changed since the RQ3 sample was frozen")


def validated_rows(root):
    verify_inputs(root)
    out = root / "rq3"
    cases = lines(out / "judge_cases.jsonl")
    if digest(cases) != read(out / "rq3_manifest.json")["sample_hash"]:
        raise ValueError("Judge sample changed")
    results = lines(out / "judge_results.jsonl")
    expected = {r["prediction_id"] for r in cases}
    if len(results) != len(expected) or {r["prediction_id"] for r in results} != expected:
        raise ValueError("Final analysis requires complete unique judge results")
    if len({r["model"] for r in results}) != 1:
        raise ValueError("Mixed evaluator models")
    res = {r["prediction_id"]: r for r in results}
    truth = lines(out / "judge_case_truth.jsonl")
    if len(truth) != len(expected) or {t["prediction_id"] for t in truth} != expected:
        raise ValueError("Final analysis requires complete unique truth records")
    source = {r["prediction_id"]: r for m in ("A", "B")
              for r in lines(root / "test" / f"scored_{m}.jsonl")}
    for t in truth:
        if any(t[k] != source[t["prediction_id"]][k]
               for k in ("gold", "pred", "correct", "severity_tier")):
            raise ValueError("Judge truth differs from sealed scored predictions")
    from judge import parse_judge_response, build_judge_messages
    case_map = {c["prediction_id"]: c for c in cases}
    run_counts = {len(r.get("labels", [])) for r in results}
    if len(run_counts) != 1 or not next(iter(run_counts)):
        raise ValueError("Missing or mixed judge repetition counts")
    for t in truth:
        r = res[t["prediction_id"]]
        labels = r["labels"]
        history = r.get("history", [])
        if len(history) != len(labels):
            raise ValueError("Judge history/labels mismatch")
        case = case_map[t["prediction_id"]]
        if case["predicted_code"] != t["pred"]:
            raise ValueError("Judge case and truth disagree")
        ev = {e["id"]: e for e in case["evidence"]}
        for i, (label, record) in enumerate(zip(labels, history), 1):
            order = record.get("evidence_order", [])
            if len(order) != len(ev) or set(order) != set(ev):
                raise ValueError("Evidence order is not a permutation")
            sy, us = build_judge_messages(case, [ev[e] for e in order])
            if record.get("prompt_hash") != digest([sy, us]) or record.get("run") != i:
                raise ValueError("Judge prompt/history mismatch")
            if label is None:
                if record.get("status") != "failed":
                    raise ValueError("Failed judge label/history mismatch")
            elif (record.get("status") != "ok" or
                  parse_judge_response(record["raw"])["label"] != label or
                  record.get("label") != label):
                raise ValueError("Judge label differs from raw response")
        valid = [SCORE[x] for x in labels if x is not None]
        counts = Counter(x for x in labels if x is not None)
        majority = next((x for x, n in counts.items() if n > len(labels) / 2), None)
        consistency = counts.get(majority, 0) / len(labels) if majority else 0.0
        checks = {"majority": majority, "consistency": consistency,
                  "n_valid": len(valid), "n_runs": len(labels), "abstained": majority is None}
        if any(r.get(k) != v for k, v in checks.items()):
            raise ValueError("Cached judge aggregates disagree with individual runs")
        t.update(majority=majority, consistency=consistency,
                 unanimous=len(set(labels)) == 1 and None not in labels,
                 support=float(np.mean(valid)) if valid else None)
    return truth, next(iter(res.values()))["model"]


def clustered_auc(rows, n_boot=2000, seed=7):
    scored = [r for r in rows if r["support"] is not None]
    if not scored:
        return {"auc": None, "ci95": None, "n_products": 0}
    groups = defaultdict(list)
    for r in scored:
        groups[r["prediction_id"].rsplit("|", 1)[0]].append(r)
    keys = sorted(groups)
    def calc(rs):
        return auc([r["support"] for r in rs if r["correct"]],
                   [r["support"] for r in rs if not r["correct"]])
    rng = np.random.default_rng(seed)
    boots = [calc([r for j in rng.integers(0, len(keys), len(keys)) for r in groups[keys[j]]])
             for _ in range(n_boot)]
    boots = [b for b in boots if b is not None]
    return {"auc": calc(scored), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])] if boots else None,
            "n_products": len(keys)}


def analyse(root, n_boot=2000, seed=7, output=None):
    truth, judge_model = validated_rows(root)
    out = Path(output) if output is not None else root / "rq3"
    dist = lambda rows: {l: sum(t["majority"] == l for t in rows) for l in (*LABELS, None)}
    # Same definitions as metrics.py / posthoc.py: major error = wrong chapter.
    # (The sampler's "major_error" STRATUM is wider: wrong subheading or worse.)
    groups = {"major_error_wrong_chapter": ("wrong_chapter",),
              "heading_or_worse": ("wrong_heading", "wrong_chapter")}

    def rate(label, tiers):
        rows = [t for t in truth if t["severity_tier"] in tiers]
        return {"n": len(rows), "rate": float(np.mean([t["majority"] == label for t in rows])) if rows else None}

    clustered = clustered_auc(truth, n_boot, seed)
    summary = {
        "n": len(truth),
        "judge_model": judge_model,
        "majority_labels": dist(truth),
        "by_stratum": {s: dist([t for t in truth if t["stratum"] == s]) for s in sorted({t["stratum"] for t in truth})},
        "mean_consistency": float(np.mean([t["consistency"] for t in truth])),
        "unanimous_rate": float(np.mean([t["unanimous"] for t in truth])),
        "abstentions": sum(t["majority"] is None for t in truth),
        "validity_vs_gold": {
            "auc_support_correct_vs_wrong": clustered["auc"],
            "auc_ci95": clustered["ci95"],
            "bootstrap_unit": "product", "n_products": clustered["n_products"],
            "p_supported_given_correct": float(np.mean([t["majority"] == "supported" for t in truth if t["correct"]])) if any(t["correct"] for t in truth) else None,
            **{f"p_{label}_given_{g}": rate(label, tiers) for g, tiers in groups.items() for label in ("unsupported", "supported")},
        },
        "provision_exact_10_rate": float(np.mean([t["provision_exact_10"] for t in truth])),
    }
    write(out / "analysis_inputs.json", {
        "scope": "Input snapshot at reanalysis, not a historical response seal",
        "files": {f: file_hash(root / "rq3" / f) for f in
                  ("rq3_manifest.json", "judge_cases.jsonl", "judge_case_truth.jsonl", "judge_results.jsonl")},
        "analysis_sha256": file_hash(Path(__file__)), "n_boot": n_boot, "seed": seed})
    write(out / "rq3_summary.json", summary)
    write_lines(out / "judge_case_scored.jsonl", truth)
    return summary


def export_human(root, size=30, seed=7):
    from annotations import export

    return export(lines(root / "rq3" / "judge_cases.jsonl"), root / "rq3" / "annotations", size, seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "judge", "analyse", "export-human"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--hts")
    ap.add_argument("--rulings")
    ap.add_argument("--model")
    ap.add_argument("--out", help="Separate directory for reanalysis outputs")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    root = Path(a.root)
    if a.stage == "prepare":
        r = prepare(root, a.hts, a.rulings)
    elif a.stage == "judge":
        r = judge(root, a.model, a.limit)
    elif a.stage == "analyse":
        r = analyse(root, output=a.out)
    else:
        r = export_human(root)
    print(json.dumps(r, indent=2, default=str))
