"""
judge.py
--------
Stage 5 (RQ3). Tariff-grounded evidence-support evaluator.

Given a product description, a predicted code, the model's rationale and an
evidence pack (which EXCLUDES the case's own gold ruling -- handled in data prep
so the judge cannot cheat), an LLM judge labels each case:

    supported | uncertain | unsupported     (rubric, Table 6)

Design-lock requirements implemented here:
  * repeated runs with RANDOMISED evidence order -> consistency + positional-bias check;
  * a majority label per case with a consistency score;
  * Cohen's kappa against the mandatory independent human validation (Decision 5).

Reuses the client + JSON-extraction machinery from generation.py.
"""
from __future__ import annotations
import os, sys, random
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from generation import _extract_json, MalformedOutput, MockClient, LLMClient  # noqa: E402,F401

LABELS = ("supported", "uncertain", "unsupported")

JUDGE_SYSTEM_PROMPT = (
    "You are an independent evaluator of U.S. HTS tariff classifications. You do "
    "NOT re-classify the product. You judge only whether the supplied evidence "
    "and rationale SUPPORT the predicted 10-digit code.\n\n"
    "Return ONLY this JSON object:\n"
    '{"label": "supported|uncertain|unsupported", "reason": "<one sentence>"}\n\n'
    "Rubric:\n"
    "- supported:   the evidence identifies product facts and tariff/ruling logic "
    "consistent with the predicted code, with no material contradiction.\n"
    "- uncertain:   the evidence is incomplete or insufficient to decide, but does "
    "not contradict the predicted code.\n"
    "- unsupported: the evidence contradicts the code, applies to a materially "
    "different product, or does not justify the decisive hierarchy distinction.\n"
    "Output JSON only."
)


def build_judge_messages(case: dict, evidence: list[dict]) -> tuple[str, str]:
    ev = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence) or "(no evidence provided)"
    user = (
        f"Product description:\n{case['description']}\n\n"
        f"Predicted HTS code: {case['predicted_code']}\n"
        f"Model rationale: {case.get('rationale', '(none)')}\n\n"
        f"Evidence:\n{ev}\n\n"
        "Judge whether the evidence supports the predicted code. JSON only."
    )
    return JUDGE_SYSTEM_PROMPT, user


def parse_judge_response(text: str) -> dict:
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        raise MalformedOutput("judge output is not a JSON object")
    label = str(obj.get("label", "")).strip().lower()
    if label not in LABELS:
        raise MalformedOutput(f"label not in {LABELS}: {obj.get('label')!r}")
    reason = obj.get("reason", "")
    return {"label": label, "reason": str(reason).strip()}


def judge_one(client, case, decoding, n_runs=3, shuffle_evidence=True, seed=7):
    """Run the judge n_runs times with randomised evidence order. Returns the
    per-run labels, the majority label, and a consistency score (fraction of
    valid runs agreeing with the majority)."""
    rng = random.Random(seed)
    labels, raws = [], []
    for _ in range(n_runs):
        ev = list(case.get("evidence", []))
        if shuffle_evidence:
            rng.shuffle(ev)
        system, user = build_judge_messages(case, ev)
        raw = client.complete(system, user, decoding)
        raws.append(raw)
        try:
            labels.append(parse_judge_response(raw)["label"])
        except MalformedOutput:
            labels.append(None)
    valid = [l for l in labels if l is not None]
    majority = Counter(valid).most_common(1)[0][0] if valid else None
    consistency = (sum(1 for l in valid if l == majority) / len(valid)) if valid else 0.0
    return {"case_id": case.get("id"), "labels": labels, "majority": majority,
            "consistency": float(consistency), "n_valid": len(valid)}


def run_judge(client, cases, decoding, n_runs=3, shuffle_evidence=True, seed=7):
    return [judge_one(client, c, decoding, n_runs, shuffle_evidence, seed + i)
            for i, c in enumerate(cases)]


def cohens_kappa(a, b, labels=LABELS) -> float:
    """Cohen's kappa between two categorical label sequences (judge vs human)."""
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be equal length and non-empty")
    idx = {l: i for i, l in enumerate(labels)}
    k, n = len(labels), len(a)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float(((m.sum(1) / n) * (m.sum(0) / n)).sum())
    return 1.0 if pe == 1.0 else float((po - pe) / (1.0 - pe))


def aggregate_judge(records: list[dict]) -> dict:
    labs = [r["majority"] for r in records if r["majority"]]
    dist = {l: labs.count(l) / len(labs) for l in LABELS} if labs else {l: 0.0 for l in LABELS}
    cons = [r["consistency"] for r in records if r["n_valid"] > 0]
    return {"n": len(records), "label_distribution": dist,
            "mean_consistency": float(np.mean(cons)) if cons else 0.0}
