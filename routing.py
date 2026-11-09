"""Retrospective matched coverage is NOT a deployment threshold policy.
Pooled unit is model-case recommendation; bootstrap clusters shared products.
Within-model increasing calibration preserves order except ties.
"""

import math, hashlib
import numpy as np


def _scores(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError("Finite probability vector required")
    return x


def _xy(c, y):
    x = _scores(c)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(x) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned binary labels required")
    return x, y


def _order(x, ids=None):
    ids = list(ids) if ids is not None else list(range(len(x)))
    if len(ids) != len(x):
        raise ValueError("IDs must align")
    keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in ids]
    return sorted(range(len(x)), key=lambda i: (-x[i], keys[i], i))


def routing_at_coverage(confidence, correct, coverage, ids=None):
    x, y = _xy(confidence, correct)
    if not 0 < coverage <= 1:
        raise ValueError("Coverage must be in (0,1]")
    if not len(x):
        return {
            "coverage": 0.0,
            "requested_coverage": coverage,
            "threshold": None,
            "n_streamlined": 0,
            "selective_risk": None,
            "accuracy_streamlined": None,
            "n_escalated": 0,
        }
    k = max(1, int(math.floor(len(x) * coverage + 0.5)))
    ix = _order(x, ids)[:k]
    acc = float(y[ix].mean())
    return {
        "requested_coverage": coverage,
        "coverage": k / len(x),
        "threshold": float(min(x[ix])),
        "n_streamlined": k,
        "selective_risk": 1 - acc,
        "accuracy_streamlined": acc,
        "n_escalated": len(x) - k,
        "review_workload": 1 - k / len(x),
    }


def risk_coverage_curve(confidence, correct, coverages=None, ids=None):
    return [
        routing_at_coverage(confidence, correct, c, ids)
        for c in (
            coverages if coverages is not None else [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
    ]


def threshold_for_target_risk(conf_val, y_val, target_risk=0.10):
    """Largest WHOLE-score-group empirical validation prefix meeting target.
    No finite-sample risk guarantee. Increasing calibration alone usually leaves
    the selected prefix unchanged. Never choose this threshold on test labels.
    """
    x, y = _xy(conf_val, y_val)
    if not 0 <= target_risk <= 1:
        raise ValueError("Invalid target risk")
    best = None
    for t in sorted(set(x), reverse=True):
        accept = x >= t
        if 1 - float(y[accept].mean()) <= target_risk:
            best = float(t)
    return {"threshold": best, **apply_threshold(x, y, best)}


def apply_threshold(confidence, correct, threshold):
    x, y = _xy(confidence, correct)
    mask = np.zeros(len(x), dtype=bool) if threshold is None else x >= threshold
    n = int(mask.sum())
    acc = float(y[mask].mean()) if n else None
    return {
        "coverage": n / len(x) if len(x) else 0.0,
        "n_streamlined": n,
        "selective_risk": None if acc is None else 1 - acc,
        "accuracy_streamlined": acc,
        "n_escalated": len(x) - n,
    }


def compare_pooled_routing(items, coverage=0.70, coverages=None, primary_coverage=None):
    if primary_coverage is not None:
        coverage = primary_coverage
    if not items:
        raise ValueError("Nonempty recommendation pool required")
    ids = [r.get("prediction_id", i) for i, r in enumerate(items)]
    y = [r["correct"] for r in items]
    a = routing_at_coverage([r["raw"] for r in items], y, coverage, ids)
    b = routing_at_coverage([r["cal"] for r in items], y, coverage, ids)
    return {
        "primary_coverage": coverage,
        "n_pooled": len(items),
        "unit": "model_case_prediction",
        "selective_risk_raw": a["selective_risk"],
        "selective_risk_calibrated": b["selective_risk"],
        "improvement": a["selective_risk"] - b["selective_risk"],
        "raw": a,
        "calibrated": b,
    }


def bootstrap_routing_gap(items, coverage=0.70, n_boot=2000, alpha=0.05, seed=7):
    if not items or n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap inputs")
    groups = {}
    for i, r in enumerate(items):
        groups.setdefault(r.get("case_id", f"legacy-row:{i}"), []).append(r)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        batch = []
        for draw, j in enumerate(rng.integers(0, len(keys), len(keys))):
            batch.extend(
                {**r, "prediction_id": f'{draw}|{r.get("prediction_id",k)}'}
                for k, r in enumerate(groups[keys[j]])
            )
        boots.append(compare_pooled_routing(batch, coverage)["improvement"])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return compare_pooled_routing(items, coverage)["improvement"], float(lo), float(hi)


def selective_risk_at_coverage(confidence, correct, coverage):
    r = routing_at_coverage(confidence, correct, coverage)
    return r["selective_risk"], r["n_streamlined"]


def threshold_for_coverage(confidence, coverage):
    x = _scores(confidence)
    if not len(x) or not 0 < coverage <= 1:
        raise ValueError("Nonempty scores and valid coverage required")
    k = max(1, int(math.floor(len(x) * coverage + 0.5)))
    return float(sorted(x, reverse=True)[k - 1])


def single_model_invariance_demo(scores, correct, calibrated, coverage=0.70):
    a = routing_at_coverage(scores, correct, coverage)
    b = routing_at_coverage(calibrated, correct, coverage)
    return {
        "selective_risk_raw": a["selective_risk"],
        "selective_risk_calibrated": b["selective_risk"],
        "identical": a["selective_risk"] == b["selective_risk"],
        "same_order": _order(_scores(scores)) == _order(_scores(calibrated)),
    }
