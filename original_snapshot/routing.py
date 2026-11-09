"""
routing.py
----------
Stage 7 + Decision 1 (second half of RQ2). Human-review routing as a policy
simulation: cases above a confidence threshold get a streamlined review; the
rest are escalated.

Three quantities (Stage 7):
  * coverage        -- fraction streamlined (not escalated);
  * selective risk  -- error rate among streamlined cases;
  * review workload -- fraction escalated (1 - coverage).

Decision 1. For a SINGLE model, Platt/isotonic calibration is monotonic, so at
equal coverage raw and calibrated confidence select the same cases and give the
same selective risk. `single_model_invariance_demo` makes that explicit. The
real comparison is therefore POOLED two-model routing: calibrating each model
separately puts both on a common scale, so the combined ranking -- and the
selective risk at matched coverage -- genuinely changes. Target-risk
thresholding is the second argument: calibrated confidence approximates a
probability, so a validation-chosen threshold reliably controls test risk.

Thresholds are chosen on validation and evaluated on test, never the reverse.
"""
from __future__ import annotations
import numpy as np

PRIMARY_COVERAGE = 0.70
COVERAGE_POINTS = (0.50, 0.60, 0.70, 0.80, 0.90)


def _as_arrays(scores, correct):
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(correct, dtype=float).ravel()
    if s.shape != y.shape:
        raise ValueError("scores and correctness must be the same length")
    return s, y


# --------------------------------------------------------------------------- #
# Coverage-matched evaluation on a single ranked set                          #
# --------------------------------------------------------------------------- #
def selective_risk_at_coverage(scores, correct, coverage):
    """Error rate among the top-`coverage` fraction ranked by score (desc).
    Returns (selective_risk, k). Ties broken stably by original order."""
    s, y = _as_arrays(scores, correct)
    n = len(s)
    k = int(round(coverage * n))
    if k <= 0:
        return float("nan"), 0
    order = np.argsort(-s, kind="stable")
    top = y[order][:k]
    return float(1.0 - top.mean()), k


def risk_coverage_curve(scores, correct, coverages=COVERAGE_POINTS):
    out = []
    for c in coverages:
        risk, k = selective_risk_at_coverage(scores, correct, c)
        out.append({"coverage": float(c), "k": k,
                    "selective_risk": risk, "review_workload": float(1.0 - c)})
    return out


# --------------------------------------------------------------------------- #
# Validation-chosen thresholds, evaluated on test                             #
# --------------------------------------------------------------------------- #
def threshold_for_coverage(scores, coverage):
    """Score threshold on a (validation) set that streamlines `coverage` of it."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(s)
    k = int(round(coverage * n))
    if k <= 0:
        return float("inf")
    if k >= n:
        return float("-inf")
    return float(s[k - 1])


def apply_threshold(scores, correct, threshold):
    """Evaluate a fixed threshold on a (test) set: realized coverage and risk."""
    s, y = _as_arrays(scores, correct)
    mask = s >= threshold
    cov = float(mask.mean())
    risk = float(1.0 - y[mask].mean()) if mask.any() else float("nan")
    return {"threshold": float(threshold), "coverage": cov,
            "selective_risk": risk, "review_workload": float(1.0 - cov)}


def threshold_for_target_risk(scores, correct, target_risk):
    """Largest-coverage threshold whose top-prefix selective risk <= target_risk
    on a (validation) set. With calibrated scores this threshold is ~ a
    probability, so realized test risk tracks the target; with raw scores it
    does not."""
    s, y = _as_arrays(scores, correct)
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order]
    cum = np.cumsum(ys)
    ks = np.arange(1, len(ss) + 1)
    prefix_risk = 1.0 - cum / ks
    ok = np.where(prefix_risk <= target_risk)[0]
    if len(ok) == 0:
        return {"threshold": float("inf"), "coverage": 0.0, "target_risk": target_risk}
    k = int(ok.max()) + 1
    return {"threshold": float(ss[k - 1]), "coverage": float(k / len(ss)),
            "target_risk": target_risk}


# --------------------------------------------------------------------------- #
# Pooled two-model routing (the Decision 1 experiment)                        #
# --------------------------------------------------------------------------- #
def assemble_pool(items):
    """items: list of {'raw': float, 'cal': float, 'correct': 0/1}. Returns arrays."""
    raw = np.array([it["raw"] for it in items], dtype=float)
    cal = np.array([it["cal"] for it in items], dtype=float)
    cor = np.array([it["correct"] for it in items], dtype=float)
    return raw, cal, cor


def compare_pooled_routing(items, coverages=COVERAGE_POINTS, primary_coverage=PRIMARY_COVERAGE):
    """Pool both models' outputs; compare routing by RAW vs CALIBRATED score.
    Positive `improvement` = calibrated gives lower selective risk."""
    raw, cal, cor = assemble_pool(items)
    r_raw, _ = selective_risk_at_coverage(raw, cor, primary_coverage)
    r_cal, _ = selective_risk_at_coverage(cal, cor, primary_coverage)
    return {
        "primary_coverage": primary_coverage,
        "selective_risk_raw": r_raw,
        "selective_risk_calibrated": r_cal,
        "improvement": r_raw - r_cal,
        "curve_raw": risk_coverage_curve(raw, cor, coverages),
        "curve_calibrated": risk_coverage_curve(cal, cor, coverages),
    }


def bootstrap_routing_gap(items, coverage=PRIMARY_COVERAGE, n_boot=2000, alpha=0.05, seed=7):
    """(point, lo, hi) for (raw risk - calibrated risk) at a coverage, by case
    resampling. A CI above 0 supports H2 (calibration lowers selective risk)."""
    raw, cal, cor = assemble_pool(items)
    n = len(cor)
    rng = np.random.default_rng(seed)

    def gap(idx):
        r = selective_risk_at_coverage(raw[idx], cor[idx], coverage)[0]
        c = selective_risk_at_coverage(cal[idx], cor[idx], coverage)[0]
        return r - c

    point = gap(np.arange(n))
    boots = np.array([gap(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Honest single-model demonstration (addresses the marker directly)           #
# --------------------------------------------------------------------------- #
def single_model_invariance_demo(scores, correct, calibrated_scores, coverage=PRIMARY_COVERAGE):
    """Show that a strictly monotonic recalibration of ONE model's scores leaves
    selective risk unchanged at matched coverage. Returns both risks; they are
    equal whenever the recalibration preserves the ordering (e.g. Platt)."""
    r_raw, _ = selective_risk_at_coverage(scores, correct, coverage)
    r_cal, _ = selective_risk_at_coverage(calibrated_scores, correct, coverage)
    return {"coverage": coverage, "selective_risk_raw": r_raw,
            "selective_risk_calibrated": r_cal, "identical": abs(r_raw - r_cal) < 1e-12}
