"""Tests for routing.py, including the Decision 1 core result."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing import (
    selective_risk_at_coverage, risk_coverage_curve, threshold_for_coverage,
    apply_threshold, threshold_for_target_risk, compare_pooled_routing,
    bootstrap_routing_gap, single_model_invariance_demo,
)


def test_selective_risk_basic():
    # scores high->low; the two least-confident are wrong
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [1, 1, 1, 0, 0]
    risk, k = selective_risk_at_coverage(scores, correct, 0.6)   # top 3
    assert k == 3 and risk == 0.0
    risk, k = selective_risk_at_coverage(scores, correct, 1.0)   # all
    assert k == 5 and abs(risk - 0.4) < 1e-9


def test_review_workload_is_complement():
    curve = risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0], [0.5, 0.75])
    for row in curve:
        assert abs(row["review_workload"] - (1 - row["coverage"])) < 1e-9


def test_threshold_roundtrip():
    scores = np.linspace(0.1, 0.99, 100)
    thr = threshold_for_coverage(scores, 0.70)
    res = apply_threshold(scores, np.ones(100), thr)
    assert abs(res["coverage"] - 0.70) < 0.02


def test_single_model_monotonic_invariance():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 200)
    correct = (rng.uniform(0, 1, 200) < scores).astype(float)
    # strictly monotonic recalibration (Platt-like sigmoid) preserves ordering
    calibrated = 1 / (1 + np.exp(-(3 * scores - 1.5)))
    demo = single_model_invariance_demo(scores, correct, calibrated, coverage=0.70)
    assert demo["identical"] is True
    assert demo["selective_risk_raw"] == demo["selective_risk_calibrated"]


def test_pooled_calibration_lowers_risk():
    # Model A: raw 0.80, 80% correct, well-calibrated (cal 0.80)
    # Model B: raw 0.95 but only 40% correct, calibrated DOWN to 0.40
    items = []
    items += [{"raw": 0.80, "cal": 0.80, "correct": 1} for _ in range(8)]
    items += [{"raw": 0.80, "cal": 0.80, "correct": 0} for _ in range(2)]
    items += [{"raw": 0.95, "cal": 0.40, "correct": 1} for _ in range(4)]
    items += [{"raw": 0.95, "cal": 0.40, "correct": 0} for _ in range(6)]
    res = compare_pooled_routing(items, coverages=[0.5], primary_coverage=0.5)
    # raw ranks B (0.95) first -> top-10 are all B -> risk 0.6
    assert abs(res["selective_risk_raw"] - 0.6) < 1e-9
    # calibrated ranks A (0.80) first -> top-10 are all A -> risk 0.2
    assert abs(res["selective_risk_calibrated"] - 0.2) < 1e-9
    assert res["improvement"] > 0        # calibration helps in the pooled setting


def test_bootstrap_routing_gap_positive():
    items = []
    items += [{"raw": 0.80, "cal": 0.80, "correct": 1} for _ in range(8)]
    items += [{"raw": 0.80, "cal": 0.80, "correct": 0} for _ in range(2)]
    items += [{"raw": 0.95, "cal": 0.40, "correct": 1} for _ in range(4)]
    items += [{"raw": 0.95, "cal": 0.40, "correct": 0} for _ in range(6)]
    point, lo, hi = bootstrap_routing_gap(items, coverage=0.5, n_boot=400, seed=1)
    assert point > 0 and lo <= point <= hi


def test_target_risk_threshold():
    # top cases correct, tail wrong; target risk 0.0 should stop before the wrong ones
    scores = [0.9, 0.85, 0.8, 0.4, 0.3]
    correct = [1, 1, 1, 0, 0]
    res = threshold_for_target_risk(scores, correct, target_risk=0.0)
    assert res["coverage"] == 0.6          # exactly the three correct, high-confidence cases


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
