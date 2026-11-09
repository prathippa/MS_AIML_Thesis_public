"""Tests for calibration.py: metrics behave correctly and calibration helps."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from calibration import (
    ece,
    brier,
    reliability_curve,
    bootstrap_metric_ci,
    PlattCalibrator,
    IsotonicCalibrator,
    fit_calibrator,
    calibrate_and_eval,
)


def _overconfident(n=4000, seed=0):
    """Model reports confidence higher than its true accuracy -> miscalibrated."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.3, 0.95, n)
    reported = np.clip(true_p + 0.15, 0, 1)  # systematically overconfident
    correct = (rng.uniform(0, 1, n) < true_p).astype(float)
    return reported, correct


def test_ece_zero_when_perfectly_calibrated():
    # confidence exactly equals bin accuracy -> ECE ~ 0
    conf = np.concatenate([np.full(500, 0.25), np.full(500, 0.75)])
    correct = np.concatenate([(np.arange(500) < 125), (np.arange(500) < 375)]).astype(
        float
    )
    assert ece(conf, correct, n_bins=10) < 0.02


def test_brier_bounds():
    assert brier([1, 0], [1, 0]) == 0.0
    assert brier([1, 0], [0, 1]) == 1.0


def test_reliability_curve_counts():
    conf = [0.05, 0.15, 0.95]
    rc = reliability_curve(conf, [0, 0, 1], n_bins=10)
    assert sum(b["count"] for b in rc["bins"]) == 3


def test_platt_reduces_ece_on_overconfident_data():
    conf, correct = _overconfident()
    split = len(conf) // 2
    res = calibrate_and_eval(
        conf[:split], correct[:split], conf[split:], correct[split:], method="platt"
    )
    assert res["calibrated"]["ece"] < res["raw"]["ece"]
    assert res["calibrated"]["brier"] <= res["raw"]["brier"] + 1e-6


def test_isotonic_reduces_ece_on_overconfident_data():
    conf, correct = _overconfident()
    split = len(conf) // 2
    res = calibrate_and_eval(
        conf[:split], correct[:split], conf[split:], correct[split:], method="isotonic"
    )
    assert res["calibrated"]["ece"] < res["raw"]["ece"]


def test_auto_picks_platt_for_small_sample():
    conf, correct = _overconfident(n=200)
    cal = fit_calibrator(conf, correct, method="auto")
    assert cal.method == "platt"  # 200 < MIN_N_FOR_ISOTONIC


def test_auto_picks_isotonic_for_large_sample():
    conf, correct = _overconfident(n=1000)
    cal = fit_calibrator(conf, correct, method="auto")
    assert cal.method == "isotonic"


def test_calibrator_output_in_unit_interval():
    conf, correct = _overconfident(n=500)
    for Cal in (PlattCalibrator, IsotonicCalibrator):
        out = Cal().fit(conf, correct).transform(conf)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_degenerate_all_correct_does_not_crash():
    out = PlattCalibrator().fit([0.3, 0.6, 0.9], [1, 1, 1]).transform([0.3, 0.6])
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_bootstrap_ci_brackets_point():
    conf, correct = _overconfident(n=600)
    point, lo, hi = bootstrap_metric_ci(conf, correct, metric="ece", n_boot=300, seed=3)
    assert lo <= point <= hi


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
