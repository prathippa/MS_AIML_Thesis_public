"""
calibration.py
--------------
Stage 6 (first half of RQ2). Calibrate self-reported model confidence and
measure how well it matches empirical correctness.

Methods are those appropriate for SCALAR confidence (not logits):
  * Platt (logistic) scaling  -- the small-sample-safe default;
  * Isotonic regression       -- flexible, but needs more data for a stable fit.

Metrics: Expected Calibration Error (ECE), Brier score, and reliability-curve
points. Because the ATLAS/CROSS validation split is only 200 cases, `fit_calibrator`
defaults to Platt below a configurable sample size, and bootstrap CIs are
provided for ECE/Brier so conclusions reflect the limited sample.

Calibrators are fit on the validation split and applied to the test split --
never fit on the test set.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

MIN_N_FOR_ISOTONIC = 500   # below this, Platt is the safer default (design-lock)


def _arrays(conf, correct):
    c = np.asarray(conf, dtype=float).ravel()
    y = np.asarray(correct, dtype=float).ravel()
    if c.shape != y.shape:
        raise ValueError("confidence and correctness must be the same length")
    if np.any((c < 0) | (c > 1)):
        raise ValueError("confidences must lie in [0, 1]")
    return c, y


# --------------------------------------------------------------------------- #
# Calibration-quality metrics                                                 #
# --------------------------------------------------------------------------- #
def ece(conf, correct, n_bins=10) -> float:
    """Expected Calibration Error with equal-width bins over [0, 1]."""
    c, y = _arrays(conf, correct)
    if len(c) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += (m.mean()) * abs(y[m].mean() - c[m].mean())
    return float(total)


def brier(conf, correct) -> float:
    """Brier score = mean squared error between confidence and 0/1 correctness."""
    c, y = _arrays(conf, correct)
    return float(np.mean((c - y) ** 2)) if len(c) else 0.0


def reliability_curve(conf, correct, n_bins=10) -> dict:
    """Per-bin mean confidence, empirical accuracy and count for reliability diagrams."""
    c, y = _arrays(conf, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "count": int(m.sum()),
            "mean_confidence": float(c[m].mean()) if m.any() else None,
            "accuracy": float(y[m].mean()) if m.any() else None,
        })
    return {"n_bins": n_bins, "bins": bins}


def bootstrap_metric_ci(conf, correct, metric="ece", n_boot=2000, alpha=0.05,
                        seed=7, n_bins=10):
    """(point, lo, hi) for ECE or Brier via resampling of (confidence, label) pairs."""
    c, y = _arrays(conf, correct)
    rng = np.random.default_rng(seed)
    n = len(c)
    fn = (lambda cc, yy: ece(cc, yy, n_bins)) if metric == "ece" else brier
    if n == 0:
        return 0.0, 0.0, 0.0
    point = fn(c, y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boots[b] = fn(c[i], y[i])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Calibrators                                                                 #
# --------------------------------------------------------------------------- #
class PlattCalibrator:
    """Logistic (Platt) scaling: maps raw confidence through a fitted sigmoid."""
    method = "platt"
    def fit(self, conf, correct):
        c, y = _arrays(conf, correct)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")
        if len(np.unique(y)) < 2:          # degenerate: all right or all wrong
            self._const = float(y.mean()) if len(y) else 0.5
            self._lr = None
        else:
            self._lr.fit(c.reshape(-1, 1), y)
            self._const = None
        return self
    def transform(self, conf):
        c = np.asarray(conf, dtype=float).ravel()
        if self._lr is None:
            return np.full_like(c, self._const)
        return self._lr.predict_proba(c.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression: flexible monotonic map, clipped to [0, 1]."""
    method = "isotonic"
    def fit(self, conf, correct):
        c, y = _arrays(conf, correct)
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(c, y)
        return self
    def transform(self, conf):
        c = np.asarray(conf, dtype=float).ravel()
        return np.clip(self._iso.transform(c), 0.0, 1.0)


def fit_calibrator(conf, correct, method="auto", min_n_isotonic=MIN_N_FOR_ISOTONIC):
    """Fit and return a calibrator. 'auto' uses Platt when the fitting sample is
    smaller than `min_n_isotonic` (the 200-case validation split falls here)."""
    c, _ = _arrays(conf, correct)
    if method == "auto":
        method = "isotonic" if len(c) >= min_n_isotonic else "platt"
    cal = IsotonicCalibrator() if method == "isotonic" else PlattCalibrator()
    return cal.fit(conf, correct)


def calibrate_and_eval(val_conf, val_correct, test_conf, test_correct,
                       method="auto", n_bins=10):
    """Fit on validation, apply to test, and report ECE/Brier before and after.
    Returns the calibrated test confidences plus a metrics summary."""
    cal = fit_calibrator(val_conf, val_correct, method=method)
    cal_test = cal.transform(test_conf)
    return {
        "method": cal.method,
        "calibrated_test_confidence": cal_test,
        "raw": {"ece": ece(test_conf, test_correct, n_bins), "brier": brier(test_conf, test_correct)},
        "calibrated": {"ece": ece(cal_test, test_correct, n_bins), "brier": brier(cal_test, test_correct)},
    }
