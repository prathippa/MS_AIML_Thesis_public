"""
metrics.py
----------
Stages 3-4: turn per-case predictions into the RQ1 evidence.

Given records that pair a gold code with a predicted code, this computes:
  * conventional baselines  -- exact 10, exact 6, chapter, heading, rate-line accuracy;
  * prefix accuracy         -- at 2/4/6/8/10 digits (the 8-digit level added per Decision 2);
  * severity distribution   -- the six-tier ladder (correct_10 ... wrong_chapter);
  * hierarchical partial credit -- mean under each weight scheme (sensitivity analysis);
  * bootstrap CIs           -- for any scalar metric;
  * model comparison        -- the RQ1 contrast: two models can share exact-match
                               accuracy yet differ sharply in error severity.

Depends only on hts_hierarchy + numpy.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from hts_hierarchy import evaluate, SEVERITY_ORDER, WEIGHT_SCHEMES, LEVELS, LEVEL_NAME  # noqa: E402

SEVERITY_TIERS = [name for name, _ in SEVERITY_ORDER]


# --------------------------------------------------------------------------- #
# Per-case evaluation                                                         #
# --------------------------------------------------------------------------- #
def eval_records(records: list[dict]) -> list[dict]:
    """Attach the hierarchy evaluation to each {gold, pred, ...} record."""
    out = []
    for r in records:
        e = evaluate(r["pred"], r["gold"])
        e["case_id"] = r.get("case_id")
        e["model"] = r.get("model")
        if "confidence" in r:
            e["confidence"] = r["confidence"]
        out.append(e)
    return out


def _mean(evals, key) -> float:
    return float(np.mean([bool(e[key]) for e in evals])) if evals else 0.0


def _prefix_mean(evals, n) -> float:
    return float(np.mean([e["prefix_correct"][n] for e in evals])) if evals else 0.0


# --------------------------------------------------------------------------- #
# Aggregate metrics                                                           #
# --------------------------------------------------------------------------- #
def conventional_baselines(evals: list[dict]) -> dict:
    return {
        "exact_10": _mean(evals, "exact_10"),
        "exact_6": _mean(evals, "exact_6"),
        "chapter_acc": _prefix_mean(evals, 2),
        "heading_acc": _prefix_mean(evals, 4),
        "rate_line_acc": _prefix_mean(evals, 8),
    }


def prefix_accuracy(evals: list[dict]) -> dict:
    return {f"prefix_{n}_{LEVEL_NAME[n]}": _prefix_mean(evals, n) for n in LEVELS}


def severity_distribution(evals: list[dict], as_proportion=True) -> dict:
    counts = {t: 0 for t in SEVERITY_TIERS}
    for e in evals:
        counts[e["severity_tier"]] += 1
    n = len(evals)
    if as_proportion and n:
        return {t: counts[t] / n for t in SEVERITY_TIERS}
    return counts


def mean_partial_credit(evals: list[dict]) -> dict:
    return {
        scheme: (float(np.mean([e[f"partial_{scheme}"] for e in evals])) if evals else 0.0)
        for scheme in WEIGHT_SCHEMES
    }


def major_error_rate(evals: list[dict]) -> float:
    """Proportion placed in an entirely wrong chapter -- the costliest error."""
    return severity_distribution(evals)["wrong_chapter"]


def aggregate(records: list[dict]) -> dict:
    evals = eval_records(records)
    return {
        "n": len(evals),
        "conventional": conventional_baselines(evals),
        "prefix_accuracy": prefix_accuracy(evals),
        "severity_distribution": severity_distribution(evals),
        "mean_partial_credit": mean_partial_credit(evals),
        "major_error_rate": major_error_rate(evals),
        "mean_severity_ordinal": float(np.mean([e["severity_ordinal"] for e in evals])) if evals else 0.0,
    }


# --------------------------------------------------------------------------- #
# Bootstrap                                                                   #
# --------------------------------------------------------------------------- #
def bootstrap_ci(evals: list[dict], metric_fn, n_boot=2000, alpha=0.05, seed=7):
    """Return (point, lo, hi) for a scalar metric via case resampling.

    The ATLAS/CROSS test split is only 200 cases, so the proposal reports
    bootstrap CIs on the principal metrics; this is that routine.
    """
    rng = np.random.default_rng(seed)
    n = len(evals)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(metric_fn(evals))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = metric_fn([evals[i] for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_difference(evals_a, evals_b, metric_fn, n_boot=2000, alpha=0.05,
                         seed=7, paired=True):
    """(point, lo, hi) for metric_fn(A) - metric_fn(B).

    Both models are run on the same test cases, so the default is a PAIRED
    bootstrap (resample case indices once, apply to both) -- the correct way to
    test whether a severity or routing gap between two models is real. This is
    the 'bootstrap comparison' the proposal commits to.
    """
    rng = np.random.default_rng(seed)
    na, nb = len(evals_a), len(evals_b)
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0
    point = float(metric_fn(evals_a) - metric_fn(evals_b))
    boots = np.empty(n_boot)
    if paired:
        if na != nb:
            raise ValueError("paired bootstrap needs equal, case-aligned samples")
        for b in range(n_boot):
            idx = rng.integers(0, na, na)
            boots[b] = metric_fn([evals_a[i] for i in idx]) - metric_fn([evals_b[i] for i in idx])
    else:
        for b in range(n_boot):
            sa = [evals_a[i] for i in rng.integers(0, na, na)]
            sb = [evals_b[i] for i in rng.integers(0, nb, nb)]
            boots[b] = metric_fn(sa) - metric_fn(sb)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Model comparison (the RQ1 contrast)                                         #
# --------------------------------------------------------------------------- #
def compare_models(records_a: list[dict], records_b: list[dict],
                   label_a="Model A", label_b="Model B") -> dict:
    """Side-by-side summary foregrounding the RQ1 point: equal exact-match
    accuracy can hide very different error-severity profiles."""
    ea, eb = eval_records(records_a), eval_records(records_b)
    out = {label_a: {}, label_b: {}}
    for lbl, ev in ((label_a, ea), (label_b, eb)):
        out[lbl] = {
            "exact_10": _mean(ev, "exact_10"),
            "exact_6": _mean(ev, "exact_6"),
            "major_error_rate": major_error_rate(ev),
            "mean_severity_ordinal": float(np.mean([e["severity_ordinal"] for e in ev])) if ev else 0.0,
            "severity_distribution": severity_distribution(ev),
        }
    out["exact_10_gap"] = round(out[label_a]["exact_10"] - out[label_b]["exact_10"], 4)
    out["major_error_gap"] = round(out[label_a]["major_error_rate"] - out[label_b]["major_error_rate"], 4)
    return out
