"""Tests for metrics.py aggregation, bootstrap and the RQ1 model comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import (
    eval_records, conventional_baselines, severity_distribution, mean_partial_credit,
    prefix_accuracy, aggregate, bootstrap_ci, compare_models, major_error_rate, bootstrap_difference,
)

GOLD = "0101210010"
# five predictions landing in five distinct tiers
RECS = [
    {"case_id": "a", "gold": GOLD, "pred": "0101210010"},  # correct_10
    {"case_id": "b", "gold": GOLD, "pred": "0101210011"},  # wrong_stat_suffix (6 correct, 8 correct)
    {"case_id": "c", "gold": GOLD, "pred": "0101211010"},  # wrong_rate_line  (6 correct)
    {"case_id": "d", "gold": GOLD, "pred": "0101990010"},  # wrong_subheading (4 correct)
    {"case_id": "e", "gold": GOLD, "pred": "9999999999"},  # wrong_chapter
]


def test_conventional_baselines():
    ev = eval_records(RECS)
    cb = conventional_baselines(ev)
    assert cb["exact_10"] == 0.2            # 1 of 5 exact
    assert cb["exact_6"] == 0.6             # a,b,c correct through 6
    assert cb["chapter_acc"] == 0.8         # all but 'e'
    assert cb["rate_line_acc"] == 0.4       # a,b correct through 8


def test_severity_distribution_sums_to_one():
    ev = eval_records(RECS)
    dist = severity_distribution(ev)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["wrong_chapter"] == 0.2 and dist["correct_10"] == 0.2


def test_prefix_accuracy_keys_and_values():
    ev = eval_records(RECS)
    pa = prefix_accuracy(ev)
    assert pa["prefix_8_rate_line"] == 0.4
    assert pa["prefix_2_chapter"] == 0.8


def test_mean_partial_credit_between_zero_and_one():
    ev = eval_records(RECS)
    mpc = mean_partial_credit(ev)
    assert set(mpc) == {"balanced", "subheading_heavy", "extension_heavy"}
    assert all(0.0 <= v <= 1.0 for v in mpc.values())


def test_bootstrap_ci_brackets_point():
    ev = eval_records(RECS)
    point, lo, hi = bootstrap_ci(ev, lambda e: sum(x["exact_10"] for x in e) / len(e),
                                 n_boot=500, seed=1)
    assert lo <= point <= hi and 0.0 <= lo <= hi <= 1.0


def test_rq1_contrast_same_exact_different_severity():
    # both models: 4 of 8 exact -> identical exact_10, but very different severity
    exact = {"gold": GOLD, "pred": "0101210010"}
    minor = {"gold": GOLD, "pred": "0101210011"}   # wrong_stat_suffix
    major = {"gold": GOLD, "pred": "9999999999"}   # wrong_chapter
    model_a = [exact] * 4 + [minor] * 4            # mild errors
    model_b = [exact] * 4 + [major] * 4            # severe errors
    cmp = compare_models(model_a, model_b)
    assert cmp["Model A"]["exact_10"] == cmp["Model B"]["exact_10"] == 0.5
    assert cmp["exact_10_gap"] == 0.0              # identical on exact match
    assert cmp["Model A"]["major_error_rate"] == 0.0
    assert cmp["Model B"]["major_error_rate"] == 0.5
    assert cmp["major_error_gap"] == -0.5          # the difference exact match hides


def test_aggregate_shape():
    agg = aggregate(RECS)
    assert agg["n"] == 5
    for k in ("conventional", "prefix_accuracy", "severity_distribution",
              "mean_partial_credit", "major_error_rate", "mean_severity_ordinal"):
        assert k in agg


def test_bootstrap_difference_paired():
    exact = {"gold": GOLD, "pred": "0101210010"}
    major = {"gold": GOLD, "pred": "9999999999"}
    a = eval_records([exact] * 4 + [major] * 4)   # major_error_rate 0.5
    b = eval_records([exact] * 8)                  # major_error_rate 0.0
    point, lo, hi = bootstrap_difference(a, b, major_error_rate, n_boot=500, seed=2)
    assert abs(point - 0.5) < 1e-9 and lo <= point <= hi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
