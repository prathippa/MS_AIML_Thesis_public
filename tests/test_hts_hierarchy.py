"""Unit tests for the corrected hierarchy/severity logic (Decision 2)."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hts_hierarchy import (
    normalize_hts,
    severity,
    partial_credit,
    prefix_correct,
    evaluate,
    WEIGHT_SCHEMES,
)

GOLD = "0101210010"  # chapter 01, heading 0101, subheading 010121, rate line 01012100

CASES = {
    "0101210010": "correct_10",  # exact
    "0101210011": "wrong_stat_suffix",  # only digits 9-10 differ (rate line correct)
    "0101211010": "wrong_rate_line",  # digits 7-8 differ (subheading correct)
    "0101990010": "wrong_subheading",  # digits 5-6 differ (heading correct)
    "0199210010": "wrong_heading",  # digits 3-4 differ (chapter correct)
    "9999999999": "wrong_chapter",  # chapter differs
}


def test_severity_tiers():
    for pred, expected in CASES.items():
        assert severity(pred, GOLD).tier == expected, (
            pred,
            expected,
            severity(pred, GOLD).tier,
        )


def test_severity_ordinal_monotonic():
    # ordinals must increase as errors get worse, in the order defined above
    ords = [severity(p, GOLD).ordinal for p in CASES]
    assert ords == sorted(ords), ords


def test_normalize_recovers_leading_zero():
    # an integer-read code loses its leading zero; normalize must restore it
    assert normalize_hts(101210010) == "0101210010"
    assert normalize_hts("0101.21.0010") == "0101210010"


def test_partial_credit_bounds_and_order():
    scores = {p: partial_credit(p, GOLD) for p in CASES}
    assert scores["0101210010"] == 1.0  # exact -> full credit
    assert scores["9999999999"] == 0.0  # wrong chapter -> none
    # worse severity never scores higher than a milder one
    ordered = [
        "0101210010",
        "0101210011",
        "0101211010",
        "0101990010",
        "0199210010",
        "9999999999",
    ]
    vals = [scores[p] for p in ordered]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), vals


def test_all_weight_schemes_sum_to_one():
    for name, w in WEIGHT_SCHEMES.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, (name, sum(w.values()))


def test_prefix_correct_vector():
    pc = prefix_correct("0101211010", GOLD)  # wrong rate line
    assert pc == {2: True, 4: True, 6: True, 8: False, 10: False}, pc


def test_evaluate_record():
    rec = evaluate("0101210011", GOLD)  # wrong stat suffix
    assert rec["exact_10"] is False
    assert rec["exact_6"] is True
    assert rec["severity_tier"] == "wrong_stat_suffix"
    assert (
        rec["partial_balanced"] == 0.85
    )  # 0.15+0.20+0.30+0.20, missing 0.15 for digit 10


def test_short_code_rejected_not_padded():
    for bad in ["010121", "123", "01012100"]:  # 6/3/8 digits -> error, never padded
        try:
            normalize_hts(bad)
            assert False, f"should reject {bad}"
        except ValueError:
            pass
    assert (
        normalize_hts("101210010") == "0101210010"
    )  # 9-digit leading-zero recovery still works


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
