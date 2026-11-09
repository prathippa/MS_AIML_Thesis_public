"""Tests for judge.py: parsing, consistency, aggregation and Cohen's kappa."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from judge import (
    parse_judge_response,
    judge_one,
    run_judge,
    cohens_kappa,
    aggregate_judge,
    MockClient,
    build_judge_messages,
)

DEC = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 200}
CASE = {
    "id": "j1",
    "description": "cotton t-shirt",
    "predicted_code": "6109100012",
    "rationale": "knitted cotton shirt",
    "evidence": [
        {"id": "e1", "text": "Heading 6109 covers T-shirts, knitted."},
        {"id": "e2", "text": "Chapter 61 covers knitted apparel."},
    ],
}


def test_parse_valid_and_malformed():
    assert (
        parse_judge_response('{"label":"supported","reason":"ok"}')["label"]
        == "supported"
    )
    for bad in ['{"label":"maybe","reason":"x"}', "no json", '{"reason":"x"}']:
        try:
            parse_judge_response(bad)
            assert False
        except Exception:
            pass


def test_consistent_runs():
    client = MockClient(
        "mockJ", ['{"label":"supported","reason":"a"}']
    )  # same every call
    res = judge_one(client, CASE, DEC, n_runs=3)
    assert (
        res["majority"] == "supported"
        and res["consistency"] == 1.0
        and res["n_valid"] == 3
    )


def test_majority_with_disagreement():
    replies = [
        '{"label":"supported","reason":"a"}',
        '{"label":"supported","reason":"b"}',
        '{"label":"uncertain","reason":"c"}',
    ]
    client = MockClient("mockJ", replies)
    res = judge_one(client, CASE, DEC, n_runs=3, shuffle_evidence=False)
    assert res["majority"] == "supported" and abs(res["consistency"] - 2 / 3) < 1e-9


def test_evidence_order_randomised_but_stable_count():
    s1, u1 = build_judge_messages(CASE, CASE["evidence"])
    s2, u2 = build_judge_messages(CASE, list(reversed(CASE["evidence"])))
    assert (
        s1 == s2 and u1 != u2 and "e1" in u1 and "e1" in u2
    )  # same evidence, different order


def test_cohens_kappa_perfect_and_chance():
    a = ["supported", "unsupported", "uncertain", "supported"]
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-9
    # total disagreement with no overlap on the diagonal -> kappa <= 0
    b = ["unsupported", "supported", "supported", "uncertain"]
    assert cohens_kappa(a, b) <= 0.0


def test_aggregate_distribution():
    client = MockClient("mockJ", ['{"label":"supported","reason":"a"}'])
    recs = run_judge(client, [CASE, {**CASE, "id": "j2"}], DEC, n_runs=1)
    agg = aggregate_judge(recs)
    assert agg["n"] == 2 and abs(agg["label_distribution"]["supported"] - 1.0) < 1e-9


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
