"""Tests for data_loader.py: real ATLAS messages parsing + stratified subset."""

import sys, os, json, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import (
    parse_atlas_assistant,
    load_atlas_messages,
    exclude_gold_ruling,
    stratified_judge_subset,
)


def test_parse_assistant_single_multi_noisy():
    assert parse_atlas_assistant("HTS US Code -> 2922.41.0010\nReasoning -> x")[0] == [
        "2922.41.0010"
    ]
    assert parse_atlas_assistant(
        "HTS US Code -> 8477.10.9015; 8480.71.8045\nReasoning -> y"
    )[0] == ["8477.10.9015", "8480.71.8045"]
    # parenthetical "$10" must not be picked up as a code
    codes, _ = parse_atlas_assistant(
        "HTS US Code -> 3926.20.6050 (if valued over $10) or 3926.20.9050\nReasoning -> z"
    )
    assert codes == ["3926.20.6050", "3926.20.9050"]


def test_load_atlas_messages():
    rows = [
        {
            "id": "a",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the HTS US Code for seamless copper tubes from Vietnam?",
                },
                {
                    "role": "assistant",
                    "content": "HTS US Code -> 7411.10.1090\nReasoning -> refined copper tubes.",
                },
            ],
        },
        {
            "id": "b",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the HTS US Code for an injection molding machine and molds?",
                },
                {
                    "role": "assistant",
                    "content": "HTS US Code -> 8477.10.9015; 8480.71.8045\nReasoning -> two articles.",
                },
            ],
        },
    ]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "atlas.jsonl")
        open(p, "w").write("\n".join(json.dumps(r) for r in rows))
        recs = load_atlas_messages(p)
    assert len(recs) == 2
    assert (
        recs[0]["description"] == "seamless copper tubes from Vietnam"
    )  # prefix + '?' stripped
    assert recs[0]["gold"] == "7411101090" and recs[0]["multi_code"] is False
    assert recs[1]["multi_code"] is True and recs[1]["gold_all"] == [
        "8477109015",
        "8480718045",
    ]


def test_exclude_gold_ruling():
    ev = [{"id": "HQ1", "text": "a"}, {"id": "HQ2", "text": "b"}]
    assert [e["id"] for e in exclude_gold_ruling(ev, "HQ1")] == ["HQ2"]


def test_stratified_subset_four_strata():
    recs = []
    recs += [
        {
            "case_id": f"ch{i}",
            "exact_10": True,
            "severity_tier": "correct_10",
            "confidence": 0.9,
        }
        for i in range(20)
    ]
    recs += [
        {
            "case_id": f"cl{i}",
            "exact_10": True,
            "severity_tier": "correct_10",
            "confidence": 0.3,
        }
        for i in range(20)
    ]
    recs += [
        {
            "case_id": f"nm{i}",
            "exact_10": False,
            "severity_tier": "wrong_rate_line",
            "confidence": 0.7,
        }
        for i in range(20)
    ]
    recs += [
        {
            "case_id": f"mj{i}",
            "exact_10": False,
            "severity_tier": "wrong_chapter",
            "confidence": 0.8,
        }
        for i in range(20)
    ]
    sub = stratified_judge_subset(recs, n_per_stratum=15, seed=1)
    assert {r["stratum"] for r in sub} == {
        "correct_high",
        "correct_low",
        "near_miss",
        "major_error",
    }
    assert len(sub) == 60


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
