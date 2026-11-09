"""OFFLINE SOFTWARE FIXTURES ONLY: no real predictions, annotations or findings."""

import json, csv
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import config
from pipeline import init, Study
from study_io import write, read, write_lines, lines, digest, bind, verify_bound, lock
from generation import (
    MockClient,
    parse_response,
    MalformedOutput,
    generate_one,
    run_generation,
    AnthropicClient,
)
from budget import Budget, BudgetedClient, BudgetExceeded
from empirical_report import build_report
from hts_hierarchy import normalize_hts
import calibration as C
import routing as R
from data_loader import load_atlas_messages, stratified_judge_subset
from judge import judge_one, cohens_kappa, parse_judge_response
from annotations import export, validate
from evidence import corpus

GOOD = "0101210010"
BAD = "9999999999"
DEC = {"max_tokens": 1200}


def reply(code=GOOD, confidence=0.7):
    return json.dumps(
        {
            "hts_code": code,
            "confidence": confidence,
            "rationale": "OFFLINE fixture rationale",
            "evidence_ids": [],
        }
    )


def models():
    return {
        "A": {
            "model": "offline-A",
            "provider": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "B": {"model": "offline-B", "provider": "groq", "api_key_env": "GROQ_API_KEY"},
        "J": {"model": "offline-J", "provider": "groq", "api_key_env": "GROQ_API_KEY"},
    }


def make_study(tmp, mode="retrieved", budget=200):
    m = models()
    root = tmp / "study"
    with patch.object(config, "MODEL_A", m["A"]), patch.object(
        config, "MODEL_B", m["B"]
    ), patch.object(config, "JUDGE_MODEL", m["J"]):
        init(root, "OFFLINE-FIXTURE", mode, max_calls=budget, human_size=2, n_boot=20)
    files = {}
    for split in ("validation", "test"):
        rows = []
        for i in range(4):
            rows.append(
                {
                    "id": str(i),
                    "source_ruling_id": f"OFFLINE-{split}-{i}",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"What is the HTS US Code for copper {split} fixture item {i}?",
                        },
                        {
                            "role": "assistant",
                            "content": f"HTS US Code -> {GOOD if i<2 else BAD}\nReasoning -> PRIVATE GOLD ANSWER {split} {i}",
                        },
                    ],
                }
            )
        files[split] = tmp / f"{split}.jsonl"
        write_lines(files[split], rows)
    s = Study(root)
    s.prepare(files["validation"], files["test"])
    cp = tmp / "corpus.jsonl"
    write_lines(
        cp,
        [
            {
                "id": "ev1",
                "text": "Copper tariff notes for illustration of retrieval only.",
                "source_url": "https://example.org/OFFLINE-FIXTURE",
                "source_ruling_id": "OFFLINE-TRAIN-1",
                "accessed_at": "OFFLINE-FIXTURE",
                "version": "OFFLINE-FIXTURE",
            }
        ],
    )
    s.evidence(cp if mode == "retrieved" else None)
    return s, files, cp


def populate(s):
    for split in ("validation", "test"):
        s.generate(
            split,
            "A",
            MockClient(
                "offline-A",
                [reply(GOOD, 0.8), reply(BAD, 0.7), reply(BAD, 0.6), reply(BAD, 0.5)],
            ),
        )
        s.generate(
            split,
            "B",
            MockClient(
                "offline-B",
                [reply(GOOD, 0.99), reply(GOOD, 0.95), "bad", "bad", reply(GOOD, 0.9)],
            ),
        )
    return s


def filled_annotations(s, tmp):
    path = s.export_human()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(
            label="supported",
            reason="OFFLINE ANNOTATION FIXTURE",
            annotator_id="OFFLINE-HUMAN",
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    att = tmp / "attestation.json"
    write(
        att,
        {
            "annotator_id": "OFFLINE-HUMAN",
            "expertise": "OFFLINE TEST ONLY, not a real reviewer",
            "independent_of_generation": True,
            "blinded_to_gold_and_judge": True,
            "rubric_version": "v2",
        },
    )
    return path, att


def test_real_orchestration_offline_contract(tmp_path):
    s, _, _ = make_study(tmp_path)
    populate(s)
    m = s.metrics()
    assert m["B"]["n_all"] == 4 and m["B"]["n_valid"] == 3 and m["B"]["n_failed"] == 1
    assert m["B"]["all_case_exact10_accuracy"] == 0.5
    c = s.calibrate()
    assert c["A"]["n_validation_valid"] == 4
    r = s.route()
    assert (
        r["n_pooled"] == 7
        and r["n_case_clusters"] == 4
        and r["denominators"]["test"]["forced_escalations"] == 1
    )
    out = build_report(s.root)
    assert out["human_annotations_complete"] is False
    with pytest.raises(ValueError, match="annotations"):
        build_report(s.root, require_human=True)
    s.prepare_judge()
    s.judge(
        "J",
        MockClient(
            "offline-J", ['{"label":"supported","reason":"OFFLINE judge fixture"}']
        ),
    )
    path, att = filled_annotations(s, tmp_path)
    content = Path(path).read_text()
    assert "PRIVATE GOLD ANSWER" not in content and "generator_model" not in content
    h = s.validate_human(path, att)
    assert (
        h["n"] == 2
        and h["human_annotations_complete"]
        and not h["judge_validity_established"]
    )
    result = build_report(s.root, require_human=True)
    assert Path(result["report"]).exists()
    assert all(
        not x["judge_is_generator"] for x in lines(s.root / "judge_results.jsonl")
    )


@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        None,
        float("nan"),
        float("inf"),
        -0.01,
        1.1,
        "85",
        "not a probability",
    ],
)
def test_bad_generation_probabilities_rejected(bad):
    obj = json.loads(reply())
    obj["confidence"] = bad
    with pytest.raises(MalformedOutput):
        parse_response(json.dumps(obj))


@pytest.mark.parametrize(
    "bad", ["abc0101210010", "0101210010xyz", 101210010, None, "123", "010121", ""]
)
def test_bad_generated_codes_rejected(bad):
    obj = json.loads(reply())
    obj["hts_code"] = bad
    with pytest.raises(MalformedOutput):
        parse_response(json.dumps(obj))


@pytest.mark.parametrize("bad", [[], True, 1, "string", {"x": 1}])
def test_bad_evidence_list_rejected(bad):
    if bad == []:
        bad = [1]
    obj = json.loads(reply())
    obj["evidence_ids"] = bad
    with pytest.raises(MalformedOutput):
        parse_response(json.dumps(obj))


@pytest.mark.parametrize("bad", ["0101x210010", True, 101210010.0, "123", "01012100"])
def test_strict_normalization(bad):
    with pytest.raises(ValueError):
        normalize_hts(bad)


@pytest.mark.parametrize(
    "values", [[float("nan")], [1.1], [-0.1], [[0.5]], [float("inf")]]
)
def test_invalid_calibration_probabilities(values):
    with pytest.raises(ValueError):
        C.ece(values, [1])


@pytest.mark.parametrize("values", [[2], [-1], [0.5], [float("nan")], [[1]]])
def test_invalid_calibration_labels(values):
    with pytest.raises(ValueError):
        C.brier([0.5], values)


@pytest.mark.parametrize("coverage", [0, -0.1, 1.1, float("nan")])
def test_invalid_coverage(coverage):
    with pytest.raises(ValueError):
        R.routing_at_coverage([0.5], [1], coverage)


def test_threshold_does_not_split_tied_group():
    fit = R.threshold_for_target_risk([0.9, 0.9], [1, 0], 0)
    assert fit["threshold"] is None and fit["n_streamlined"] == 0
    assert R.apply_threshold([0.9, 0.9], [1, 0], fit["threshold"])["n_streamlined"] == 0


def test_identity_ties_ignore_input_order():
    ids = ["c", "a", "b", "d"]
    x = [0.8] * 4
    y = [0, 1, 1, 0]
    a = R.routing_at_coverage(x, y, 0.5, ids)
    p = [3, 1, 0, 2]
    b = R.routing_at_coverage(
        [x[i] for i in p], [y[i] for i in p], 0.5, [ids[i] for i in p]
    )
    assert a == b


def test_calibration_slope_nonnegative():
    c = C.PlattCalibrator().fit([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert c.slope_ >= 0
    z = c.transform([0.1, 0.2, 0.8, 0.9])
    assert np.all(np.diff(z) >= 0)


def test_judge_tie_abstains_and_persists_raw_order():
    c = MockClient(
        "m",
        [
            json.dumps({"label": lab, "reason": "OFFLINE fixture"})
            for lab in ("supported", "uncertain", "unsupported")
        ],
    )
    r = judge_one(
        c,
        {
            "case_id": "c",
            "prediction_id": "c|A",
            "description": "fixture",
            "predicted_code": GOOD,
            "evidence": [{"id": "x", "text": "fixture"}],
        },
        DEC,
    )
    assert r["majority"] is None and r["consistency"] == 0 and len(r["history"]) == 3
    assert all("raw" in h and h["evidence_order"] == ["x"] for h in r["history"])


def test_one_valid_judge_run_not_full_consistency():
    c = MockClient(
        "m", ["bad", "bad", '{"label":"supported","reason":"OFFLINE fixture"}']
    )
    r = judge_one(c, {"id": "c", "description": "fixture", "predicted_code": GOOD}, DEC)
    assert r["n_valid"] == 1 and r["majority"] is None and r["consistency"] == 0


def test_degenerate_kappa_undefined():
    assert np.isnan(cohens_kappa(["supported"], ["supported"]))


def test_empty_judge_reason_rejected():
    with pytest.raises(MalformedOutput):
        parse_judge_response('{"label":"supported","reason":""}')


def test_api_error_is_saved_not_silent():
    class Fail:
        model = "offline"

        def complete(self, *a):
            raise RuntimeError("secret not serialized")

    r = generate_one(Fail(), {"case_id": "c", "description": "fixture"}, [], DEC)
    assert (
        r.status == "failed"
        and r.history[0]["error"] == "RuntimeError"
        and "secret" not in str(r.history)
    )


def test_budget_persists_across_client_and_restarts(tmp_path):
    p = tmp_path / "budget.sqlite"
    c = BudgetedClient(MockClient("m", [reply()]), Budget(p, 1), {})
    c.complete("s", "u", DEC)
    assert Budget(p, 1).used == 1
    with pytest.raises(BudgetExceeded):
        c.complete("s", "u", DEC)
    with pytest.raises(ValueError):
        Budget(p, 2)


def test_bad_local_options_do_not_reserve(tmp_path):
    c = object.__new__(AnthropicClient)
    c.model = c.name = "offline"
    b = BudgetedClient(c, Budget(tmp_path / "b.sqlite", 1), {})
    with pytest.raises(ValueError):
        b.complete("s", "u", {"temperature": 0})
    assert b.budget.used == 0


def test_cache_invalidates_changed_prompt_model_or_settings(tmp_path):
    p = str(tmp_path / "gen.jsonl")
    case = [{"case_id": "c", "description": "fixture"}]
    run_generation(MockClient("a", [reply()]), case, {}, DEC, p, retry_once=False)
    for client, cases, dec in [
        (MockClient("b", [reply()]), case, DEC),
        (MockClient("a", [reply()]), [{"case_id": "c", "description": "changed"}], DEC),
        (MockClient("a", [reply()]), case, {"max_tokens": 99}),
    ]:
        with pytest.raises(ValueError):
            run_generation(client, cases, {}, dec, p, retry_once=False)


def test_missing_outputs_count_as_wrong(tmp_path):
    s, _, _ = make_study(tmp_path, "none")
    s.generate("test", "A", MockClient("offline-A", [reply()]))
    m = s.metrics()
    assert m["B"]["n_missing"] == 4 and m["B"]["all_case_exact10_accuracy"] == 0


def test_no_evidence_cannot_report_rq3(tmp_path):
    s, _, _ = make_study(tmp_path, "none")
    with pytest.raises(ValueError, match="RQ3"):
        s.prepare_judge()


def test_overlapping_splits_rejected(tmp_path):
    s, files, _ = make_study(tmp_path)
    with pytest.raises(ValueError, match="Identical"):
        s.prepare(files["test"], files["test"])


def test_protocol_edit_rejected(tmp_path):
    s, _, _ = make_study(tmp_path)
    p = read(s.root / "protocol.json")
    p["coverage"] = 0.5
    write(s.root / "protocol.json", p)
    with pytest.raises(ValueError, match="Protocol"):
        Study(s.root)


def test_input_record_edit_rejected(tmp_path):
    s, _, _ = make_study(tmp_path)
    p = s.path("test", "records.jsonl")
    r = lines(p)
    r[0]["gold"] = BAD
    write_lines(p, r)
    with pytest.raises(ValueError, match="records"):
        s.records("test")


def test_scored_artifact_edit_rejected(tmp_path):
    s, _, _ = make_study(tmp_path)
    populate(s)
    s.calibrate()
    p = s.path("test", "scored_A.jsonl")
    r = lines(p)
    r[0]["calibrated_confidence"] = 0.123
    write_lines(p, r)
    with pytest.raises(ValueError, match="calibration"):
        s.route()


def test_generation_cache_edit_rejected(tmp_path):
    s, _, _ = make_study(tmp_path)
    populate(s)
    p = s.path("test", "generations_A.jsonl")
    r = lines(p)
    r[0]["confidence"] = 0.123
    write_lines(p, r)
    with pytest.raises(ValueError, match="generation"):
        s.joined("test", "A")


def test_heldout_ruling_rejected_from_corpus(tmp_path):
    s, _, cp = make_study(tmp_path)
    r = lines(cp)
    r[0]["source_ruling_id"] = "OFFLINE-test-0"
    write_lines(cp, r)
    with pytest.raises(ValueError, match="Held-out"):
        corpus(cp, s.records("validation") + s.records("test"))


def test_missing_source_ruling_fails_closed(tmp_path):
    s, _, cp = make_study(tmp_path)
    r = s.records("test")
    r[0]["source_ruling_id"] = None
    with pytest.raises(ValueError, match="EVERY"):
        corpus(cp, r)


def test_multi_code_exclusion_audited(tmp_path):
    p = tmp_path / "data.jsonl"
    write_lines(
        p,
        [
            {
                "messages": [
                    {"role": "user", "content": "fixture"},
                    {"role": "assistant", "content": f"HTS US Code -> {GOOD}; {BAD}"},
                ]
            }
        ],
    )
    rs, a = load_atlas_messages(p, "test", "exclude", True)
    assert rs == [] and a["excluded_count"] == 1


def test_wrong_subheading_included_in_judge_sample():
    r = {
        "case_id": "x",
        "model": "A",
        "exact_10": False,
        "severity_tier": "wrong_subheading",
        "confidence": 0.7,
    }
    assert stratified_judge_subset([r])[0]["stratum"] == "major_error"


def test_human_annotation_content_change_rejected(tmp_path):
    s, _, _ = make_study(tmp_path)
    populate(s)
    s.prepare_judge()
    s.judge(
        "J",
        MockClient("offline-J", ['{"label":"supported","reason":"OFFLINE fixture"}']),
    )
    path, att = filled_annotations(s, tmp_path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["description"] = "CHANGED"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="content"):
        s.validate_human(path, att)


def test_experiment_lock(tmp_path):
    with lock(tmp_path):
        with pytest.raises(RuntimeError):
            with lock(tmp_path):
                pass
    assert not (tmp_path / ".active.lock").exists()


def test_plain_json_nonfinite_serializes_null(tmp_path):
    p = tmp_path / "x.json"
    write(p, {"x": float("nan")})
    assert read(p) == {"x": None}
