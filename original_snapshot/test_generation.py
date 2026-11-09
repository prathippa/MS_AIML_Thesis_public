"""Offline tests for generation.py using MockClient (no SDKs, no network)."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generation import (
    parse_response, MalformedOutput, MockClient, generate_one, run_generation,
    build_messages,
)

DECODING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 600, "seed": 7}
CASE = {"id": "c1", "description": "Live purebred breeding horse"}
EVID = [{"id": "cross:HQ1", "text": "Chapter 01 covers live animals."}]


def test_parse_plain_json():
    out = parse_response('{"hts_code":"0101210010","confidence":0.8,"rationale":"live horse","evidence_ids":["cross:HQ1"]}')
    assert out["hts_code"] == "0101210010" and out["confidence"] == 0.8


def test_parse_fenced_and_prose():
    txt = 'Here is my answer:\n```json\n{"hts_code":"0101.21.0010","confidence":0.9,"rationale":"x","evidence_ids":[]}\n```'
    out = parse_response(txt)
    assert out["hts_code"] == "0101210010" and out["confidence"] == 0.9


def test_explicit_percent_string_honoured():
    out = parse_response('{"hts_code":"0101210010","confidence":"85%","rationale":"x","evidence_ids":[]}')
    assert out["confidence"] == 0.85


def test_malformed_raises():
    for bad in ['not json', '{"confidence":0.5}', '{"hts_code":"123","confidence":0.5,"rationale":"x"}',
                '{"hts_code":"0101210010","confidence":2,"rationale":"x"}',       # bare out-of-range
                '{"hts_code":"0101210010","confidence":85,"rationale":"x"}',      # bare out-of-range
                '{"hts_code":"0101210010","confidence":0.5,"rationale":""}']:
        try:
            parse_response(bad); assert False, f"should have raised: {bad}"
        except MalformedOutput:
            pass


def test_retry_once_recovers():
    client = MockClient("mockA", ["garbage no json",
                                  '{"hts_code":"0101210010","confidence":0.7,"rationale":"ok","evidence_ids":[]}'])
    rec = generate_one(client, CASE, EVID, DECODING, retry_once=True)
    assert rec.status == "ok" and rec.attempts == 2 and client.calls == 2


def test_failed_after_retry():
    client = MockClient("mockA", ["bad", "still bad"])
    rec = generate_one(client, CASE, EVID, DECODING, retry_once=True)
    assert rec.status == "failed" and rec.attempts == 2 and rec.error


def test_identical_prompt_across_models():
    s1, u1 = build_messages(CASE, EVID)
    s2, u2 = build_messages(CASE, EVID)
    assert s1 == s2 and u1 == u2   # prompt depends only on case+evidence, not model


def test_cache_resume_and_cap():
    cases = [{"id": f"c{i}", "description": f"product {i}"} for i in range(5)]
    good = '{"hts_code":"0101210010","confidence":0.6,"rationale":"ok","evidence_ids":[]}'
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gen.jsonl")
        # cap at 4 calls, retry off -> only 4 cases processed
        c1 = MockClient("mockB", [good])
        recs = run_generation(c1, cases, {}, DECODING, path, max_calls=4, retry_once=False)
        assert len(recs) == 4, len(recs)
        # resume: first 4 are cached, only the 5th is attempted now
        c2 = MockClient("mockB", [good])
        recs2 = run_generation(c2, cases, {}, DECODING, path, max_calls=4, retry_once=False)
        assert len(recs2) == 1 and recs2[0].case_id == "c4"
        # all 5 now persisted exactly once
        with open(path) as f:
            ids = [json.loads(l)["case_id"] for l in f]
        assert sorted(ids) == ["c0", "c1", "c2", "c3", "c4"]


def test_json_with_brace_inside_string():
    txt = 'Sure!\n{"hts_code":"0101210010","confidence":0.7,"rationale":"applies note 2(a} edge","evidence_ids":[]}'
    out = parse_response(txt)
    assert out["hts_code"] == "0101210010" and "note 2(a}" in out["rationale"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
