"""Tests for the post-hoc RQ-alignment analyses and the thesis-claims checker."""

import copy, json, os, sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from rq_alignment import _policies, _risk, _select  # noqa: E402
from supplementary import auc, tie_averaged_risk  # noqa: E402
import thesis_claims as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def row(pid, conf, pred, correct):
    return {"prediction_id": pid, "confidence": conf, "pred": pred, "exact_10": correct}


def test_select_rounds_half_up_and_keeps_one():
    rs = [row(str(i), i / 10, "x", True) for i in range(10)]
    assert len(_select(rs, 0.7, lambda r: -r["confidence"])) == 7
    assert len(_select(rs[:1], 0.1, lambda r: -r["confidence"])) == 1


def test_schedule_check_escalates_invalid_codes_first():
    ten = {"0101210000", "0101290000"}
    rs = [row("a", 0.95, "9999999999", False),   # most confident, but not a schedule line
          row("b", 0.60, "0101210000", True),
          row("c", 0.50, "0101290000", True),
          row("d", 0.40, "0101290000", False)]
    conf, sched = _policies(rs, ten, 0.5)
    assert [r["prediction_id"] for r in conf] == ["a", "b"]
    assert [r["prediction_id"] for r in sched] == ["b", "c"]
    assert _risk(conf) == 0.5 and _risk(sched) == 0.0


def test_schedule_check_equals_confidence_when_all_codes_valid():
    ten = {"0101210000"}
    rs = [row(str(i), c, "0101210000", i % 2 == 0) for i, c in enumerate([0.9, 0.7, 0.5, 0.3])]
    conf, sched = _policies(rs, ten, 0.75)
    assert conf == sched


def test_tie_averaged_risk_equals_plain_risk_without_ties():
    conf, y = [0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 1, 1, 0]
    assert tie_averaged_risk(conf, y, 0.6) == _risk([row(str(i), c, "x", bool(v))
                                                     for i, (c, v) in enumerate(zip(conf[:3], y[:3]))])


def test_tie_averaged_risk_splits_a_tie_at_the_cut():
    # k = 2: the 0.9 case (correct) plus one of three tied cases, of which one is correct
    assert abs(tie_averaged_risk([0.9, 0.5, 0.5, 0.5], [1, 1, 0, 0], 0.5) - 1 / 3) < 1e-12


def test_monotone_transform_preserves_auc():
    scores, labels = [0.9, 0.7, 0.6, 0.4, 0.2], [1, 0, 1, 0, 0]
    squashed = [s ** 3 / 10 for s in scores]
    assert auc(scores, labels) == auc(squashed, labels)


@pytest.fixture(scope="module")
def stored():
    needed = [ROOT / "results" / f for f in ("supplementary.json", "rq_alignment.json")]
    if not all(p.exists() for p in needed):
        pytest.skip("stored results not present")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        yield tc.load()
    finally:
        os.chdir(cwd)


def test_every_thesis_claim_holds_on_stored_results(stored):
    failed = [r["id"] for r in tc.claims(stored) if not r["holds"]]
    assert failed == []


def test_checker_detects_a_changed_result(stored):
    d = copy.deepcopy(stored)
    d["rq3"]["unanimous_rate"] = 0.5
    d["routing"]["main_v1"]["ci95_raw_minus_calibrated"] = [0.01, 0.05]
    status = {r["id"]: r["holds"] for r in tc.claims(d)}
    assert not status["C18"] and not status["C13"] and not status["H2"]


def test_constant_prevalence_brier_matches_calibration_file(stored):
    """The baseline's calibrated Brier must agree with the sealed calibration.json."""
    cal = json.load(open(ROOT / "results" / "main_v1" / "calibration.json"))
    base = stored["supp"]["main_v1"]["constant_prevalence_baseline"]
    for m in ("A", "B"):
        assert abs(base[m]["brier_calibrated"] - round(cal[m]["calibrated"]["brier"], 3)) <= 0.001


def test_checker_flags_a_wrong_judge_identity(stored):
    d = copy.deepcopy(stored)
    d["judge_models_used"] = ["openai/gpt-5-mini"]  # not what the outputs actually recorded
    status = {r["id"]: r["holds"] for r in tc.claims(d)}
    assert not status["P1"]


def test_checker_flags_a_changed_withholding_count(stored):
    d = copy.deepcopy(stored)
    d["supp"]["main_v1"]["rq3_withholding"]["withheld"]["not_supported"] = 3
    status = {r["id"]: r["holds"] for r in tc.claims(d)}
    assert not status["C24"]
