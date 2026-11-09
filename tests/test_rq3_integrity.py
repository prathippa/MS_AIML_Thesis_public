"""Reject corrupted or incomplete evaluator inputs before publishing a summary."""
import json
from pathlib import Path
import pytest
from analysis import rq3_judge as rq3


def fixture_root(tmp_path, monkeypatch):
    monkeypatch.setattr(rq3, 'verify_inputs', lambda root: None)
    out = tmp_path / 'rq3'
    out.mkdir()
    (tmp_path / 'test').mkdir()
    cases = [{'prediction_id': 'x|A'}]
    truth = [{'prediction_id': 'x|A', 'gold': '1234567890',
              'pred': '1234567890', 'correct': True, 'severity_tier': 'correct_10'}]
    result = [{'prediction_id': 'x|A', 'model': 'judge', 'labels': ['supported']}]
    rq3.write(out / 'rq3_manifest.json', {'sample_hash': rq3.digest(cases)})
    for filename, rows in [('judge_cases', cases), ('judge_case_truth', truth), ('judge_results', result)]:
        rq3.write_lines(out / f'{filename}.jsonl', rows)
    rq3.write_lines(tmp_path / 'test/scored_A.jsonl', truth)
    rq3.write_lines(tmp_path / 'test/scored_B.jsonl', [])
    return out, truth, result


@pytest.mark.parametrize('kind', ['duplicate_results', 'missing_results', 'duplicate_truth', 'changed_truth', 'changed_sample'])
def test_rejects_corrupt_rq3_inputs(tmp_path, monkeypatch, kind):
    out, truth, result = fixture_root(tmp_path, monkeypatch)
    if kind == 'duplicate_results':
        rq3.write_lines(out / 'judge_results.jsonl', result * 2)
    elif kind == 'missing_results':
        rq3.write_lines(out / 'judge_results.jsonl', [])
    elif kind == 'duplicate_truth':
        rq3.write_lines(out / 'judge_case_truth.jsonl', truth * 2)
    elif kind == 'changed_truth':
        truth[0]['correct'] = False
        rq3.write_lines(out / 'judge_case_truth.jsonl', truth)
    else:
        rq3.write_lines(out / 'judge_cases.jsonl', [{'prediction_id': 'foreign'}])
    with pytest.raises(ValueError):
        rq3.analyse(tmp_path, n_boot=2)
    assert not (out / 'rq3_summary.json').exists()
