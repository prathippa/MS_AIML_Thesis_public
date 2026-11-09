from analysis.posthoc_v11 import decompose, wilson


def row(i, agree, listed, correct, conf):
    return {'id': f'p{i}', 'eligible': True, 'confidence': conf, 'agreement': agree, 'listed': listed,
            'correct': correct, 'wrong_chapter': False}


def test_decomposition_separates_agreed_unlisted_from_schedule_only():
    # Combined takes p0 (agree, listed) then p1 (agree, unlisted); schedule takes the two listed codes.
    rows = [row(0, True, True, True, .9), row(1, True, False, False, .8),
            row(2, False, True, True, .7), row(3, False, False, False, .6)]
    out = decompose(rows, {r['id']: True for r in rows}, coverage=0.5)
    assert out['both']['n'] == 1
    assert out['combined_only']['cells'] == {'agree_unlisted': {'n': 1, 'errors': 1, 'wrong_chapter': 0, 'gold_unlisted': 0}}
    assert out['schedule_only']['cells'] == {'disagree_listed': {'n': 1, 'errors': 0, 'wrong_chapter': 0, 'gold_unlisted': 0}}
    assert out['combined_agree_unlisted']['benchmark_correct'] == 0


def test_wilson_interval_brackets_estimate():
    lo, hi = wilson(7, 44)
    assert round(lo, 3) == 0.079 and round(hi, 3) == 0.294
