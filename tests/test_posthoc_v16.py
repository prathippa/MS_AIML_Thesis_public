import numpy as np
from calibration import ece
from analysis.posthoc_v16 import consistent_bins, ece_consistent, rq3_by_provision, routing_contrasts


def test_decimal_confidences_fall_in_their_own_bin():
    # The pipeline's linspace edges put 0.3, 0.6 and 0.7 in the bin below; the check must not.
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.29, 0.31]
    assert consistent_bins(values).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 0, 2, 3]
    edges = np.linspace(0, 1, 11)[1:-1]
    assert np.digitize([0.3, 0.6, 0.7], edges).tolist() == [2, 5, 6]


def test_consistent_ece_matches_pipeline_away_from_edges():
    conf = [0.05, 0.15, 0.25, 0.35, 0.85, 0.95]; y = [0, 0, 1, 0, 1, 1]
    assert abs(ece_consistent(conf, y) - ece(conf, y)) < 1e-12


def test_consistent_ece_differs_when_mass_sits_on_an_edge():
    conf = [0.3] * 4 + [0.25] * 4; y = [1, 1, 1, 1, 0, 0, 0, 0]
    assert abs(ece(conf, y) - 0.225) < 1e-9          # one mixed bin
    assert abs(ece_consistent(conf, y) - 0.475) < 1e-9  # two separate bins


def judged(i, exact, correct, majority, support):
    return {'prediction_id': f't:{i}|A', 'provision_exact_10': exact, 'correct': correct,
            'majority': majority, 'support': support}


def test_rq3_split_reports_flag_alone():
    rows = [judged(0, True, True, 'supported', 1.0), judged(1, True, False, 'uncertain', 0.5),
            judged(2, False, True, 'unsupported', 0.0), judged(3, False, False, 'unsupported', 0.0)]
    out = rq3_by_provision(rows)
    assert out['exact_provision_shown']['majority_labels'] == {'supported': 1, 'uncertain': 1}
    assert out['fallback_note_shown']['labels_on_correct'] == {'unsupported': 1}
    assert out['exact_provision_shown']['auc_support'] == 1.0
    assert out['auc_of_exact_provision_flag_alone'] == 0.5


def test_routing_contrast_point_estimates():
    def row(i, agree, listed, correct, conf, chapter=False):
        return {'id': f'p{i}', 'eligible': True, 'confidence': conf, 'agreement': agree, 'listed': listed,
                'correct': correct, 'wrong_chapter': chapter}
    rows = [row(0, True, True, True, .9), row(1, True, False, False, .8, True),
            row(2, False, True, True, .7), row(3, False, False, False, .6)]
    out = routing_contrasts(rows, coverage=0.5)
    assert out['combined_minus_schedule_risk'] == 0.5       # combined keeps p1, schedule keeps p2
    assert out['wrong_chapter_combined_minus_confidence'] == 0.0
