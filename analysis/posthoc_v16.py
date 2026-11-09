"""Post-hoc sensitivity checks added at the final review (v16); offline, no API calls.

Written after all results were known. It reads sealed outputs, leaves the frozen pipeline
modules and every earlier result file unchanged, and writes only to results/posthoc_v16/.
  1. ECE bin-edge sensitivity. calibration.ece builds its edges with numpy.linspace, whose
     values for 0.3, 0.6 and 0.7 are one ulp above the decimals, so a stated confidence of
     exactly 0.30, 0.60 or 0.70 is counted in the bin below. Here every bin is [lo, hi).
  2. RQ3 verdicts split by whether the evidence pack showed an exact 10-digit provision or the
     "not in the current schedule" fallback note, which the evaluator could read.
  3. Follow-up, GPT-5 at 50% coverage: paired interval for combined minus schedule-only risk
     and for the wrong-chapter difference between combined and confidence-only routing.
  4. Open-weight study denominators (all attempted cases versus valid outputs).
"""
from pathlib import Path
import json, re, sys
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from calibration import ece
from study_io import read, write, file_hash
from analysis.product_routing import product_rows, select, metrics
from analysis.rq3_judge import validated_rows, auc

N_BOOT, SEED = 2000, 7
STUDIES = ('main_v1', 'oss_v1', 'hscc_v1', 'prospective_v8a')


def consistent_bins(conf, n_bins=10):
    """Bin index with every bin closed below and open above; 1.0 joins the top bin."""
    c = np.asarray(conf, float)
    return np.minimum(np.floor(np.round(c * n_bins, 9)).astype(int), n_bins - 1)


def ece_consistent(conf, correct, n_bins=10):
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    idx = consistent_bins(c, n_bins)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - c[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def ece_sensitivity(scored):
    rng = np.random.default_rng(SEED)
    out = {}
    for m in 'AB':
        rs = [r for r in scored if r['model'] == m and r['status'] == 'ok']
        raw = np.array([r['confidence'] for r in rs]); cal = np.array([r['calibrated_confidence'] for r in rs])
        y = np.array([r['correct'] for r in rs], float)
        moved = int((consistent_bins(raw) != np.clip(np.digitize(raw, np.linspace(0, 1, 11)[1:-1]), 0, 9)).sum())
        draws = [rng.integers(0, len(rs), len(rs)) for _ in range(N_BOOT)]
        gap = [ece_consistent(raw[i], y[i]) - ece_consistent(cal[i], y[i]) for i in draws]
        out[m] = {'n': len(rs), 'raw_predictions_rebinned': moved,
                  'pipeline': {'raw': ece(raw, y), 'calibrated': ece(cal, y)},
                  'consistent_edges': {'raw': ece_consistent(raw, y), 'calibrated': ece_consistent(cal, y),
                                       'reduction': ece_consistent(raw, y) - ece_consistent(cal, y),
                                       'reduction_ci95': [float(v) for v in np.percentile(gap, [2.5, 97.5])]}}
    return out


def rq3_by_provision(rows):
    out = {}
    for name, flag in (('exact_provision_shown', True), ('fallback_note_shown', False)):
        rs = [r for r in rows if r['provision_exact_10'] is flag]
        pos = [r['support'] for r in rs if r['correct']]; neg = [r['support'] for r in rs if not r['correct']]
        out[name] = {'n': len(rs), 'correct': len(pos), 'wrong': len(neg),
                     'majority_labels': dict(Counter(r['majority'] for r in rs)),
                     'labels_on_correct': dict(Counter(r['majority'] for r in rs if r['correct'])),
                     'auc_support': auc(pos, neg)}
    pos = [float(r['provision_exact_10']) for r in rows if r['correct']]
    neg = [float(r['provision_exact_10']) for r in rows if not r['correct']]
    out['auc_of_exact_provision_flag_alone'] = auc(pos, neg)
    out['auc_support_all'] = auc([r['support'] for r in rows if r['correct']],
                                 [r['support'] for r in rows if not r['correct']])
    return out


def routing_contrasts(rows, coverage=0.5):
    def point(batch):
        m = {p: metrics(batch, select(batch, coverage, p)) for p in ('confidence', 'schedule', 'combined')}
        return (m['combined']['selective_risk'] - m['schedule']['selective_risk'],
                m['combined']['wrong_chapter_risk'] - m['confidence']['wrong_chapter_risk'])
    rng = np.random.default_rng(SEED)
    boots = np.array([point([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(N_BOOT)])
    risk, chapter = point(rows)
    return {'coverage': coverage, 'n_products': len(rows),
            'combined_minus_schedule_risk': risk,
            'combined_minus_schedule_ci95': [float(v) for v in np.percentile(boots[:, 0], [2.5, 97.5])],
            'wrong_chapter_combined_minus_confidence': chapter,
            'wrong_chapter_ci95': [float(v) for v in np.percentile(boots[:, 1], [2.5, 97.5])]}


def oss_denominators(scored):
    rs = [r for r in scored if r['model'] == 'B']; ok = [r for r in rs if r['status'] == 'ok']
    six = sum(bool(r['prefix_correct']['6']) for r in ok); chapter = sum(r['severity_tier'] == 'wrong_chapter' for r in ok)
    return {'n_all': len(rs), 'n_valid': len(ok), 'correct_to_6': six, 'wrong_chapter': chapter,
            'correct_to_6_rate_valid': six / len(ok), 'correct_to_6_rate_all': six / len(rs),
            'wrong_chapter_rate_valid': chapter / len(ok), 'wrong_chapter_rate_all': chapter / len(rs)}


def main():
    base = Path(__file__).resolve().parents[1]; results = base / 'results'
    hts = base / 'data/hts_2026_revision_22.json'
    if file_hash(hts) != read(results / 'main_v1/rq3/rq3_manifest.json')['hts_export']['file']:
        raise ValueError('Tariff snapshot changed')
    codes = {re.sub(r'\D', '', r.get('htsno', '')) for r in read(hts)}; codes = {c for c in codes if len(c) == 10}
    scored = {s: Study(results / s).scored('test') for s in STUDIES}
    truth, judge_model = validated_rows(results / 'main_v1')
    result = {'status': 'post_hoc_sensitivity_checks_after_all_results; frozen pipeline and earlier outputs unchanged',
              'source_sha256': file_hash(Path(__file__)), 'n_boot': N_BOOT, 'seed': SEED,
              'inputs': {f'{s}/test/scored_{m}.jsonl': file_hash(results / s / 'test' / f'scored_{m}.jsonl')
                         for s in STUDIES for m in 'AB'},
              'ece_bin_edge_sensitivity': {s: ece_sensitivity(scored[s]) for s in STUDIES},
              'rq3_by_provision': {'judge_model': judge_model, **rq3_by_provision(truth)},
              'followup_gpt5_50pct': routing_contrasts(product_rows(scored['prospective_v8a'], 'B', codes)),
              'oss_model_b_denominators': oss_denominators(scored['oss_v1'])}
    write(results / 'posthoc_v16/results.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
