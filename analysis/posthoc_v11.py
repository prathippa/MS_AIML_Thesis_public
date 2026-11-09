"""Post-hoc explanations for the completed follow-up (v11); offline, no API calls.

Written after the frozen primary result was known. It reads the sealed prospective_v8a outputs,
reuses the frozen ranking code unchanged, and writes only to results/posthoc_v11/.
  1. Paired GPT-5 minus Claude accuracy at each code depth, for comparison with the main study.
  2. Whether either output cap was reached.
  3. Decomposition of the 50% GPT-5 combined and schedule-only selections.
"""
from pathlib import Path
import json, math, re, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from study_io import read, write, lines, file_hash
from analysis.product_routing import product_rows, select

DEPTHS = ('2', '4', '6', '8', '10')


def wilson(k, n, z=1.959964):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]


def paired_depths(scored, n_boot=2000, seed=7):
    by = {m: {r['case_id']: r for r in scored if r['model'] == m} for m in 'AB'}
    ids = sorted(by['A'])
    if set(ids) != set(by['B']):
        raise ValueError('Products not paired')
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(ids), (n_boot, len(ids)))
    out = {}
    for d in DEPTHS:
        a = np.array([bool(by['A'][i]['prefix_correct'][d]) for i in ids], float)
        b = np.array([bool(by['B'][i]['prefix_correct'][d]) for i in ids], float)
        diff = b-a; boots = diff[draws].mean(axis=1)
        out[d] = {'claude': a.mean(), 'gpt5': b.mean(), 'gpt5_minus_claude': diff.mean(),
                  'paired_ci95': [float(v) for v in np.percentile(boots, [2.5, 97.5])]}
    return out


def output_caps(root):
    out = {}
    for label, field, cap, stop in (('A', 'output_tokens', 4000, 'max_tokens'), ('B', 'completion_tokens', 8000, 'length')):
        used, stops = [], 0
        for split in ('validation', 'test'):
            for g in lines(root/split/f'generations_{label}.jsonl'):
                for h in g['history']:
                    m = h['metadata']; used.append(m['usage'][field]); stops += m.get('stop_reason') == stop
        out[label] = {'cap': cap, 'max_output_tokens': max(used), 'responses': len(used), 'stopped_at_cap': stops}
    return out


def decompose(rows, gold_listed, coverage=0.5):
    comb = {r['id'] for r in select(rows, coverage, 'combined')}
    sched = {r['id'] for r in select(rows, coverage, 'schedule')}
    groups = {'both': comb & sched, 'combined_only': comb - sched, 'schedule_only': sched - comb}
    by = {r['id']: r for r in rows}
    out = {}
    for name, ids in groups.items():
        cells = {}
        for r in (by[i] for i in sorted(ids)):
            key = f"{'agree' if r['agreement'] else 'disagree'}_{'listed' if r['listed'] else 'unlisted'}"
            c = cells.setdefault(key, {'n': 0, 'errors': 0, 'wrong_chapter': 0, 'gold_unlisted': 0})
            c['n'] += 1; c['errors'] += not r['correct']; c['wrong_chapter'] += r['wrong_chapter']
            c['gold_unlisted'] += not gold_listed[r['id']]
        out[name] = {'n': len(ids), 'errors': sum(c['errors'] for c in cells.values()),
                     'wrong_chapter': sum(c['wrong_chapter'] for c in cells.values()), 'cells': cells}
    au = [by[i] for i in comb if by[i]['agreement'] and not by[i]['listed']]
    k = sum(r['correct'] for r in au)
    out['combined_agree_unlisted'] = {'n': len(au), 'benchmark_correct': k,
                                      'correct_wilson95': wilson(k, len(au)) if au else None,
                                      'gold_unlisted': sum(not gold_listed[r['id']] for r in au)}
    return out


def main():
    base = Path(__file__).resolve().parents[1]; root = base/'results/prospective_v8a'
    hts = base/'data/hts_2026_revision_22.json'
    if file_hash(hts) != read(base/'results/main_v1/rq3/rq3_manifest.json')['hts_export']['file']:
        raise ValueError('Tariff snapshot changed')
    codes = {re.sub(r'\D', '', r.get('htsno', '')) for r in read(hts)}; codes = {c for c in codes if len(c) == 10}
    scored = Study(root).scored('test')
    gold_listed = {r['case_id']: r['gold'] in codes for r in scored}
    result = {'status': 'post_hoc_explanation_after_primary_result; frozen analysis unchanged',
              'inputs': {f'test/scored_{m}.jsonl': file_hash(root/'test'/f'scored_{m}.jsonl') for m in 'AB'},
              'source_sha256': file_hash(Path(__file__)),
              'paired_accuracy_by_depth': paired_depths(scored),
              'output_caps': output_caps(root),
              'gpt5_50pct_combined_vs_schedule': decompose(product_rows(scored, 'B', codes), gold_listed),
              'claude_50pct_combined_vs_schedule': decompose(product_rows(scored, 'A', codes), gold_listed)}
    out = base/'results/posthoc_v11'
    write(out/'explanations.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
