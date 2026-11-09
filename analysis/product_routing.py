"""Exploratory product-level policy comparison; offline, no API calls.

One designated-model recommendation per product. Both designations are reported.
Failed designated outputs are always referred; failed secondary outputs cannot agree.
Schedule policies require the original release file with its recorded SHA-256.
"""
from pathlib import Path
import argparse, hashlib, json, math, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from study_io import bind, write, read, file_hash

COVERAGES = (0.3, 0.5, 0.7)
POLICIES = ('confidence', 'agreement', 'schedule', 'combined')


def product_rows(rows, designated, codes=None):
    groups = {}
    for r in rows:
        models = groups.setdefault(r['case_id'], {})
        if r['model'] in models:
            raise ValueError('Duplicate product/model record')
        models[r['model']] = r
    out = []
    for cid, models in sorted(groups.items()):
        if set(models) != {'A', 'B'}:
            raise ValueError('Both model records required, including failures')
        r, other = models[designated], models['B' if designated == 'A' else 'A']
        if r['description_hash'] != other['description_hash'] or r['gold'] != other['gold']:
            raise ValueError('Product alignment mismatch')
        ok = r['status'] == 'ok'
        out.append({'id': cid, 'eligible': ok,
                    'confidence': r['confidence'] if ok else 0.0,
                    'agreement': bool(ok and other['status'] == 'ok' and r['pred'] == other['pred']),
                    'listed': r['pred'] in codes if ok and codes is not None else None,
                    'correct': bool(r['correct']),
                    'wrong_chapter': ok and r['severity_tier'] == 'wrong_chapter'})
    return out


def select(rows, coverage, policy):
    if not 0 < coverage <= 1 or policy not in POLICIES:
        raise ValueError('Invalid coverage/policy')
    if policy in ('schedule', 'combined') and any(r['listed'] is None for r in rows if r['eligible']):
        raise ValueError('Verified schedule membership required')
    def key(r):
        parts = []
        if policy in ('agreement', 'combined'):
            parts.append(-int(r['agreement']))
        if policy in ('schedule', 'combined'):
            parts.append(-int(r['listed']))
        return (*parts, -r['confidence'], hashlib.sha256(r['id'].encode()).hexdigest())
    k = max(1, math.floor(len(rows) * coverage + 0.5)) if rows else 0
    return sorted((r for r in rows if r['eligible']), key=key)[:k]


def metrics(rows, selected):
    n, k = len(rows), len(selected)
    return {'n_products': n, 'n_streamlined': k, 'n_referred': n-k,
            'coverage': k/n if n else 0,
            'selective_risk': sum(not r['correct'] for r in selected)/k if k else None,
            'wrong_chapter_risk': sum(r['wrong_chapter'] for r in selected)/k if k else None}


def compare(rows, policies, n_boot=2000, seed=7):
    if not rows or n_boot < 1:
        raise ValueError('Nonempty products and positive bootstrap count required')
    result = []
    for coverage in COVERAGES:
        points = {p: metrics(rows, select(rows, coverage, p)) for p in policies}
        gaps = {p: [] for p in policies}
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            batch = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
            base = metrics(batch, select(batch, coverage, 'confidence'))['selective_risk']
            for p in policies:
                risk = metrics(batch, select(batch, coverage, p))['selective_risk']
                if base is not None and risk is not None:
                    gaps[p].append(risk-base)
        for p in policies:
            risk, base = points[p]['selective_risk'], points['confidence']['selective_risk']
            result.append({'policy': p, 'requested_coverage': coverage, **points[p],
                           'risk_minus_confidence': risk-base if risk is not None and base is not None else None,
                           'paired_ci95': [float(v) for v in np.percentile(gaps[p], [2.5, 97.5])] if gaps[p] else None})
    curves = {p: [{'requested_coverage': float(c), **metrics(rows, select(rows, float(c), p))}
                  for c in np.linspace(.1, 1, 19)] for p in policies}
    return {'comparisons': result, 'curves': curves}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hts', type=Path)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    codes = None
    expected = read(args.results/'main_v1/rq3/rq3_manifest.json')['hts_export']['file']
    if args.hts:
        if file_hash(args.hts) != expected:
            raise ValueError('HTS file differs from original audit; use a separate release-sensitivity protocol')
        import re
        codes = {re.sub(r'\D', '', r.get('htsno', '')) for r in read(args.hts)}
        codes = {c for c in codes if len(c)==10}
        if not codes:
            raise ValueError('No ten-digit schedule lines')
    studies = ('main_v1', 'hscc_v1')
    policies = POLICIES if codes is not None else POLICIES[:2]
    inputs = {}; datasets = {}
    for name in studies:
        study = Study(args.results/name)
        for split in ('validation', 'test'):
            datasets[name, split] = study.scored(split)
            for m in ('A','B'):
                f = f'{name}/{split}/scored_{m}.jsonl'; inputs[f] = file_hash(args.results/f)
    protocol = {'analysis_status': 'post_hoc_exploratory_existing_seen_results',
                'unit': 'product', 'designations': ['A','B'], 'policies': list(policies),
                'coverage_points': list(COVERAGES), 'bootstrap': 'paired_product_percentile',
                'n_boot': args.n_boot, 'seed': 7, 'failure_policy': 'refer designated failures; secondary failures cannot agree',
                'tie_rule': 'SHA256 product ID; same for every policy',
                'multiplicity': 'pointwise descriptive intervals, no confirmatory significance declarations',
                'schedule_sha256': expected if codes is not None else None,
                'schedule_policy_status': 'available' if codes is not None else 'not_run_missing_verified_snapshot',
                'api_calls': 0, 'new_api_cost_eur': 0, 'source_sha256': file_hash(Path(__file__)), 'inputs': inputs}
    bind(args.out/'protocol.json', protocol)  # refuses changed code/config/inputs in same run
    output = {}
    for (name, split), rows in datasets.items():
        for m in ('A','B'):
            data = product_rows(rows, m, codes)
            output[f'{name}/{split}/{m}'] = compare(data, policies, args.n_boot)
    write(args.out/'results.json', output)
    lines = ['# Product-level routing extension', '',
             'Post-hoc exploratory; existing test outcomes were previously inspected. No new model calls.', '',
             'Negative differences favour the alternative. Intervals are paired product-bootstrap, pointwise and unadjusted.', '',
             f'Schedule policies: {protocol["schedule_policy_status"]}.', '',
             '| Study/split/designation | Coverage | Policy | Risk | Difference vs confidence [95% CI] |',
             '|---|---:|---|---:|---|']
    for label, value in output.items():
        for r in value['comparisons']:
            lo,hi = r['paired_ci95']
            lines.append(f'| {label} | {r["coverage"]:.1%} | {r["policy"]} | {r["selective_risk"]:.3f} | {r["risk_minus_confidence"]:+.3f} [{lo:+.3f}, {hi:+.3f}] |')
    lines += ['', 'Two-model agreement requires a second generation per product in deployment. This run reuses cached calls; zero new spend is not zero deployment cost.',
              'No claim is made about reviewer correction accuracy, review minutes, or legal/financial loss.']
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')
    write(args.out/'outputs.sha256.json', {f:file_hash(args.out/f) for f in ('protocol.json','results.json','summary.md')})
    print(args.out/'summary.md')

if __name__ == '__main__': main()
