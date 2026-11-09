"""Single pre-specified primary comparison plus explicitly secondary policy tables."""
from pathlib import Path
import json,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from study_io import read,write,file_hash
from analysis.product_routing import product_rows,compare,POLICIES


def analyse(root,plan_path):
    plan=read(plan_path);frozen=read(root/'extension_protocol.json')['contents']['plan']
    if plan!=frozen:raise ValueError('Frozen prospective plan changed')
    study=Study(root);base=Path(__file__).resolve().parents[1]
    hts=base/'data/hts_2026_revision_22.json'
    if file_hash(hts)!=read(base/'results/main_v1/rq3/rq3_manifest.json')['hts_export']['file']:
        raise ValueError('Tariff snapshot changed')
    codes={re.sub(r'\D','',r.get('htsno','')) for r in read(hts)};codes={c for c in codes if len(c)==10}
    data=study.scored('test');output={m:compare(product_rows(data,m,codes),POLICIES,n_boot=2000,seed=7) for m in ('A','B')}
    primary=plan['primary_comparison'];r=next(r for r in output['B']['comparisons'] if r['policy']=='combined' and r['requested_coverage']==.5)
    out=root/'product_analysis'
    write(out/'results.json',output)
    write(out/'primary.json',{'specification':primary,'observed':r,'supports_directional_hypothesis':r['paired_ci95'][1]<0,
                             'scope':'one prospective primary comparison; other intervals secondary/descriptive'})
    lines=['# Prospective product-level analysis','','One pre-specified primary comparison; additional rows are descriptive.','',
           '| Designated | Coverage | Policy | Risk | Difference [95% CI] |','|---|---:|---|---:|---|']
    for m,result in output.items():
        for row in result['comparisons']:
            lo,hi=row['paired_ci95'];lines.append(f'| {m} | {row["coverage"]:.1%} | {row["policy"]} | {row["selective_risk"]:.3f} | {row["risk_minus_confidence"]:+.3f} [{lo:+.3f}, {hi:+.3f}] |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    write(out/'input_output_hashes.json',{'inputs':{f'{m}':file_hash(root/'test'/f'scored_{m}.jsonl') for m in ('A','B')},
                                       'outputs':{f:file_hash(out/f) for f in ('results.json','primary.json','summary.md')}})
