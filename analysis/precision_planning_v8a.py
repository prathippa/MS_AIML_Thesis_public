"""Conditional precision illustration from earlier validation data; no power claim."""
from pathlib import Path
import sys,json,re
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from analysis.product_routing import product_rows,select
from study_io import write,file_hash

def main():
 base=Path(__file__).resolve().parents[1]
 codes={re.sub(r'\D','',r.get('htsno','')) for r in json.load(open(base/'data/hts_2026_revision_22.json'))}
 result={}
 for name in ('main_v1','hscc_v1'):
  rows=product_rows(Study(base/'results'/name).scored('validation'),'B',codes)
  orders={p:np.array([rows.index(r) for r in select(rows,1,p)]) for p in ('confidence','combined')}
  errors=np.array([not r['correct'] for r in rows]);rng=np.random.default_rng(20261104);sizes={}
  for n in (400,700,900):
   differences=[]
   for _ in range(5000):
    weights=rng.multinomial(n,np.full(len(rows),1/len(rows)));risks={}
    for p,order in orders.items():
     counts=weights[order];before=np.cumsum(counts)-counts;take=np.minimum(counts,np.maximum(0,n//2-before));risks[p]=float(np.dot(take,errors[order])/take.sum())
    differences.append(risks['combined']-risks['confidence'])
   lo,hi=np.percentile(differences,[2.5,97.5]);sizes[n]={'conditional_bootstrap_interval_width':float(hi-lo),'approximate_half_width':float((hi-lo)/2),'sd':float(np.std(differences,ddof=1))}
  result[name]={'planning_products':len(rows),'sizes':sizes}
 write(base/'results/extension_v8/precision_planning_v8a.json',{'method':'5000 paired empirical product resamples at each proposed n; validation only, fixed policy ranking, seed 20261104. Conditional precision illustration, not power or guaranteed minimum detectable effect. Duplicated empirical support and distribution shift limit extrapolation.','results':result})
if __name__=='__main__':main()
