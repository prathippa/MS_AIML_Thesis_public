"""Offline independent reranking/bootstrap and provenance audit. No API calls."""
from pathlib import Path
import sys,json,hashlib,sqlite3
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from study_io import read,lines,verify_bound,file_hash,write
from analysis.euro_guard_v2 import usage_charge

def order(rows,other,policy,codes):
 def key(i):
  r=rows[i];parts=[]
  if policy in ('agreement','combined'):parts.append(-int(other[i]['status']=='ok' and r['pred']==other[i]['pred']))
  if policy in ('schedule','combined'):parts.append(-int(r['pred'] in codes))
  return tuple(parts)+(-r['confidence'],hashlib.sha256(r['case_id'].encode()).hexdigest())
 return np.array(sorted((i for i,r in enumerate(rows) if r['status']=='ok'),key=key),dtype=int)

def risk(weights,ranking,errors,k):
 counts=weights[ranking];before=np.cumsum(counts)-counts;take=np.minimum(counts,np.maximum(0,k-before))
 return float(np.dot(take,errors[ranking])/take.sum()) if take.sum() else None

def main():
 if not __debug__:raise RuntimeError("Run this audit without Python -O: checks must remain enabled")
 base=Path(__file__).resolve().parents[1];root=base/'results/prospective_v8a';plan=read(base/'prospective_plan_v8a.json');ext=verify_bound(root/'extension_protocol.json')
 assert ext['plan']==plan
 for f,h in ext['sources'].items():assert file_hash(base/'analysis'/f)==h
 for f,h in plan['candidate_files'].items():assert file_hash(base/f)==h
 status=read(root/'run_status.json');assert status['state']=='complete'
 for name in ('main_v1','oss_v1','hscc_v1','prospective_v8a'):
  s=Study(base/'results'/name)
  for split in ('validation','test'):s.scored(split)
 study=Study(root);tickets=[]
 for split,n in [('validation',100),('test',700)]:
  assert len(study.records(split))==n
  for label in 'AB':
   gs=lines(root/split/f'generations_{label}.jsonl')
   assert len(gs)==n and len({g['case_id'] for g in gs})==n and all(g['status']=='ok' for g in gs)
   assert all(g['decoding']==({'max_tokens':4000} if label=='A' else {'max_completion_tokens':8000}) for g in gs)
   tickets.extend(h['metadata']['eur_reservation_ticket'] for g in gs for h in g['history'])
 old=lines(base/'data/prospective_candidate_v8/development_records.jsonl')
 assert not {r['description_hash'] for r in old}&{r['description_hash'] for r in study.records('test')}
 manifest=read(root/'product_analysis/input_output_hashes.json')
 for m,h in manifest['inputs'].items():assert file_hash(root/'test'/f'scored_{m}.jsonl')==h
 for f,h in manifest['outputs'].items():assert file_hash(root/'product_analysis'/f)==h
 hts=base/'data/hts_2026_revision_22.json';assert file_hash(hts)==read(base/'results/main_v1/rq3/rq3_manifest.json')['hts_export']['file']
 codes={''.join(c for c in r.get('htsno','') if c.isdigit()) for r in read(hts)};codes={c for c in codes if len(c)==10}
 data=study.scored('test');saved=read(root/'product_analysis/results.json');result={};policies=('confidence','agreement','schedule','combined')
 for label in 'AB':
  rows=sorted([r for r in data if r['model']==label],key=lambda r:r['case_id']);other=sorted([r for r in data if r['model']!=label],key=lambda r:r['case_id'])
  assert all(a['case_id']==b['case_id'] and a['gold']==b['gold'] and a['description_hash']==b['description_hash'] for a,b in zip(rows,other))
  errors=np.array([r['pred']!=r['gold'] for r in rows]);assert all(bool(e)==(not r['correct']) for e,r in zip(errors,rows))
  orders={p:order(rows,other,p,codes) for p in policies};comparisons=[]
  for coverage in (.3,.5,.7):
   k=int(np.floor(len(rows)*coverage+.5));rng=np.random.default_rng(7);gaps={p:[] for p in policies}
   for _ in range(2000):
    weights=np.bincount(rng.integers(0,len(rows),len(rows)),minlength=len(rows));baseline=risk(weights,orders['confidence'],errors,k)
    for p in policies:gaps[p].append(risk(weights,orders[p],errors,k)-baseline)
   baseline=float(errors[orders['confidence'][:k]].mean())
   for p in policies:
    selected=orders[p][:k];r=float(errors[selected].mean());ci=np.percentile(gaps[p],[2.5,97.5]);v=next(v for v in saved[label]['comparisons'] if v['policy']==p and v['requested_coverage']==coverage)
    assert np.allclose([r,r-baseline,*ci],[v['selective_risk'],v['risk_minus_confidence'],*v['paired_ci95']],rtol=0,atol=1e-12)
    wrong=sum(rows[i]['pred'][:2]!=rows[i]['gold'][:2] for i in selected);assert abs(wrong/k-v['wrong_chapter_risk'])<1e-12
    comparisons.append({'policy':p,'coverage':coverage,'accepted':k,'errors':int(errors[selected].sum()),'risk':r,'difference':r-baseline,'ci95':ci.tolist(),'wrong_chapter_count':wrong})
  a=set(orders['schedule'][:350]);b=set(orders['combined'][:350])
  result[label]={'n':700,'correct':int((~errors).sum()),'accuracy':float((~errors).mean()),'comparisons':comparisons,'schedule_combined_overlap_at50':len(a&b),'schedule_only_errors':int(errors[list(a-b)].sum()),'combined_only_errors':int(errors[list(b-a)].sum())}
 spec=plan['primary_comparison'];primary=read(root/'product_analysis/primary.json');assert primary['specification']==spec
 match=next(v for v in saved[spec['designated_model']]['comparisons'] if v['policy']==spec['policy'] and v['requested_coverage']==spec['coverage'])
 assert primary['observed']==match and primary['supports_directional_hypothesis']==(match['paired_ci95'][1]<0)
 with sqlite3.connect('file:'+str(base/'results/prospective_budget/eur_budget.sqlite')+'?mode=ro',uri=True) as db:
  reservations={i:(a,json.loads(c)) for i,a,c in db.execute('SELECT id,amount,context FROM reservations')};settlements=list(db.execute('SELECT reservation_id,charge,released,response_id,evidence,sha256 FROM settlements'))
 for ticket,charge,released,response,evidence,digest in settlements:
  assert hashlib.sha256(evidence.encode()).hexdigest()==digest
  e=json.loads(evidence);assert charge==usage_charge(e['model'],e['metadata'])
  assert reservations[ticket][0]==charge+released and reservations[ticket][1]['model']==e['model']
  assert response==e['metadata']['response_id'] and ticket==e['metadata']['eur_reservation_ticket']
 used=sum(v[0] for v in reservations.values())-sum(s[2] for s in settlements)
 assert used<=100000000 and abs(used/1e6-status['conservative_usage_plus_outstanding_reservations_eur'])<1e-9
 assert len(tickets)==1600 and len(set(tickets))==1600 and set(tickets)<=set(reservations)
 result['audit']={'historical_sources_and_stages':'passed for all four studies','paired_comparisons_independently_reproduced':24,'new_generations':1600,'new_attempts':len(tickets),'old_development_evaluation_overlap':0,'shared_ledger_attempts':len(reservations),'settlements_verified':len(settlements),'conservative_accounted_eur':used/1e6,'actual_invoice':'not reconciled','finished_at':status['updated_at'],'scope':'Independent reranking and weighted-count bootstrap of fixed outputs; not independent data or expert validation.'}
 write(base/'results/verification_v10/audit.json',result);print(json.dumps(result['audit'],indent=2))
if __name__=='__main__':main()
