"""Retain unseen 400 evaluation products; append 300; reserve entire old development set."""
from pathlib import Path
import sys,json,hashlib
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_loader import load_atlas_messages
from study_io import lines,write,write_lines,bind,file_hash
from analysis.prepare_extension_data import to_message

def main():
 base=Path(__file__).resolve().parents[1];old=base/'data/prospective_candidate_v8';out=base/'data/prospective_candidate_v8a'
 # Refuse amendments after any old or new evaluation generation.
 for run in ('prospective_v8','prospective_v8a'):
  for p in (base/'results'/run/'test').glob('generations_*.jsonl'):
   if lines(p):raise ValueError('Evaluation has begun')
 dev=lines(old/'development_records.jsonl');evaluation=lines(old/'evaluation_records.jsonl')
 prior={r['description_hash']:r for r in dev+evaluation}
 for p in (base/'results').glob('*/**/records.jsonl'):
  for r in lines(p):prior[r['description_hash']]=r
 rows,_=load_atlas_messages(base/'data/atlas_candidate/train.jsonl','candidate','exclude',True)
 groups=defaultdict(list)
 for r in rows:groups[r['description_hash']].append(r)
 oldreason={r.get('reasoning','').strip() for r in prior.values() if r.get('reasoning','').strip()}
 candidates=[rs[0] for h,rs in sorted(groups.items()) if h not in prior and len({r['gold'] for r in rs})==1 and rs[0].get('reasoning','').strip() not in oldreason]
 oldrows=list(prior.values());vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=100000,dtype=np.float32)
 m=vec.fit_transform([r['description'] for r in oldrows+candidates]);a=m[len(oldrows):];b=m[:len(oldrows)]
 maxima=[]
 for i in range(0,len(candidates),256):maxima.extend((a[i:i+256]@b.T).max(axis=1).toarray().ravel().tolist())
 eligible=[i for i,v in enumerate(maxima) if v<.9];np.random.default_rng(20261104).shuffle(eligible)
 picked=[]
 for i in eligible:
  if picked and float((a[i]@a[picked].T).max())>=.9:continue
  picked.append(i)
  if len(picked)==300:break
 if len(picked)!=300:raise ValueError('Insufficient candidates')
 selecteddev=dev[:100]
 # Known token-cap failures first for technical verification; same first-100 membership.
 failed={r['case_id'].split(':',1)[-1] for r in lines(base/'results/prospective_v8/validation/generations_B.jsonl') if r['status']!='ok'}
 selecteddev.sort(key=lambda r:str(r['source_row_id']) not in failed)
 evaluation=evaluation+[candidates[i] for i in picked]
 assert not {r['description_hash'] for r in dev}&{r['description_hash'] for r in evaluation}
 meta={'date':'2026-11-04','seed':20261104,'development_n':100,'evaluation_n':700,'retained_unseen_evaluation_n':400,'added_evaluation_n':300,'all_original_development_products_excluded_from_evaluation':200,'selection':'100 first original development products; known cap failures first. Original unseen 400 evaluation plus 300 seeded draws after exact-description, conflicting-gold, exact-reasoning and char_wb TFIDF(3,5) cosine>=.90 exclusions against prior records and all original 600 cohort products; greedy near-duplicate exclusion within additions. No model correctness used.','source_sha256':file_hash(base/'data/atlas_candidate/train.jsonl'),'selection_script_sha256':file_hash(Path(__file__)),'limitations':'Heuristic duplicate filtering does not prove independent rulings or absence of model-training exposure; gold release mismatch retained.'}
 bind(out/'selection_protocol.json',meta)
 for name,batch in [('development',selecteddev),('evaluation',evaluation)]:
  write_lines(out/f'{name}_records.jsonl',batch);write_lines(out/f'{name}.jsonl',[to_message(r) for r in batch])
  loaded,_=load_atlas_messages(out/f'{name}.jsonl',name,'exclude',True)
  assert len(loaded)==len(batch) and [r['gold'] for r in loaded]==[r['gold'] for r in batch]
 write(out/'files.sha256.json',{p.name:file_hash(p) for p in out.iterdir() if p.name!='files.sha256.json'})
 print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
