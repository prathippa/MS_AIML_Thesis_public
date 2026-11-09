"""Prepare candidate cohorts without model calls or inspecting new model outcomes.
Near-duplicate filtering is heuristic; benchmark training exposure cannot be ruled out.
"""
from pathlib import Path
import sys,json,hashlib
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_loader import load_atlas_messages
from study_io import lines,write,write_lines,bind,file_hash


def to_message(r):
    return {"id":r["source_row_id"],"messages":[{"role":"user","content":r["description"]},{"role":"assistant","content":r["gold"]+" Reasoning -> "+r["reasoning"]}]}


def main():
    base=Path(__file__).resolve().parents[1]
    source=base/'data/atlas_candidate/train.jsonl'
    rows,audit=load_atlas_messages(source,'candidate','exclude',True)
    prior={}
    for f in (base/'results').glob('*/**/records.jsonl'):
        for r in lines(f):prior[r['description_hash']]=r
    groups=defaultdict(list)
    for r in rows:groups[r['description_hash']].append(r)
    conflicts={h for h,rs in groups.items() if len({r['gold'] for r in rs})>1}
    old_reason={hashlib.sha256(r['reasoning'].strip().encode()).hexdigest() for r in prior.values() if r.get('reasoning','').strip()}
    candidates=[];same_reason=0
    for h,rs in sorted(groups.items()):
        if h in prior or h in conflicts:continue
        r=rs[0]
        if r.get('reasoning','').strip() and hashlib.sha256(r['reasoning'].strip().encode()).hexdigest() in old_reason:
            same_reason+=1;continue
        candidates.append(r)
    old=list(prior.values())
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=100000,dtype=np.float32)
    matrix=vec.fit_transform([r['description'] for r in old+candidates])
    mat=matrix[len(old):];oldmat=matrix[:len(old)]
    maxima=[]
    for i in range(0,len(candidates),256):
        maxima.extend((mat[i:i+256] @ oldmat.T).max(axis=1).toarray().ravel().tolist())
    eligible=[i for i,x in enumerate(maxima) if x < .90]
    rng=np.random.default_rng(20261104);rng.shuffle(eligible)
    selected=[]
    for i in eligible:
        if selected and float((mat[i] @ mat[selected].T).max()) >= .90:continue
        selected.append(i)
        if len(selected)==600:break
    chosen=[candidates[i] for i in selected]
    if len(chosen)<600:raise ValueError('Insufficient candidates')
    out=base/'data/prospective_candidate_v8';out.mkdir(parents=True,exist_ok=True)
    meta={'status':'candidate_cohort_no_model_calls; prospective_protocol_not_yet_final',
          'source_sha256':file_hash(source),'source_revision':'0a66ad345d3fc9828f362836d86b31e20798513c',
          'seed':20261104,'single_code_rows':len(rows),'distinct_descriptions':len(groups),
          'conflicting_gold_description_groups_excluded':len(conflicts),
          'exact_description_groups_overlapping_prior':sum(h in prior for h in groups),
          'exact_reasoning_matches_excluded':same_reason,'near_prior_matches_excluded':sum(x>=.90 for x in maxima),
          'near_duplicate_rule':'char_wb TFIDF (3,5), max_features=100000, cosine >= 0.90',
          'development_n':200,'evaluation_n':400,
          'limits':['Source-ruling IDs absent; heuristic deduplication cannot prove ruling independence',
                    'Public benchmark training split; model-training exposure unknown',
                    'Historical golds may differ from current tariff release',
                    'Cohort size is provisional, not justified as adequate power'],
          'analysis_source_sha256':file_hash(Path(__file__))}
    bind(out/'selection_protocol.json',meta)
    codes={''.join(c for c in r.get('htsno','') if c.isdigit()) for r in json.load(open(base/'data/hts_2026_revision_22.json'))}
    for name,batch in [('development',chosen[:200]),('evaluation',chosen[200:])]:
        write_lines(out/f'{name}_records.jsonl',batch)
        write_lines(out/f'{name}.jsonl',[to_message(r) for r in batch])
        meta[name+'_gold_listed_count']=sum(r['gold'] in codes for r in batch)
    write(out/'audit.json',meta)
    write(out/'files.sha256.json',{p.name:file_hash(p) for p in sorted(out.iterdir()) if p.name!='files.sha256.json'})
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
