"""Reconcile only recorded provider usage; preserve every original reservation."""
from pathlib import Path
import sys,sqlite3
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_io import lines,write,file_hash
from analysis.euro_guard_v2 import EuroLedger

def main():
 base=Path(__file__).resolve().parents[1];out=base/'results/prospective_budget';dbpath=out/'eur_budget.sqlite'
 snapshot=out/'before_v8a_settlement.sqlite'
 if not snapshot.exists():
  with sqlite3.connect(dbpath) as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
 ledger=EuroLedger(dbpath);inputs={};seen=set();count=0
 for label,model in [('A','claude-sonnet-5-5'),('B','gpt-5')]:
  p=base/f'results/prospective_v8/validation/generations_{label}.jsonl';inputs[str(p.relative_to(base))]=file_hash(p)
  for row in lines(p):
   for h in row['history']:
    m=h.get('metadata',{});ticket=m.get('eur_reservation_ticket')
    if ticket is None:continue
    if ticket in seen:raise ValueError('Duplicate usage ticket')
    seen.add(ticket);ledger.settle(ticket,model,m);count+=1
 report={'recorded_attempts_reconciled':count,'inputs_sha256':inputs,'before_snapshot_sha256':file_hash(snapshot),'conservative_usage_plus_unknown_reservations_eur':ledger.committed()/1e6,'accounting':'Official token rates times 1.5 EUR/USD fee/FX allowance; not actual invoice cost. Unknown charges retain original reservation. No reservation deleted.'}
 write(out/'pilot_reconciliation_v8a.json',report);print(report)
if __name__=='__main__':main()
