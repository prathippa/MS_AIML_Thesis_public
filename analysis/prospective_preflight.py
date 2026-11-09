"""Readiness check only. Never sends a request or prints credentials."""
from pathlib import Path
import json,os,sys,sqlite3
from dotenv import dotenv_values
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_io import read,file_hash,write

def main():
    base=Path(__file__).resolve().parents[1];plan=read(base/'prospective_plan.json')
    cohort=base/'data/prospective_candidate_v8';hashes=read(cohort/'files.sha256.json')
    invalid=[n for n,h in hashes.items() if file_hash(cohort/n)!=h]
    blocked=[]
    if invalid:blocked.append('Candidate files fail hash verification')
    external=dotenv_values(plan['credentials_file_external']) if plan.get('credentials_file_external') else {}
    for name in ['ANTHROPIC_API_KEY','OPENAI_API_KEY']:
        if not (os.getenv(name) or external.get(name)):blocked.append(name+' not configured in process')
    if plan['status']!='frozen':blocked.append('Prospective protocol is draft: primary comparison, precision and sample size not finalised')
    if plan['verified_price_table'] is None:blocked.append('Provider pricing and conservative per-request EUR bounds not supplied')
    if not plan['currency_guard_integrated']:blocked.append('Original pipeline enforces request count only; EUR reservation guard must be integrated before paid execution')
    reserved=0
    ledger=base/'results/prospective_budget/eur_budget.sqlite'
    if ledger.exists():
        with sqlite3.connect(ledger) as db:reserved=db.execute('SELECT COALESCE(SUM(amount),0) FROM reservations').fetchone()[0]/1000000
    report={'ready_for_paid_run':not blocked,'blockers':blocked,'budget_eur':100,'reserved_eur_upper_bound':reserved,'actual_invoice_cost_eur':None,'model_calls_made_by_this_check':0}
    write(base/'results/extension_v8/paid_preflight.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
