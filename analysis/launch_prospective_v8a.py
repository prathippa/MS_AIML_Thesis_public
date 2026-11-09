"""Run the frozen new study, with external credentials and a shared EUR ledger.
Two workers (one per model); no concurrent writes to a prediction cache.
"""
from pathlib import Path
import sys,os,json,sqlite3,threading,time
from concurrent.futures import ThreadPoolExecutor
from dotenv import dotenv_values
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from study_io import read,write,bind,file_hash,lines,lock,utcnow
from budget import BudgetExceeded
from analysis.euro_guard_v2 import EuroLedger,MonetaryClient
ROOT=BASE/'results/prospective_v8a'
LEDGER=BASE/'results/prospective_budget/eur_budget.sqlite'
STOP=threading.Event()


def status(state, detail=None):
    counts={}
    for split in ('validation','test'):
        for label in ('A','B'):
            try:
                rows=lines(ROOT/split/f'generations_{label}.jsonl')
                counts[f'{split}/{label}']={'completed':len(rows),'failed':sum(r['status']!='ok' for r in rows)}
            except (ValueError,OSError):pass
    reserved=0
    if LEDGER.exists():
        with sqlite3.connect(LEDGER) as db:
            reserved=EuroLedger(LEDGER).committed()/1000000
    write(ROOT/'run_status.json',{'state':state,'detail':detail,'updated_at':utcnow(),'pid':os.getpid(),
                                'counts':counts,'conservative_usage_plus_outstanding_reservations_eur':reserved,'budget_eur':100,
                                'actual_invoice_cost':'not yet reconciled; reservations are conservative upper bounds'})


class TechnicalStop(BudgetExceeded):
    pass


class CircuitClient(MonetaryClient):
    def __init__(self,*args):super().__init__(*args);self.errors=0
    def complete(self,system,user,decoding):
        if STOP.is_set():raise TechnicalStop('Technical circuit breaker; not budget exhaustion')
        try:
            result=super().complete(system,user,decoding)
        except BudgetExceeded:
            STOP.set();raise
        except Exception:
            self.errors+=1
            if self.errors>=3:STOP.set()
            raise
        from generation import parse_response
        try:parse_response(result);self.errors=0
        except ValueError:
            self.errors+=1
            if self.errors>=3:STOP.set()
        return result


def main():
    plan=read(BASE/'prospective_plan_v8a.json')
    if plan['status']!='frozen' or not plan['verified_price_table']:raise ValueError('Protocol/prices not frozen')
    values=dotenv_values(plan['credentials_file_external'])
    for name in ('ANTHROPIC_API_KEY','OPENAI_API_KEY'):
        if not values.get(name):raise ValueError('Missing required external credential')
        os.environ[name]=values[name]
    os.environ.pop('OPENAI_BASE_URL',None)
    for label in ('A','B'):os.environ[f'HTS_MODEL_{label}']=plan['models'][label]
    os.environ['HTS_A_PROVIDER']='anthropic';os.environ['HTS_B_PROVIDER']='openai'
    from generation import get_client,build_messages
    import config
    from pipeline import Study,init
    for f,h in plan['candidate_files'].items():
        if file_hash(BASE/f)!=h:raise ValueError('Candidate cohort changed')
    config.DECODING_BY_PROVIDER['anthropic']={'max_tokens':plan['verified_price_table']['A']['max_output_tokens']}
    config.DECODING_BY_PROVIDER['openai']={'max_completion_tokens':plan['verified_price_table']['B']['max_output_tokens']}
    with lock(ROOT):
        bind(ROOT/'extension_protocol.json',{'plan':plan,'sources':{p.name:file_hash(p) for p in
            [Path(__file__),BASE/'analysis/euro_guard.py',BASE/'analysis/euro_guard_v2.py',BASE/'analysis/product_routing.py',BASE/'analysis/analyse_prospective.py']}})
        if not (ROOT/'protocol.json').exists():
            init(ROOT,'ATLAS-training-mirror@0a66ad345d3fc9828f362836d86b31e20798513c;cohort-v8a',evidence_mode='none',max_calls=plan['max_requests'])
        study=Study(ROOT)
        if not (ROOT/'validation/input.manifest.json').exists():
            study.prepare(BASE/'data/prospective_candidate_v8a/development.jsonl',BASE/'data/prospective_candidate_v8a/evaluation.jsonl')
            study.evidence()
        for split in ('validation','test'):
            for row in study.records(split):
                sy,us=build_messages({'description':row['description']},[])
                if len((sy+us).encode())>2048:raise ValueError('Prompt exceeds pricing bound')
        ledger=EuroLedger(LEDGER,plan['budget_eur'])
        finished=threading.Event()
        def monitor():
            while not finished.wait(10):status('running')
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        try:
            status('running','Generation started; validation precedes evaluation')
            def generate(split,label):
                client=CircuitClient(get_client(study.protocol['models'][label]),ledger,plan['verified_price_table'][label])
                return Study(ROOT).generate(split,label,client=client)
            for split in ('validation','test'):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(generate,split,label) for label in ('A','B')]
                    for f in futures:f.result()
                if STOP.is_set():raise RuntimeError('Technical circuit breaker stopped the study')
                for label in ('A','B'):
                    rows=lines(ROOT/split/f'generations_{label}.jsonl')
                    if len(rows)!=len(study.records(split)):raise RuntimeError('Incomplete generation')
                    if sum(r['status']!='ok' for r in rows)/len(rows)>.10:
                        raise RuntimeError('More than 10 percent failed outputs; stop for technical review without changing evaluation policy')
                print(split,'generation complete',flush=True)
            study.metrics();study.calibrate()
            from analysis.analyse_prospective import analyse
            analyse(ROOT,BASE/'prospective_plan_v8a.json')
            finished.set();watcher.join();status('complete')
            print('Study and frozen product-level analysis complete',flush=True)
        except Exception as exc:
            STOP.set();finished.set();watcher.join();status('stopped',type(exc).__name__+': '+str(exc)[:180])
            print('Stopped:',type(exc).__name__,flush=True)
            raise SystemExit(1)

if __name__=='__main__':main()
