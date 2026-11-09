import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from analysis.euro_guard_v2 import EuroLedger,usage_charge
from budget import BudgetExceeded

def meta(t=1):
 return {'response_id':f'r{t}','resolved_model':'gpt-5-2025-08-07','usage':{'prompt_tokens':200,'completion_tokens':8000},'eur_reservation_ticket':t}
def test_reasoning_included_and_settlement_idempotent(tmp_path):
 l=EuroLedger(tmp_path/'b.db');t=l.reserve('.13',{'model':'gpt-5'});m=meta(t)
 assert usage_charge('gpt-5',m)==120375
 l.settle(t,'gpt-5',m);l.settle(t,'gpt-5',m)
 assert l.committed()==120375
 m['usage']['completion_tokens']=1
 with pytest.raises(ValueError):l.settle(t,'gpt-5',m)
def test_unknown_and_overbound_never_release(tmp_path):
 l=EuroLedger(tmp_path/'b.db');t=l.reserve('.075',{'model':'gpt-5'})
 with pytest.raises(ValueError):l.settle(t,'gpt-5',meta(t))
 assert l.committed()==75000
 m=meta(t);del m['usage']['completion_tokens']
 with pytest.raises(ValueError):l.settle(t,'gpt-5',m)
 assert l.committed()==75000
def test_concurrent_cap(tmp_path):
 l=EuroLedger(tmp_path/'b.db',1)
 def reserve(_):
  try:l.reserve('.13',{'model':'gpt-5'});return 1
  except BudgetExceeded:return 0
 with ThreadPoolExecutor(max_workers=8) as pool:assert sum(pool.map(reserve,range(20)))==7
 assert l.committed()==910000
def test_wrong_model_and_duplicate_response(tmp_path):
 l=EuroLedger(tmp_path/'b.db');a=l.reserve('.13',{'model':'gpt-5'});b=l.reserve('.13',{'model':'gpt-5'})
 with pytest.raises(ValueError):l.settle(a,'claude-sonnet-5-5',meta(a))
 l.settle(a,'gpt-5',meta(a));m=meta(b);m['response_id']='r1'
 with pytest.raises(sqlite3.IntegrityError):l.settle(b,'gpt-5',m)
 assert l.committed()==250375
