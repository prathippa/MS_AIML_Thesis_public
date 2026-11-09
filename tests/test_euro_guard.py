import pytest
from budget import BudgetExceeded
from analysis.euro_guard import EuroLedger,MonetaryClient


def test_persistent_ceiling(tmp_path):
    path=tmp_path/'euro.sqlite';b=EuroLedger(path,1);b.reserve('.6',{})
    with pytest.raises(BudgetExceeded):EuroLedger(path,1).reserve('.5',{})


def test_cap_cannot_increase(tmp_path):
    path=tmp_path/'euro.sqlite';EuroLedger(path,1)
    with pytest.raises(ValueError):EuroLedger(path,2)
    with pytest.raises(ValueError):EuroLedger(tmp_path/'other',101)


class Broken:
    model='fake'
    def complete(self,*args):raise RuntimeError('network error')


def wrapper(tmp_path):
    b=EuroLedger(tmp_path/'ledger',1)
    c=MonetaryClient(Broken(),b,{'model':'fake','max_request_eur':1,'max_output_tokens':10,'max_prompt_utf8_bytes':20})
    return b,c


def test_failed_call_reservation_not_refunded(tmp_path):
    b,c=wrapper(tmp_path)
    with pytest.raises(RuntimeError):c.complete('s','u',{'max_tokens':10})
    with pytest.raises(BudgetExceeded):b.reserve('.01',{})


def test_oversized_prompt_rejected_before_reservation(tmp_path):
    b,c=wrapper(tmp_path)
    with pytest.raises(ValueError):c.complete('x'*21,'',{'max_tokens':10})
    b.reserve(1,{})


def test_oversized_output_rejected(tmp_path):
    b,c=wrapper(tmp_path)
    with pytest.raises(ValueError):c.complete('s','u',{'max_tokens':11})
    b.reserve(1,{})
