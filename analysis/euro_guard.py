"""Conservative EUR reservation ledger. Never refunds failed or unknown requests.
Safety depends on verified per-request price bounds covering input/output, FX and fees.
The frozen historical pipeline is unchanged; prospective generation injects this wrapper.
"""
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import json,sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from budget import BudgetExceeded


def units(value):
    d=Decimal(str(value))
    if not d.is_finite() or d<=0:raise ValueError('Positive finite EUR amount required')
    return int((d*1000000).to_integral_value(rounding=ROUND_CEILING))


class EuroLedger:
    def __init__(self,path,cap_eur=100):
        self.path=str(path);self.cap=units(cap_eur)
        if self.cap>100000000:raise ValueError('User ceiling is EUR 100')
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, cap INTEGER)')
            db.execute('INSERT OR IGNORE INTO settings VALUES (1,?)',(self.cap,))
            if db.execute('SELECT cap FROM settings WHERE id=1').fetchone()[0]!=self.cap:raise ValueError('Frozen monetary cap differs')
            db.execute('CREATE TABLE IF NOT EXISTS reservations (id INTEGER PRIMARY KEY, amount INTEGER, context TEXT)')
    def reserve(self,amount,context):
        amount=units(amount)
        with sqlite3.connect(self.path,timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            used=db.execute('SELECT COALESCE(SUM(amount),0) FROM reservations').fetchone()[0]
            if used+amount>self.cap:raise BudgetExceeded('EUR reservation ceiling reached')
            return db.execute('INSERT INTO reservations(amount,context) VALUES (?,?)',(amount,json.dumps(context))).lastrowid


class MonetaryClient:
    def __init__(self,client,ledger,bound):
        if client.model!=bound['model']:raise ValueError('Pricing model mismatch')
        units(bound['max_request_eur'])
        self.client,self.ledger,self.bound=client,ledger,bound
        self.model=self.name=client.model;self.last_metadata={}
    def complete(self,system,user,decoding):
        max_out=decoding.get('max_completion_tokens',decoding.get('max_tokens'))
        if not isinstance(max_out,int) or not 0<max_out<=self.bound['max_output_tokens']:
            raise ValueError('Output limit exceeds verified price bound')
        if len((system+user).encode('utf-8'))>self.bound['max_prompt_utf8_bytes']:
            raise ValueError('Prompt exceeds verified price bound')
        ticket=self.ledger.reserve(self.bound['max_request_eur'],{'model':self.model})
        result=self.client.complete(system,user,decoding)
        self.last_metadata={**getattr(self.client,'last_metadata',{}),'eur_reservation_ticket':ticket,
                            'reserved_eur_upper_bound':self.bound['max_request_eur']}
        return result
