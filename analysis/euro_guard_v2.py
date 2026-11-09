"""Append-only, usage-settled budget reservations; unknown charges stay reserved."""
from decimal import Decimal, ROUND_CEILING
import json, sqlite3, hashlib
from analysis.euro_guard import EuroLedger as OriginalLedger, MonetaryClient as OriginalClient, units
from budget import BudgetExceeded


def usage_charge(model, metadata):
    u=metadata.get('usage',{})
    def count(key):
        v=u.get(key)
        if type(v) is not int or v<0:raise ValueError('Missing/invalid token usage: '+key)
        return v
    if not metadata.get('response_id'):raise ValueError('Missing response ID')
    resolved=metadata.get('resolved_model','')
    if model=='gpt-5':
        if resolved!='gpt-5' and not resolved.startswith('gpt-5-2025-'):raise ValueError('Unexpected resolved model')
        inp,out=count('prompt_tokens'),count('completion_tokens');rate=Decimal('1.25')
    elif model=='claude-sonnet-5-5':
        if resolved!=model:raise ValueError('Unexpected resolved model')
        inp,out=count('input_tokens'),count('output_tokens');rate=Decimal('2')
        if count('cache_creation_input_tokens') or count('cache_read_input_tokens'):
            raise ValueError('Caching not supported by frozen price bound')
    else:raise ValueError('Unknown pricing model')
    # Micro-EUR: token count * USD/million * conservative EUR/USD allowance.
    return int(((Decimal(inp)*rate+Decimal(out)*10)*Decimal('1.5')).to_integral_value(rounding=ROUND_CEILING))


class EuroLedger(OriginalLedger):
    def __init__(self,path,cap_eur=100):
        super().__init__(path,cap_eur)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS settlements (reservation_id INTEGER PRIMARY KEY, charge INTEGER NOT NULL, released INTEGER NOT NULL, response_id TEXT UNIQUE NOT NULL, evidence TEXT NOT NULL, sha256 TEXT NOT NULL)')
    def committed(self):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT (SELECT COALESCE(SUM(amount),0) FROM reservations)-(SELECT COALESCE(SUM(released),0) FROM settlements)').fetchone()[0]
    def reserve(self,amount,context):
        amount=units(amount)
        with sqlite3.connect(self.path,timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            used=db.execute('SELECT (SELECT COALESCE(SUM(amount),0) FROM reservations)-(SELECT COALESCE(SUM(released),0) FROM settlements)').fetchone()[0]
            if used+amount>self.cap:raise BudgetExceeded('EUR 100 shared commitment ceiling reached')
            return db.execute('INSERT INTO reservations(amount,context) VALUES (?,?)',(amount,json.dumps(context))).lastrowid
    def settle(self,ticket,model,metadata):
        if type(ticket) is not int or metadata.get('eur_reservation_ticket')!=ticket:raise ValueError('Reservation mismatch')
        charge=usage_charge(model,metadata)
        evidence=json.dumps({'model':model,'metadata':metadata},sort_keys=True,separators=(',',':'))
        digest=hashlib.sha256(evidence.encode()).hexdigest()
        with sqlite3.connect(self.path,timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT amount,context FROM reservations WHERE id=?',(ticket,)).fetchone()
            if not row or json.loads(row[1]).get('model')!=model:raise ValueError('Unknown reservation/model')
            if charge>row[0]:raise ValueError('Usage charge exceeds reserved bound')
            old=db.execute('SELECT sha256 FROM settlements WHERE reservation_id=?',(ticket,)).fetchone()
            if old:
                if old[0]!=digest:raise ValueError('Settlement evidence changed')
                return
            db.execute('INSERT INTO settlements VALUES (?,?,?,?,?,?)',(ticket,charge,row[0]-charge,metadata['response_id'],evidence,digest))


class MonetaryClient(OriginalClient):
    def complete(self,system,user,decoding):
        result=super().complete(system,user,decoding)
        try:self.ledger.settle(self.last_metadata['eur_reservation_ticket'],self.model,self.last_metadata)
        except ValueError as exc:
            # Unknown usage keeps full reservation. A violated cost bound must stop.
            if 'exceeds reserved bound' in str(exc):raise BudgetExceeded(str(exc)) from exc
            self.last_metadata['budget_settlement']='full reservation retained: '+str(exc)
        return result
