"""Paid generation entry point; fails closed unless frozen protocol and prices exist."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import Study
from generation import get_client
from study_io import read,bind,lock
from analysis.euro_guard import EuroLedger,MonetaryClient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--model',choices=['A','B'],required=True);p.add_argument('--split',choices=['validation','test'],required=True)
    a=p.parse_args();plan=read(a.plan)
    if plan.get('status')!='frozen' or not plan.get('verified_price_table'):
        raise ValueError('Freeze protocol and supply verified conservative price bounds before paid generation')
    if plan['budget_eur']>100:raise ValueError('User limit is EUR 100')
    if a.root.resolve().name in ('main_v1','oss_v1','hscc_v1'):raise ValueError('Use a new prospective study directory')
    with lock(a.root):
        study=Study(a.root);config=study.protocol['models'][a.model]
        bound=plan['verified_price_table'][a.model]
        if not bound.get('source_url') or not bound.get('verified_date'):
            raise ValueError('Pricing provenance is required')
        bind(a.root/'monetary_protocol.json',plan)
        client=MonetaryClient(get_client(config),EuroLedger(Path(__file__).resolve().parents[1]/'results/prospective_budget/eur_budget.sqlite',plan['budget_eur']),bound)
        print(study.generate(a.split,a.model,client=client))

if __name__=='__main__':main()
