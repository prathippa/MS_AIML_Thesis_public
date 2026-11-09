import numpy as np
from analysis.verify_v10 import order,risk

def test_weighted_selection_counts_duplicate_products_at_boundary():
 # Three copies of item1 cross k=2: both accepted copies must count as errors.
 assert risk(np.array([0,3,1]),np.array([0,1,2]),np.array([False,True,False]),2)==1

def test_no_eligible_outputs_has_no_risk():
 assert risk(np.array([2]),np.array([],dtype=int),np.array([True]),1) is None

def test_order_does_not_use_gold_and_excludes_failure():
 rows=[{'case_id':'a','status':'ok','pred':'1234567890','confidence':.7,'gold':'0000000000'}, {'case_id':'b','status':'failed','pred':None,'confidence':1,'gold':'1234567890'}]
 other=[dict(r) for r in rows]
 a=order(rows,other,'combined',{'1234567890'})
 rows[0]['gold']='1234567890';b=order(rows,other,'combined',{'1234567890'})
 assert a.tolist()==b.tolist()==[0]
