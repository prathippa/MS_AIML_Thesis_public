import copy
import pytest
from analysis.product_routing import product_rows, select, compare


def records():
    return [{'case_id': str(i), 'model': m, 'description_hash': str(i), 'gold': '1234567890',
             'status': 'ok', 'pred': '1234567890' if i==0 or m=='A' else '1234567891',
             'confidence': .2 if i==0 else .9, 'correct': i==0, 'severity_tier': 'correct_10' if i==0 else 'wrong_chapter'}
            for i in range(2) for m in ('A','B')]


def test_one_row_per_product_and_agreement_priority():
    rows=product_rows(records(),'A',{'1234567890'})
    assert len(rows)==2
    assert select(rows,.5,'confidence')[0]['id']=='1'
    assert select(rows,.5,'agreement')[0]['id']=='0'


def test_gold_and_correctness_do_not_enter_policy():
    rows=product_rows(records(),'A',{'1234567890'})
    altered=copy.deepcopy(rows)
    for r in altered: r['correct']=not r['correct'];r['wrong_chapter']=not r['wrong_chapter']
    for p in ('confidence','agreement','schedule','combined'):
        assert [r['id'] for r in select(rows,.5,p)]==[r['id'] for r in select(altered,.5,p)]


def test_failed_designated_output_always_referred():
    data=records();data[0]['status']='failed';data[0]['confidence']=None
    rows=product_rows(data,'A')
    assert len(select(rows,1,'confidence'))==1
    assert not rows[0]['agreement']


def test_failed_secondary_is_disagreement():
    data=records();data[1]['status']='failed'
    assert not product_rows(data,'A')[0]['agreement']


def test_missing_schedule_is_not_treated_as_unlisted():
    with pytest.raises(ValueError,match='schedule'):
        select(product_rows(records(),'A'),.5,'schedule')


def test_duplicate_model_fails():
    data=records()
    with pytest.raises(ValueError,match='Duplicate'): product_rows(data+[data[0]],'A')


def test_missing_model_fails():
    with pytest.raises(ValueError,match='Both'): product_rows(records()[:-1],'A')


def test_mismatched_product_fails():
    data=records();data[1]['description_hash']='different'
    with pytest.raises(ValueError,match='alignment'): product_rows(data,'A')


def test_paired_bootstrap_reproducible_and_baseline_zero():
    rows=product_rows(records(),'A')
    a=compare(rows,('confidence','agreement'),n_boot=30)
    assert a==compare(rows,('confidence','agreement'),n_boot=30)
    assert all(r['paired_ci95']==[0.,0.] for r in a['comparisons'] if r['policy']=='confidence')


def test_ties_independent_of_input_order():
    rows=product_rows(records(),'A')
    for r in rows:r['confidence']=.5
    assert select(rows,.5,'confidence')==select(list(reversed(rows)),.5,'confidence')


def test_candidate_export_preserves_gold_and_reasoning(tmp_path):
    import json
    from analysis.prepare_extension_data import to_message
    from data_loader import load_atlas_messages
    row={'source_row_id':'x','description':'item','gold':'1234567890','reasoning':'Not 9876543210.'}
    path=tmp_path/'rows.jsonl';path.write_text(json.dumps(to_message(row))+'\n')
    rows=load_atlas_messages(path,'validation','exclude')
    assert len(rows)==1 and rows[0]['gold']==row['gold'] and rows[0]['reasoning']==row['reasoning']
