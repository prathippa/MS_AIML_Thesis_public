from pathlib import Path
import shutil
import pytest
from analysis import rq3_judge as rq3

@pytest.fixture
def sample(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[1]/'results/main_v1'
    shutil.copytree(root/'rq3',tmp_path/'rq3')
    shutil.copytree(root/'test',tmp_path/'test')
    monkeypatch.setattr(rq3,'verify_inputs',lambda root:None)
    return tmp_path

@pytest.mark.parametrize('field,value',[('majority','unsupported'),('consistency',.123),('n_runs',99)])
def test_changed_aggregate_rejected(sample,field,value):
    path=sample/'rq3/judge_results.jsonl';rows=rq3.lines(path);rows[0][field]=value;rq3.write_lines(path,rows)
    with pytest.raises(ValueError,match='aggregates'):rq3.validated_rows(sample)


def test_changed_raw_response_rejected(sample):
    path=sample/'rq3/judge_results.jsonl';rows=rq3.lines(path)
    rows[0]['history'][0]['raw']='{"label":"unsupported","reason":"changed"}'
    rq3.write_lines(path,rows)
    with pytest.raises(ValueError,match='raw response'):rq3.validated_rows(sample)


def test_real_rows_and_product_cluster(sample):
    rows,model=rq3.validated_rows(sample)
    result=rq3.clustered_auc(rows,n_boot=30)
    assert len(rows)==60 and result['n_products']==55
    assert result['auc']==pytest.approx(.7411111111)
