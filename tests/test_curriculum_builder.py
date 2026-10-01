from src.curriculum_builder.core import stage_for,deterministic_interleave,dataset_fingerprint
ST={'A':{'python':['P00','P16'],'dsa':['D00','D11']},'B':{'python':['P17','P32'],'dsa':['D12','D28']},'C':{'python':['P33','P55'],'dsa':['D29','D41']},'D':{'python':['P56','P62'],'dsa':['D42','D61']}}
def test_stage(): assert stage_for({'domain':'dsa','topic_id':'D29'},ST)=='C'
def test_shuffle_reproducible():
    rows=[{'id':str(i),'domain':'python','topic_id':'P01','level':i%2,'task_type':'implementation','source':'a'} for i in range(20)]
    assert [x['id'] for x in deterministic_interleave(rows,42)]==[x['id'] for x in deterministic_interleave(rows,42)]
    assert dataset_fingerprint(rows)==dataset_fingerprint(rows)
