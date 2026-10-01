from src.judge.scoring import score_executable,score_explanatory

def test_execution_is_hard_gate():
    x=score_executable(execution=False,correctness=1,clarity=1,test_quality=1,curriculum_fit=1); assert x.tier=='reject' and x.hard_reject

def test_gold_score():
    x=score_executable(execution=True,correctness=.95,clarity=.95,test_quality=.9,curriculum_fit=1); assert x.tier=='gold'

def test_explanatory_score():
    x=score_explanatory(correctness=.9,completeness=.9,conceptual_consistency=.9,curriculum_fit=.9); assert .89 < x.confidence < .91
