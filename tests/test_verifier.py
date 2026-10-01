from src.verification.tests import verify_solution
from src.verification.sandbox import run_python

def test_valid_solution():
    r=verify_solution("def add(a,b): return a+b", ["assert add(2,3)==5"])
    assert r.ok

def test_wrong_solution():
    r=verify_solution("def add(a,b): return a-b", ["assert add(2,3)==5"])
    assert not r.ok

def test_timeout():
    r=run_python("while True: pass", timeout=.2)
    assert r.timed_out

def test_blocked_shell():
    r=run_python("import os\nos.system('echo bad')")
    assert not r.ok and r.returncode==126
