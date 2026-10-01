import sys
from pathlib import Path
from src.agent.tools import WorkspaceTools

def test_read_write_search(tmp_path: Path):
    t=WorkspaceTools(tmp_path)
    assert t.write_file("a.py","x = 123\n")["status"]=="ok"
    assert "123" in t.read_file("a.py")["content"]
    assert t.search_files("123")["hits"]

def test_path_escape(tmp_path: Path):
    t=WorkspaceTools(tmp_path)
    r=t.call("read_file",{"path":"../secret.txt"})
    assert r["status"]=="error"

def test_edit_file(tmp_path):
    t=WorkspaceTools(tmp_path); (tmp_path/'a.py').write_text('x = 1\n')
    assert t.edit_file('a.py','1','2')['status']=='ok'
    assert (tmp_path/'a.py').read_text()=='x = 2\n'

def test_run_shell_no_shell_expansion(tmp_path):
    t=WorkspaceTools(tmp_path)
    r=t.run_shell([sys.executable,'-c','print(2+3)'])
    assert r['returncode']==0 and '5' in r['stdout']
