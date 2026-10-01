from pathlib import Path
from src.context_proxy.store import ConversationStore
from src.context_proxy.memory import MemoryIndex
from src.context_proxy.code_index import CodeIndex
from src.context_proxy.context import build_context

def test_store_roundtrip(tmp_path):
    s=ConversationStore(str(tmp_path/'m.db')); s.add('x','user','hello'); assert s.recent('x')[0]['content']=='hello'; s.set_state('x',{'goal':'g'}); assert s.state('x')['goal']=='g'
def test_memory_fallback_or_chroma(tmp_path):
    m=MemoryIndex(str(tmp_path/'c')); m.add('1','binary search finds target',{'session_id':'s'}); assert m.search('binary target',k=2)
def test_code_index(tmp_path):
    (tmp_path/'a.py').write_text('def hello(name):\n    return name\n')
    c=CodeIndex(str(tmp_path)); assert c.index()==1; assert c.search('hello')[0]['symbol']=='hello'
def test_context_has_current():
    cfg={'context':{'system_budget':100,'summary_budget':100,'memory_budget':100,'code_budget':100,'recent_budget':100,'tool_budget':100,'output_reserve':100}}
    x=build_context(system='sys',current='question',recent=[],summary='',state={},memories=[],code=[],cfg=cfg); assert x[-1]['content']=='question'
