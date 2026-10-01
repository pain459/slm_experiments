from __future__ import annotations
from pathlib import Path
import hashlib, math, re

class MemoryIndex:
    """Chroma when installed; deterministic lexical fallback otherwise."""
    def __init__(self,path: str,collection='coding_memory'):
        self.fallback=[]; self.collection=None
        try:
            import chromadb
            Path(path).mkdir(parents=True,exist_ok=True); client=chromadb.PersistentClient(path=path); self.collection=client.get_or_create_collection(collection)
        except Exception: self.collection=None
    def add(self,ident: str,text: str,metadata: dict|None=None):
        if not text.strip(): return
        if self.collection is not None: self.collection.upsert(ids=[ident],documents=[text],metadatas=[metadata or {}])
        else: self.fallback.append((ident,text,metadata or {}))
    def search(self,query: str,k=6,where: dict|None=None):
        if self.collection is not None:
            r=self.collection.query(query_texts=[query],n_results=k,where=where); return [{'id':i,'text':d,'metadata':m} for i,d,m in zip(r['ids'][0],r['documents'][0],r['metadatas'][0])]
        q=set(re.findall(r'\w+',query.lower())); scored=[]
        for i,t,m in self.fallback:
            if where and any(m.get(a)!=b for a,b in where.items()): continue
            w=set(re.findall(r'\w+',t.lower())); s=len(q&w)/max(1,len(q|w)); scored.append((s,i,t,m))
        return [{'id':i,'text':t,'metadata':m} for s,i,t,m in sorted(scored,reverse=True)[:k] if s>0]
