from __future__ import annotations
import sqlite3, json, time
from pathlib import Path

class ConversationStore:
    def __init__(self,path: str):
        Path(path).parent.mkdir(parents=True,exist_ok=True); self.db=sqlite3.connect(path,check_same_thread=False); self.db.row_factory=sqlite3.Row; self._init()
    def _init(self):
        self.db.execute('CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,role TEXT,content TEXT,metadata TEXT,ts REAL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id,id)')
        self.db.execute('CREATE TABLE IF NOT EXISTS summaries(session_id TEXT PRIMARY KEY,content TEXT,updated REAL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS state(session_id TEXT PRIMARY KEY,json TEXT,updated REAL)')
        self.db.commit()
    def add(self,session_id,role,content,metadata=None): self.db.execute('INSERT INTO messages(session_id,role,content,metadata,ts) VALUES(?,?,?,?,?)',(session_id,role,content,json.dumps(metadata or {}),time.time())); self.db.commit()
    def recent(self,session_id,limit=20): return [dict(r) for r in self.db.execute('SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?',(session_id,limit)).fetchall()][::-1]
    def count(self,session_id): return self.db.execute('SELECT COUNT(*) FROM messages WHERE session_id=?',(session_id,)).fetchone()[0]
    def set_summary(self,session_id,content): self.db.execute('INSERT INTO summaries VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET content=excluded.content,updated=excluded.updated',(session_id,content,time.time())); self.db.commit()
    def summary(self,session_id):
        r=self.db.execute('SELECT content FROM summaries WHERE session_id=?',(session_id,)).fetchone(); return r[0] if r else ''
    def set_state(self,session_id,obj): self.db.execute('INSERT INTO state VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET json=excluded.json,updated=excluded.updated',(session_id,json.dumps(obj),time.time())); self.db.commit()
    def state(self,session_id):
        r=self.db.execute('SELECT json FROM state WHERE session_id=?',(session_id,)).fetchone(); return json.loads(r[0]) if r else {}
