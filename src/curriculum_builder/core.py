from __future__ import annotations
from collections import defaultdict
import hashlib, random
def topic_num(topic_id: str) -> int:
    try: return int(topic_id[1:])
    except Exception: return -1
def stage_for(row: dict, stages: dict) -> str:
    dom=row.get('domain','python'); tid=row.get('topic_id','')
    n=topic_num(tid)
    for stage,spec in stages.items():
        rng=spec.get(dom)
        if not rng: continue
        if topic_num(rng[0]) <= n <= topic_num(rng[1]): return stage
    return 'D'
def stratum(row: dict) -> tuple:
    return (row.get('domain'),row.get('topic_id'),int(row.get('level',0)),row.get('task_type'),row.get('teacher_model') or row.get('source'),row.get('confidence_tier','unknown'))
def deterministic_interleave(rows: list[dict], seed: int=42) -> list[dict]:
    buckets=defaultdict(list)
    for r in rows: buckets[stratum(r)].append(r)
    rng=random.Random(seed)
    for v in buckets.values(): rng.shuffle(v)
    keys=list(buckets); rng.shuffle(keys); out=[]
    while keys:
        next_keys=[]
        for k in keys:
            if buckets[k]: out.append(buckets[k].pop())
            if buckets[k]: next_keys.append(k)
        keys=next_keys
    return out
def dataset_fingerprint(rows: list[dict]) -> str:
    h=hashlib.sha256()
    for r in rows: h.update(str(r.get('id','')).encode()); h.update(b'\n')
    return h.hexdigest()
