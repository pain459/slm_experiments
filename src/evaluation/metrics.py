from __future__ import annotations
from collections import defaultdict

def summarize(rows: list[dict]) -> dict:
    total=len(rows); passed=sum(bool(r.get("passed")) for r in rows); syntax=sum(bool(r.get("syntax_valid")) for r in rows)
    by=defaultdict(lambda:[0,0])
    for r in rows:
        c=r.get("category","unknown"); by[c][1]+=1; by[c][0]+=int(bool(r.get("passed")))
    return {"n":total,"pass_rate":passed/max(1,total),"syntax_rate":syntax/max(1,total),
            "by_category":{k:{"passed":v[0],"n":v[1],"pass_rate":v[0]/v[1]} for k,v in sorted(by.items())}}
