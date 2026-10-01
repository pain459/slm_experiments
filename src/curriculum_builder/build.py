from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter,defaultdict
import yaml
from src.common.io import read_jsonl, write_jsonl
from src.curriculum_builder.core import stage_for, deterministic_interleave, dataset_fingerprint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',nargs='+',required=True); ap.add_argument('--config',default='03_curriculum/config.yaml'); ap.add_argument('--output-dir',default='03_curriculum/outputs/dataset_v1'); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); rows=[]; seen=set()
    for p in args.inputs:
        for r in read_jsonl(p):
            if r.get('confidence_tier') not in cfg.get('include_tiers',['gold','silver']): continue
            key=r.get('id') or (r.get('prompt'),r.get('answer') or r.get('response'))
            if str(key) in seen: continue
            seen.add(str(key)); rows.append(r)
    by=defaultdict(list)
    for r in rows: by[stage_for(r,cfg['stages'])].append(r)
    od=Path(args.output_dir); od.mkdir(parents=True,exist_ok=True); full=[]; manifest={'seed':cfg['seed'],'stages':{},'topics':{},'source_files':args.inputs}
    for stage in ['A','B','C','D']:
        seq=deterministic_interleave(by.get(stage,[]),cfg['seed']+ord(stage)); write_jsonl(od/f'stage_{stage.lower()}.jsonl',seq); full.extend(seq); manifest['stages'][stage]=len(seq)
    # one final deterministic interleave preserves mix while all stages remain represented
    full=deterministic_interleave(full,cfg['seed']); write_jsonl(od/'train_full.jsonl',full)
    c=Counter((r.get('topic_id'),str(r.get('level')),r.get('task_type')) for r in full)
    manifest['topics']={f'{a}|L{b}|{c0}':n for (a,b,c0),n in sorted(c.items())}; manifest['total']=len(full); manifest['fingerprint']=dataset_fingerprint(full)
    (od/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({'total':len(full),'fingerprint':manifest['fingerprint']}))
if __name__=='__main__': main()
