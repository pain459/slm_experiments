from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from src.common.io import read_jsonl, write_jsonl
from src.judge.model import HFJsonModel
from src.judge.prompts import judge_prompt, test_generation_prompt
from src.judge.scoring import score_executable, score_explanatory
from src.verification.tests import verify_solution

EXECUTABLE={"implementation","debugging","repair","optimization","test_generation","refactoring"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--config',default='02_judge/judges.yaml')
    ap.add_argument('--output-dir',default='02_judge/outputs/run'); ap.add_argument('--limit',type=int); ap.add_argument('--skip-independent-tests',action='store_true')
    args=ap.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text()); jc=cfg['judge']; model=HFJsonModel(jc['model_id'],jc.get('revision','main'),jc.get('dtype','bfloat16'))
    outs={k:[] for k in ('gold','silver','review','reject')}; rows=list(read_jsonl(args.input)); rows=rows[:args.limit] if args.limit else rows
    for row in rows:
        task=row.get('task_type','implementation'); executable=task in EXECUTABLE
        tests=list(row.get('tests') or [])
        if executable and not args.skip_independent_tests:
            try: tests += list(model.generate(test_generation_prompt(row),700,0.0).get('tests') or [])
            except Exception: pass
        execution=False
        if executable:
            res=verify_solution(row.get('answer') or row.get('response',''),tests,timeout=5.0); execution=res.ok
        try: j=model.generate(judge_prompt(row),jc.get('max_new_tokens',1200),jc.get('temperature',0.0))
        except Exception as e: j={"correctness":0,"clarity":0,"test_quality":0,"curriculum_fit":0,"completeness":0,"conceptual_consistency":0,"notes":f"judge_error:{e}"}
        if executable:
            sr=score_executable(execution=execution,correctness=j.get('correctness',0),clarity=j.get('clarity',0),test_quality=j.get('test_quality',0),curriculum_fit=j.get('curriculum_fit',0),tiers=cfg.get('tiers'),weights=cfg.get('scoring',{}).get('executable'))
        else:
            sr=score_explanatory(correctness=j.get('correctness',0),completeness=j.get('completeness',0),conceptual_consistency=j.get('conceptual_consistency',0),curriculum_fit=j.get('curriculum_fit',0),tiers=cfg.get('tiers'),weights=cfg.get('scoring',{}).get('explanatory'))
        out={**row,"confidence":sr.confidence,"confidence_tier":sr.tier,"judge":{"components":sr.components,"notes":j.get('notes',''),"independent_tests":tests[len(row.get('tests') or []):],"execution":execution,"hard_reject":sr.hard_reject}}
        outs[sr.tier].append(out)
    od=Path(args.output_dir); od.mkdir(parents=True,exist_ok=True)
    for k,v in outs.items(): write_jsonl(od/f'{k}.jsonl',v)
    report={k:len(v) for k,v in outs.items()}; report['total']=sum(report.values()); (od/'score_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__': main()
