from __future__ import annotations
import argparse, json, uuid
from src.common.io import read_jsonl, write_jsonl
from src.generation.teacher import Teacher
from src.generation.prompts import SYSTEM, REPAIR_SYSTEM, seed_prompt, repair_prompt


def convert(obj: dict, source: str) -> dict:
    return {
      "id": f"syn_{uuid.uuid4().hex[:20]}", "source": source, "license": "synthetic",
      "language":"python", "category":obj.get("category","python"), "skill":obj.get("skill","general"),
      "difficulty":int(obj.get("difficulty",2)), "task_type":obj.get("task_type","implementation"),
      "prompt":obj.get("problem",""), "response":obj.get("solution",""), "tests":obj.get("tests",[]),
      "verified":False, "metadata":{"complexity":obj.get("complexity",{}), "diagnosis":obj.get("diagnosis","")}
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--teacher", required=True); ap.add_argument("--revision",default="main")
    ap.add_argument("--seeds"); ap.add_argument("--input"); ap.add_argument("--mode",choices=["generate","repair"],default="generate")
    ap.add_argument("--count",type=int,default=100); ap.add_argument("--output",required=True); ap.add_argument("--rejected")
    ap.add_argument("--load-in-4bit", action="store_true"); args=ap.parse_args()
    source=args.seeds if args.mode=="generate" else args.input
    if not source: raise SystemExit("--seeds required for generate; --input for repair")
    rows=list(read_jsonl(source)); teacher=Teacher(args.teacher,args.revision,args.load_in_4bit); ok=[]; bad=[]
    for i,row in enumerate(rows[:args.count]):
        try:
            obj=teacher.generate(SYSTEM if args.mode=="generate" else REPAIR_SYSTEM,
                                 seed_prompt(row) if args.mode=="generate" else repair_prompt(row))
            ok.append(convert(obj, f"teacher:{args.teacher}"))
        except Exception as e:
            bad.append({"index":i,"error":repr(e),"input":row})
        if (i+1)%10==0: print(json.dumps({"processed":i+1,"ok":len(ok),"bad":len(bad)}))
    write_jsonl(args.output,ok)
    if args.rejected: write_jsonl(args.rejected,bad)
    print(json.dumps({"generated":len(ok),"rejected":len(bad)}))
if __name__=="__main__": main()
