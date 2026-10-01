from __future__ import annotations
import argparse, json, re, torch
from src.common.io import read_jsonl, write_jsonl
from src.common.modeling import load_causal_lm, chat_prompt
from src.verification.tests import verify_solution


def extract_code(text: str) -> str:
    blocks=re.findall(r"```(?:python)?\s*(.*?)```",text,re.S|re.I)
    return blocks[-1].strip() if blocks else text.strip()

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--revision",default="main")
    ap.add_argument("--dataset",required=True); ap.add_argument("--output",required=True); ap.add_argument("--max-new-tokens",type=int,default=800)
    ap.add_argument("--save-failures"); ap.add_argument("--load-in-4bit",action="store_true"); args=ap.parse_args()
    model,tok=load_causal_lm(args.model,args.revision,args.load_in_4bit); results=[]; failures=[]
    for row in read_jsonl(args.dataset):
        prompt=chat_prompt(tok,[{"role":"system","content":"Return a correct Python solution. Prefer code only."},{"role":"user","content":row["prompt"]}])
        batch=tok(prompt,return_tensors="pt").to(model.device); out=model.generate(**batch,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,batch["input_ids"].shape[1]:],skip_special_tokens=True); code=extract_code(text)
        try: compile(code,"<generated>","exec"); syntax=True
        except SyntaxError: syntax=False
        rr=verify_solution(code,row.get("tests",[])) if syntax else None; passed=bool(rr and rr.ok)
        result={"task_id":row["id"],"category":row.get("category"),"skill":row.get("skill"),"difficulty":row.get("difficulty"),
                "generated":text,"code":code,"syntax_valid":syntax,"passed":passed,"stderr":rr.stderr[-1500:] if rr else "syntax error"}
        results.append(result)
        if not passed: failures.append({**row,"student_response":code,"test_failure":result["stderr"]})
    write_jsonl(args.output,results)
    if args.save_failures: write_jsonl(args.save_failures,failures)
    print(json.dumps({"tasks":len(results),"passed":sum(r['passed'] for r in results)}))
if __name__=="__main__": main()
