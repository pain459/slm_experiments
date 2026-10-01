from __future__ import annotations
import argparse, json, re, torch
from src.agent.environment import RepoEnvironment
from src.agent.tools import WorkspaceTools
from src.common.modeling import load_causal_lm, chat_prompt

TOOL_DESC='''You may call exactly these tools: read_file(path), write_file(path, content), edit_file(path, old, new, count?), search_files(query), run_python(path), run_tests(args?), run_shell(command, timeout?), git_diff().\nWhen using a tool, output ONLY JSON: {"tool":"name","arguments":{...}}. When finished, output JSON: {"final":"summary"}.'''

def parse_obj(s):
    m=re.search(r"\{.*\}",s,re.S)
    return json.loads(m.group(0)) if m else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--repo",required=True); ap.add_argument("--max-steps",type=int,default=15)
    args=ap.parse_args(); env=RepoEnvironment(args.repo); tools=WorkspaceTools(env.tmp); model,tok=load_causal_lm(args.model)
    history=[{"role":"system","content":TOOL_DESC},{"role":"user","content":env.task.get("task","")}]; traj=[]
    for _ in range(args.max_steps):
        p=chat_prompt(tok,history); b=tok(p,return_tensors="pt").to(model.device)
        with torch.inference_mode(): out=model.generate(**b,max_new_tokens=600,do_sample=False,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,b["input_ids"].shape[1]:],skip_special_tokens=True); obj=parse_obj(text); traj.append({"assistant":text})
        if not obj: history.append({"role":"assistant","content":text}); history.append({"role":"user","content":"Invalid JSON. Use the required schema."}); continue
        if "final" in obj: break
        res=tools.call(obj.get("tool",""),obj.get("arguments",{})); traj[-1]["tool_result"]=res
        history += [{"role":"assistant","content":json.dumps(obj)},{"role":"user","content":"TOOL_RESULT "+json.dumps(res)}]
    final=tools.run_tests(); print(json.dumps({"trajectory":traj,"final_test":final},indent=2)); env.cleanup()
if __name__=="__main__": main()
