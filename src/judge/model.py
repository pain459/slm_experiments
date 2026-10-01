from __future__ import annotations
import json, re

def parse_json_object(text: str) -> dict:
    text=text.strip()
    try: return json.loads(text)
    except Exception: pass
    m=re.search(r"\{.*\}",text,re.S)
    if not m: raise ValueError("no JSON object")
    return json.loads(m.group(0))

class HFJsonModel:
    def __init__(self, model_id: str, revision: str="main", dtype: str="bfloat16", load_in_4bit: bool=False):
        from src.common.modeling import load_causal_lm
        self.model,self.tok=load_causal_lm(model_id,revision=revision,dtype=dtype,load_in_4bit=load_in_4bit)
    def generate(self, prompt: str, max_new_tokens: int=1000, temperature: float=0.0) -> dict:
        import torch
        msgs=[{"role":"user","content":prompt}]
        try: rendered=self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        except Exception: rendered=prompt
        batch=self.tok(rendered,return_tensors="pt").to(self.model.device)
        kwargs=dict(max_new_tokens=max_new_tokens,pad_token_id=self.tok.eos_token_id)
        if temperature and temperature>0: kwargs.update(do_sample=True,temperature=temperature)
        else: kwargs.update(do_sample=False)
        with torch.inference_mode(): out=self.model.generate(**batch,**kwargs)
        text=self.tok.decode(out[0,batch["input_ids"].shape[1]:],skip_special_tokens=True)
        return parse_json_object(text)
