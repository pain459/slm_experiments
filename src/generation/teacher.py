from __future__ import annotations
import json, re, torch
from src.common.modeling import load_causal_lm, chat_prompt


def extract_json(text: str) -> dict:
    text=text.strip()
    try: return json.loads(text)
    except Exception: pass
    m=re.search(r"\{.*\}", text, re.S)
    if not m: raise ValueError("no JSON object found")
    return json.loads(m.group(0))

class Teacher:
    def __init__(self, model_id: str, revision="main", load_in_4bit=False):
        self.model, self.tok = load_causal_lm(model_id, revision, load_in_4bit)

    @torch.inference_mode()
    def generate(self, system: str, user: str, max_new_tokens=1400, temperature=.35, top_p=.95) -> dict:
        prompt=chat_prompt(self.tok,[{"role":"system","content":system},{"role":"user","content":user}])
        batch=self.tok(prompt, return_tensors="pt").to(self.model.device)
        out=self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=temperature>0,
                                temperature=max(temperature,1e-5), top_p=top_p,
                                pad_token_id=self.tok.eos_token_id)
        new=out[0, batch["input_ids"].shape[1]:]
        return extract_json(self.tok.decode(new, skip_special_tokens=True))
