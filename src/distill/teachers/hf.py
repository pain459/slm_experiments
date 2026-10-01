from __future__ import annotations

import torch

from src.common.modeling import chat_prompt, load_causal_lm
from src.distill.teachers.base import TeacherBackend, TeacherConfig


class HuggingFaceTeacher(TeacherBackend):
    def __init__(self, config: TeacherConfig):
        self.config = config
        self.model, self.tokenizer = load_causal_lm(
            config.model_id,
            config.revision,
            config.load_in_4bit,
        )

    @torch.inference_mode()
    def generate_text(self, system: str, user: str) -> str:
        g = self.config.generation
        prompt = chat_prompt(self.tokenizer, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        batch = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        temperature = float(g.get("temperature", 0.4))
        output = self.model.generate(
            **batch,
            max_new_tokens=int(g.get("max_new_tokens", 1800)),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=float(g.get("top_p", 0.95)),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
