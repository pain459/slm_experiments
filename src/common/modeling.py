from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_causal_lm(model_id: str, revision: str = "main", load_in_4bit: bool = False, trust_remote_code: bool = False):
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant,
    )
    return model, tokenizer


def chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
