from __future__ import annotations
import argparse, json, os, random, subprocess
from pathlib import Path
import numpy as np, torch, yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.common.io import read_jsonl, sha256_file
from src.training.format_chat import record_to_messages


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: return "unknown"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--max_steps",type=int,default=-1); ap.add_argument("--output")
    args=ap.parse_args(); cfg=yaml.safe_load(open(args.config)); seed_all(int(cfg.get("seed",42)))
    out=Path(args.output or cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(cfg["base_model"],revision=cfg.get("revision","main")); tok.pad_token=tok.pad_token or tok.eos_token
    quant=None
    if cfg.get("load_in_4bit",False):
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(cfg["base_model"],revision=cfg.get("revision","main"),device_map="auto",quantization_config=quant,
                                             torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if cfg.get("gradient_checkpointing",True): model.gradient_checkpointing_enable()
    if cfg.get("load_in_4bit",False): model=prepare_model_for_kbit_training(model)
    lc=cfg.get("lora",{})
    if lc.get("enabled",True):
        model=get_peft_model(model,LoraConfig(r=lc.get("r",32),lora_alpha=lc.get("alpha",64),lora_dropout=lc.get("dropout",.05),
                  target_modules=lc.get("target_modules"),task_type="CAUSAL_LM"))
    def build(path):
        rows=list(read_jsonl(path)); texts=[]
        for r in rows:
            msgs=record_to_messages(r)
            text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False) if hasattr(tok,"apply_chat_template") else "\n".join(m["content"] for m in msgs)
            texts.append({"text":text})
        ds=Dataset.from_list(texts)
        def enc(batch):
            x=tok(batch["text"],truncation=True,max_length=int(cfg.get("max_length",2048)))
            x["labels"]=[ids[:] for ids in x["input_ids"]]
            return x
        return ds.map(enc,batched=True,remove_columns=["text"])
    train=build(cfg["train_file"]); val=build(cfg["validation_file"])
    ta=TrainingArguments(output_dir=str(out),learning_rate=float(cfg.get("learning_rate",1e-4)),num_train_epochs=float(cfg.get("num_train_epochs",1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size",1)),per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size",1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps",8)),logging_steps=int(cfg.get("logging_steps",10)),
        save_steps=int(cfg.get("save_steps",200)),eval_steps=int(cfg.get("eval_steps",200)),eval_strategy="steps",save_strategy="steps",
        warmup_ratio=float(cfg.get("warmup_ratio",.03)),weight_decay=float(cfg.get("weight_decay",.01)),bf16=bool(cfg.get("bf16",True) and torch.cuda.is_available()),
        fp16=bool(cfg.get("fp16",False)),max_steps=args.max_steps,report_to=[],remove_unused_columns=False)
    trainer=Trainer(model=model,args=ta,train_dataset=train,eval_dataset=val,data_collator=DataCollatorForLanguageModeling(tok,mlm=False))
    trainer.train(); trainer.save_model(out); tok.save_pretrained(out)
    meta={"config":cfg,"git_commit":git_commit(),"train_sha256":sha256_file(cfg["train_file"]),"validation_sha256":sha256_file(cfg["validation_file"])}
    (out/"experiment.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
if __name__=="__main__": main()
