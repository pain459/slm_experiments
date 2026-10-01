from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='04_train/configs/unsloth_v1.yaml'); args=ap.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig
    except Exception as e:
        raise SystemExit('Install Unsloth/TRL in the training environment. See 04_train/README.md. '+str(e))
    model,tok=FastLanguageModel.from_pretrained(model_name=cfg['model_id'],max_seq_length=cfg['max_seq_length'],load_in_4bit=cfg.get('load_in_4bit',True))
    model=FastLanguageModel.get_peft_model(model,r=cfg.get('lora_r',32),lora_alpha=cfg.get('lora_alpha',64),lora_dropout=cfg.get('lora_dropout',0.0),target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    ds=load_dataset('json',data_files=cfg['train_file'],split='train')
    def fmt(ex):
        answer=ex.get('answer') or ex.get('response') or ''
        messages=[{'role':'user','content':ex.get('prompt','')},{'role':'assistant','content':answer}]
        return {'text':tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)}
    ds=ds.map(fmt)
    out=cfg['output_dir']; Path(out).mkdir(parents=True,exist_ok=True)
    trainer=SFTTrainer(model=model,tokenizer=tok,train_dataset=ds,dataset_text_field='text',args=SFTConfig(output_dir=out,per_device_train_batch_size=cfg.get('batch_size',2),gradient_accumulation_steps=cfg.get('gradient_accumulation_steps',8),learning_rate=cfg.get('learning_rate',2e-4),num_train_epochs=cfg.get('epochs',1),logging_steps=10,save_strategy='steps',save_steps=250,report_to='none'))
    trainer.train(); model.save_pretrained(out); tok.save_pretrained(out); (Path(out)/'training_config.json').write_text(json.dumps(cfg,indent=2))
if __name__=='__main__': main()
