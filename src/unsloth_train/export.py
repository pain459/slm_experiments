from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--out',default='artifacts/gguf'); ap.add_argument('--quant',nargs='*',default=['q8_0','q4_k_m']); args=ap.parse_args()
    try: from unsloth import FastLanguageModel
    except Exception as e: raise SystemExit('Unsloth required for export: '+str(e))
    model,tok=FastLanguageModel.from_pretrained(args.model,load_in_4bit=False); od=Path(args.out); od.mkdir(parents=True,exist_ok=True)
    for q in args.quant: model.save_pretrained_gguf(str(od/q),tok,quantization_method=q)
if __name__=='__main__': main()
