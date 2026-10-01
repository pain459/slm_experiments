from __future__ import annotations
import argparse, subprocess, sys

def run(cmd):
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--eval",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    train=[sys.executable,"-m","src.training.train_sft","--config",args.config,"--output",args.output]
    if args.smoke: train += ["--max_steps","20"]
    run(train)
    result=f"{args.output}/eval_results.jsonl"
    run([sys.executable,"-m","src.evaluation.evaluate","--model",args.output,"--dataset",args.eval,"--output",result])
    run([sys.executable,"-m","src.evaluation.report",result])
if __name__=="__main__": main()
