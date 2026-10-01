from __future__ import annotations
import argparse, subprocess, sys

def run(args):
    print('+',' '.join(args)); raise SystemExit(subprocess.call(args))
def main():
    ap=argparse.ArgumentParser(description='Six-module Python specialist pipeline'); sub=ap.add_subparsers(dest='cmd',required=True)
    d=sub.add_parser('distill'); d.add_argument('rest',nargs=argparse.REMAINDER)
    j=sub.add_parser('judge'); j.add_argument('rest',nargs=argparse.REMAINDER)
    c=sub.add_parser('curriculum'); c.add_argument('rest',nargs=argparse.REMAINDER)
    t=sub.add_parser('train'); t.add_argument('rest',nargs=argparse.REMAINDER)
    a=sub.add_parser('agent'); a.add_argument('rest',nargs=argparse.REMAINDER)
    p=sub.add_parser('proxy'); p.add_argument('rest',nargs=argparse.REMAINDER)
    x=ap.parse_args(); mapping={'distill':['-m','src.distill.cli'],'judge':['-m','src.judge.run'],'curriculum':['-m','src.curriculum_builder.build'],'train':['-m','src.unsloth_train.train'],'agent':['-m','src.agent.generate_trajectories'],'proxy':['-m','src.context_proxy.server']}; run([sys.executable,*mapping[x.cmd],*x.rest])
if __name__=='__main__': main()
