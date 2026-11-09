"""CLI: python -m continuum.lab_cli build|evaluate|ask|stats"""
import argparse
import json
from . import teacherlab as lab

def main(argv=None):
    p=argparse.ArgumentParser(description='AxiomFlow Teacher/Student offline simulation lab')
    sub=p.add_subparsers(dest='cmd',required=True)
    build=sub.add_parser('build',help='Generate simulated dialogues, train an intent model and evaluate')
    build.add_argument('--simulations',type=int,default=4000)
    build.add_argument('--seed',type=int,default=20261109)
    sub.add_parser('evaluate',help='Repeat held-out evaluation')
    ask=sub.add_parser('ask',help='Ask tiny offline behavior model (not an LLM)')
    ask.add_argument('text')
    sub.add_parser('stats',help='Inspect training results')
    args=p.parse_args(argv)
    if args.cmd=='build':print(json.dumps(lab.build_default(args.simulations,args.seed),indent=2))
    elif args.cmd=='evaluate':print(json.dumps(lab.evaluate(lab.load_model(),lab.generate(350,20261110,'heldout')),indent=2))
    elif args.cmd=='ask':print(lab.respond_offline(args.text))
    elif args.cmd=='stats':print(lab.REPORT.read_text(encoding='utf8'))
if __name__=='__main__':main()
