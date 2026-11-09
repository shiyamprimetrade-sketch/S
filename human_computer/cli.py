from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import HumanComputer

DEFAULT_HOME=Path.home()/'.axiom-human-computer'

def main(argv=None):
    p=argparse.ArgumentParser(description='Axiom Human Computer: local whole-system research foundation')
    p.add_argument('--home',default=str(DEFAULT_HOME))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('shell');sub.add_parser('demo');sub.add_parser('status');sub.add_parser('desktop')
    q=sub.add_parser('ask');q.add_argument('message',nargs='+')
    args=p.parse_args(argv)
    pc=HumanComputer(args.home)
    try:
        if args.command=='desktop':
            from .desktop import run
            run(pc)
        elif args.command=='status': print(json.dumps(pc.status(),indent=2))
        elif args.command=='ask': print(json.dumps(pc.respond(' '.join(args.message)).as_json(),indent=2))
        elif args.command=='demo':
            for cmd in ('teach mirror CAT => CATTAC','teach mirror DOG => DOGGOD','invent mirror',
                        'perform mirror SUN','verify mirror SUN => SUNNUS',
                        'goal Build a human-like computer that tests its own outputs', 'status'):
                print('\n> '+cmd+'\n'+json.dumps(pc.respond(cmd).as_json(),indent=2))
        else:
            print('Axiom Human Computer | offline | type help or exit')
            while True:
                try: q=input('axiom-pc> ')
                except (KeyboardInterrupt,EOFError): break
                if q.strip().lower() in ('exit','quit'):break
                print(pc.respond(q).text)
    finally:pc.close()

if __name__=='__main__': main()