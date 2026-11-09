"""Example S API client: reads the key from an environment variable."""
import argparse
import json
import os
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('models')
    c=sub.add_parser('chat')
    c.add_argument('--session',required=True)
    c.add_argument('--text',required=True)
    c.add_argument('--model',default='s-research:local')
    a=p.parse_args()
    key=os.environ.get('S_API_KEY')
    if not key:p.error('Set S_API_KEY privately before running')
    path='/v1/models';data=None
    headers={'Authorization':'Bearer '+key}
    if a.cmd=='chat':
        path='/v1/chat'
        data=json.dumps({'session_id':a.session,'text':a.text,'model':a.model}).encode()
        headers['Content-Type']='application/json'
    r=Request('http://127.0.0.1:8765'+path,headers=headers,data=data,
              method='POST' if data else 'GET')
    try:
        with urlopen(r,timeout=180) as out:
            print(json.dumps(json.load(out),indent=2,ensure_ascii=False))
    except HTTPError as e:p.exit(1,f'S returned HTTP {e.code}; check key and scope.\n')
    except URLError:p.exit(1,'Cannot connect to local S; start the application first.\n')

if __name__=='__main__':main()
