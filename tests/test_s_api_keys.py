"""Real S API key generation and protected endpoint regression tests."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from continuum.api_keys import create_key,list_keys,revoke_key,authenticate_key
from continuum.server import AppServer
from continuum.storage import Store

class SKeyStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.s=Store(Path(self.temp.name)/'test.sqlite3')
    def tearDown(self):self.temp.cleanup()
    def test_randomness(self):
        a=create_key(self.s,'First')
        b=create_key(self.s,'Second')
        self.assertNotEqual(a['key'],b['key'])
        self.assertEqual(len(a['key'][len(a['prefix']):]),43)
        self.assertTrue(authenticate_key(self.s,a['key'],'chat'))
    def test_never_store_plaintext(self):
        a=create_key(self.s,'Local Client')
        with self.s.connect() as c:r=str(dict(c.execute('SELECT * FROM api_keys').fetchone()))
        self.assertNotIn(a['key'],r)
        self.assertNotIn(a['key'],json.dumps(list_keys(self.s)))
        self.assertNotIn('key',list_keys(self.s)[0])
    def test_persistence_and_revocation(self):
        a=create_key(self.s,'Client')
        fresh=Store(self.s.path)
        self.assertTrue(authenticate_key(fresh,a['key'],'read'))
        self.assertTrue(revoke_key(fresh,a['id'])['revoked'])
        self.assertFalse(authenticate_key(self.s,a['key'],'read'))
    def test_rejects_fake_keys_and_scopes(self):
        a=create_key(self.s,'Reader',['read'])
        self.assertTrue(authenticate_key(self.s,a['key'],'read'))
        self.assertFalse(authenticate_key(self.s,a['key'],'chat'))
        self.assertFalse(authenticate_key(self.s,a['key']+'x','read'))
        self.assertFalse(authenticate_key(self.s,None,'read'))
        self.assertFalse(authenticate_key(self.s,'sk_s_fake','read'))
    def test_validation(self):
        for name in ('',' ', 'x'*81):
            with self.assertRaises(ValueError):create_key(self.s,name)
        for scopes in ([],['write'],['read','admin'],None):
            with self.assertRaises(ValueError):create_key(self.s,'App',scopes)

class SKeyHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.srv=AppServer('127.0.0.1',0,Path(self.temp.name)/'s.db')
        self.thread=threading.Thread(target=self.srv.serve_forever,daemon=True)
        self.thread.start()
        self.base='http://127.0.0.1:'+str(self.srv.server_port)
    def tearDown(self):
        self.srv.shutdown();self.srv.server_close();self.thread.join(3)
        self.temp.cleanup()
    def send(self,path,data=None,key=None,origin=None):
        hdr={}
        if data is not None:hdr['Content-Type']='application/json'
        if key is not None:hdr['Authorization']='Bearer '+key
        if origin is not None:hdr['Origin']=origin
        req=Request(self.base+path,headers=hdr,data=json.dumps(data).encode() if data is not None else None,
                    method='POST' if data is not None else 'GET')
        try:
            with urlopen(req,timeout=10) as r:return r.status,json.load(r)
        except HTTPError as e:return e.code,json.load(e)
    def test_issuance_listing_revocation(self):
        code,a=self.send('/api/keys',{'name':'My tool','scopes':['read','chat']})
        self.assertEqual(code,200)
        self.assertTrue(a['key'].startswith('sk_s_'))
        code,k=self.send('/api/keys')
        self.assertEqual(code,200)
        self.assertNotIn('key',k[0])
        self.assertEqual(self.send('/api/revoke-key',{'id':a['id']})[1]['revoked'],True)
        self.assertEqual(self.send('/v1/models',key=a['key'])[0],401)
    def test_bearer_auth_required(self):
        self.assertEqual(self.send('/v1/models')[0],401)
        self.assertEqual(self.send('/v1/models',key='wrong')[0],401)
        self.assertEqual(self.send('/v1/chat',{'session_id':'invalid','text':'Hello'})[0],401)
    def test_scopes(self):
        _,reader=self.send('/api/keys',{'name':'Reader','scopes':['read']})
        self.assertEqual(self.send('/v1/models',key=reader['key'])[0],200)
        self.assertEqual(self.send('/v1/chat',{'session_id':'nope','text':'Hello'},key=reader['key'])[0],401)
        _,writer=self.send('/api/keys',{'name':'Chat only','scopes':['chat']})
        self.assertEqual(self.send('/v1/models',key=writer['key'])[0],401)
    def test_real_chat_with_revoke(self):
        p=self.srv.store.list_projects()[0]
        ses=self.srv.store.create_session(p['id'],'API key test')
        _,k=self.send('/api/keys',{'name':'Agent'})
        code,reply=self.send('/v1/chat',{'session_id':ses['id'],'text':'Invent a word rule','model':'s-research:local'},key=k['key'])
        self.assertEqual(code,200)
        self.assertEqual(reply['role'],'assistant')
        self.assertEqual(len(self.srv.store.list_messages(ses['id'])),2)
        self.assertIsNotNone(self.send('/api/keys')[1][0]['last_used_at'])
        self.send('/api/revoke-key',{'id':k['id']})
        self.assertEqual(self.send('/v1/chat',{'session_id':ses['id'],'text':'again'},key=k['key'])[0],401)
    def test_browser_origin_guard(self):
        self.assertEqual(self.send('/api/keys',{'name':'Attacker'},origin='https://example.invalid')[0],403)
        self.assertEqual(list_keys(self.srv.store),[])
    def test_unknown_route(self):
        _,k=self.send('/api/keys',{'name':'Agent'})
        self.assertEqual(self.send('/v1/unknown',key=k['key'])[0],404)

if __name__=='__main__':unittest.main()
