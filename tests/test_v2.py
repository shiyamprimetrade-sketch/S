import io
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from continuum.storage import Store
from continuum.transcript import linear_messages,parse_export,redact_secrets,load_export
from continuum.charllm import ByteTokenizer,train,complete
from continuum.server import AppServer


def sample():
    return [{'id':'example-1','title':'Keyboard, letters, memory and terminal','current_node':'a3','mapping':{
        'a1':{'parent':None,'message':{'id':'m1','author':{'role':'user'},'content':{'parts':['The keyboard is the foundation.']},'create_time':1700000000}},
        'a2':{'parent':'a1','message':{'id':'m2','author':{'role':'assistant'},'content':{'parts':['I will test that hypothesis.']},'create_time':1700000001}},
        'a3':{'parent':'a2','message':{'id':'m3','author':{'role':'user'},'content':{'parts':['ABCA = FOUR',{'content_type':'image_asset_pointer','asset_pointer':'opaque'}]},'create_time':1700000002}},
        'unused':{'parent':'a1','message':{'id':'dead','author':{'role':'assistant'},'content':{'parts':['Ignore this replaced branch.']},'create_time':1700000010}}
    }}]

class TranscriptTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.t.name)/'db.sqlite3')
    def tearDown(self):self.t.cleanup()
    def test_parse_active_chain_and_media(self):
        result=parse_export(sample())
        self.assertEqual(len(result),1)
        messages=result[0]['messages']
        self.assertEqual(len(messages),3)
        self.assertNotIn('Ignore this',str(messages))
        self.assertIn('[ARCHIVED MEDIA:',messages[2]['content'])
        self.assertTrue(messages[0]['created_at'].endswith('+00:00'))
    def test_secret_redaction(self):
        exposed='eyJhIjoi'+'A'*75
        self.assertIn('[REDACTED_CREDENTIAL]',redact_secrets('token '+exposed))
        obj=sample();obj[0]['mapping']['a1']['message']['content']['parts']=[exposed]
        clean=parse_export(obj)[0]['messages'][0]['content']
        self.assertNotIn(exposed,clean)
        self.assertIn(exposed,parse_export(obj,redact=False)[0]['messages'][0]['content'])
    def test_db_import_search_and_isolation(self):
        existing=self.store.create_project('Other project')
        out=self.store.import_chatgpt(sample(),'Keyboard')
        self.assertEqual(out['conversations'],1)
        self.assertEqual(out['chat_messages'],3)
        archive=out['project_id']
        self.assertEqual(self.store.search_history(archive,'ABCA')[0]['role'],'user')
        self.assertEqual(self.store.search_history(existing['id'],'ABCA'),[])
        session=self.store.list_sessions(archive)[0]
        msgs=self.store.list_messages(session['id'])
        self.assertEqual([m['role'] for m in msgs],['user','assistant','user'])
        fresh=Store(self.store.path)
        self.assertEqual(len(fresh.list_messages(session['id'])),3)
    def test_tool_messages_preserved_in_extras(self):
        c=sample();c[0]['mapping']['a2']['message']['author']['role']='tool'
        out=self.store.import_chatgpt(c)
        self.assertEqual(out['other_messages'],1)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM archive_extras').fetchone()[0],1)
    def test_parse_invalid_and_title(self):
        with self.assertRaises(ValueError):parse_export({'unsupported':1})
        with self.assertRaises(ValueError):parse_export(sample(),'nonmatching')
        self.assertEqual(len(parse_export({'conversations':sample()})),1)
    def test_tokenizer_roundtrip_unicode(self):
        tok=ByteTokenizer();txt='Axiom — বাংলা 😀'
        self.assertEqual(tok.decode(tok.encode(txt)),txt)
        self.assertEqual(tok.vocab_size,256)
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,"PyTorch is an optional extra")
    def test_small_model_training_and_inference(self):
        # Small deterministic toy corpus: proves algorithm and save/load, not fluency.
        path=Path(self.t.name)/'tiny.pt'
        result=train('Teacher: A\nStudent: B\n'*80,path,steps=3,context=16,dim=16,heads=4,layers=1,batch=2)
        self.assertGreater(result['parameters'],0)
        self.assertTrue(path.is_file())
        s=complete('Teacher: A\nStudent: ',path,max_new_tokens=9,seed=7)
        self.assertIsInstance(s,str)
    def test_browser_import_api(self):
        srv=AppServer('127.0.0.1',0,Path(self.t.name)/'server.sqlite3')
        thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
        try:
            base='http://127.0.0.1:'+str(srv.server_port)
            body=json.dumps({'document':sample(),'title_contains':'keyboard'}).encode()
            req=Request(base+'/api/import-chatgpt',data=body,headers={'Content-Type':'application/json'},method='POST')
            with urlopen(req,timeout=10) as response:info=json.load(response)
            self.assertEqual(info['chat_messages'],3)
            with urlopen(base+'/api/search?project_id='+info['project_id']+'&q=FOUR',timeout=10) as response:
                self.assertEqual(len(json.load(response)),1)
        finally:srv.shutdown();srv.server_close();thread.join(3)

if __name__=='__main__':unittest.main()
