"""Real inference without torch; strict malformed tensor and budget checks."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from s_developer.portable_native import PortableBrain,read_f32
from s_developer.models import install_demo,configure_runtime,load_config,probe_native
from continuum.storage import Store
from s_developer.specifications import Specifications

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'s_developer'/'resources'/'native-demo'


class PortableTests(unittest.TestCase):
    def test_actual_generation_is_deterministic_and_hash_identified(self):
        brain=PortableBrain(CHECKPOINT)
        first=brain.generate('Request: Hello\n',16);second=brain.generate('Request: Hello\n',16)
        self.assertEqual(first['generated_byte_ids'],second['generated_byte_ids'])
        self.assertTrue(first['inference_completed']);self.assertEqual(first['generated_tokens'],16)
        self.assertEqual(first['weights_sha256'],brain.manifest.weights_sha256)
        self.assertIn('unverified',first['coding_capability'])
    def test_invalid_inputs_and_context_do_not_expand_work_budget(self):
        brain=PortableBrain(CHECKPOINT)
        for prompt,tokens in [('',16),('x'*65,16),('😀'*17,16),('a',129),('a',True)]:
            with self.assertRaises(ValueError):brain.generate(prompt,tokens)
        with patch('s_developer.portable_native.time.monotonic',side_effect=[0,1]):
            with self.assertRaises(TimeoutError):brain.generate('a',1,timeout=0.1)
        cache=brain.new_cache();cache['position']=brain.cfg['context']
        with self.assertRaises(ValueError):brain.step(65,cache)
    def test_dependency_free_python_process_can_load_and_run(self):
        code="import sys;sys.path.insert(0,"+repr(str(ROOT))+");from s_developer.portable_native import PortableBrain;print(PortableBrain("+repr(str(CHECKPOINT))+").generate('A',2)['runtime']);assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('python-stdlib-f32',result.stdout)
    def test_local_configuration_and_project_scoped_probe_need_consent(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3';install_demo(db);configure_runtime(db,'stdlib')
            config=load_config(db);self.assertEqual(config['native_backend'],'stdlib')
            specs=Specifications(Store(db),config);pid=specs.store.create_project('Synthetic runtime fixture')['id']
            with self.assertRaises(PermissionError):specs.engine.neural_probe(dict(project_id=pid,prompt='A',tokens=2,consent=False))
            result=specs.engine.neural_probe(dict(project_id=pid,prompt='A',tokens=2,consent=True))
            self.assertTrue(result['inference_completed']);self.assertEqual(specs.engine.runs(pid)[0]['stage'],'neural-probe')
            with self.assertRaises(ValueError):probe_native(config,'A',33)
            with self.assertRaises(ValueError):configure_runtime(db,'download-model')


class TensorReaderTests(unittest.TestCase):
    def check(self,header,payload,expected=None):
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'model.safetensors'
            raw=header if isinstance(header,bytes) else json.dumps(header).encode()
            target.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
            return read_f32(target,expected or {'x':[1]})
    def test_valid_little_endian_and_malformed_dtype_offsets_shapes(self):
        item={'dtype':'F32','shape':[1],'data_offsets':[0,4]}
        self.assertEqual(self.check({'x':item},struct.pack('<f',1.5)),{'x':[1.5]})
        for changed in [dict(item,dtype='F16'),dict(item,shape=[2]),dict(item,data_offsets=[0,8]),dict(item,data_offsets=[True,4])]:
            with self.assertRaises(ValueError):self.check({'x':changed},struct.pack('<f',1.5))
        with self.assertRaises(ValueError):self.check({'x':item},struct.pack('<f',float('nan')))
        with self.assertRaises(ValueError):self.check({'x':item},struct.pack('<f',1.5)+b'extra')
    def test_duplicate_keys_overlap_and_metadata_are_rejected(self):
        item={'dtype':'F32','shape':[1],'data_offsets':[0,4]}
        duplicate=('{"x":'+json.dumps(item)+',"x":'+json.dumps(item)+'}').encode()
        with self.assertRaises(ValueError):self.check(duplicate,struct.pack('<f',1))
        with self.assertRaises(ValueError):self.check({'x':item,'y':item},struct.pack('<f',1),{'x':[1],'y':[1]})
        with self.assertRaises(ValueError):self.check({'x':item,'__metadata__':{'bad':1}},struct.pack('<f',1))
