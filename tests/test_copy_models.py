"""Local installer/provenance boundaries, without importing optional torch."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from s_brain.checkpoint import load_manifest
from s_brain.coding_curriculum import read_plan,records,digest
from s_developer.models import install_native,load_config,save_config
from s_developer.copy_models import validate_adapter,install_copy_adapter
from s_developer.providers import provider

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/learned-byte-copy-v1'
BASE=ROOT/'experiments/coding-curriculum-v3/checkpoints/selected'


class CopyModelTests(unittest.TestCase):
    def test_install_is_explicit_checksum_bound_idempotent_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3'
            with self.assertRaises(ValueError):install_copy_adapter(db,EXP/'selected')
            installed=install_native(db,BASE);before=load_config(db)
            result=install_copy_adapter(db,EXP/'selected');config=load_config(db)
            self.assertEqual(config['native_checkpoint'],before['native_checkpoint'])
            self.assertEqual(install_copy_adapter(db,EXP/'selected')['adapter_weights_sha256'],result['adapter_weights_sha256'])
            self.assertFalse(result['inference_verified']);self.assertFalse(result['automatic_default_change'])
            provenance=provider('native-copy',config).provenance()
            self.assertEqual(provenance['provider'],'native-copy');self.assertEqual(provenance['model_sha256'],installed['weights_sha256'])
            save_config(db,{'native_backend':'stdlib'})
            with self.assertRaises(PermissionError):provider('native-copy',load_config(db))
    def test_span_adapter_install_exposes_its_controller_kind(self):
        span=ROOT/'experiments/learned-span-copy-v1'
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3';install_native(db,BASE)
            result=install_copy_adapter(db,span/'selected')
            provenance=provider('native-copy',load_config(db)).provenance()
            self.assertEqual(provenance['copy_kind'],'s-learned-span-copy')
            self.assertEqual(result['adapter_weights_sha256'],provenance['copy_adapter_sha256'])
            plan=read_plan(span/'plan.json');rows=records(plan)
            original=[json.loads(r) for r in gzip.decompress((span/'dataset.jsonl.gz').read_bytes()).splitlines()]
            self.assertEqual(rows,original)
            self.assertEqual(plan['repairs_allowed'],0);self.assertEqual(plan['copy_actions_budget'],1)
            token_constants={r['constant'] for r in records(read_plan(EXP/'plan.json'))}
            self.assertFalse(token_constants & {r['constant'] for r in rows})

    def test_unknown_adapter_origin_is_not_labelled_independent_neural_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3';install_native(db,BASE);install_copy_adapter(db,EXP/'selected')
            config=load_config(db);manifest=Path(config['native_copy_adapter'])/'manifest.json'
            value=json.loads(manifest.read_text());value['independently_initialized']=False
            manifest.write_text(json.dumps(value))
            self.assertFalse(provider('native-copy',config).provenance()['independently_generated_neural_code'])

    def test_wrong_base_random_control_and_corruption_are_refused(self):
        other=ROOT/'s_developer/resources/native-demo'
        with self.assertRaises(ValueError):validate_adapter(EXP/'selected',load_manifest(other))
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3';install_native(db,BASE)
            with self.assertRaises(ValueError):install_copy_adapter(db,EXP/'random-control')
            result=install_copy_adapter(db,EXP/'selected');config=load_config(db)
            (Path(config['native_copy_adapter'])/'adapter.safetensors').write_bytes(b'corruption')
            with self.assertRaises(ValueError):install_copy_adapter(db,EXP/'selected')
    def test_exact_dataset_and_model_are_frozen_before_test(self):
        plan=read_plan(EXP/'plan.json');rows=records(plan)
        archived=[json.loads(r) for r in gzip.decompress((EXP/'dataset.jsonl.gz').read_bytes()).splitlines()]
        self.assertEqual(rows,archived)
        training=json.loads((EXP/'training-report.json').read_text())
        self.assertEqual(training['dataset_sha256'],digest(rows));self.assertEqual(training['plan_sha256'],digest(plan))
        self.assertFalse(training['test_generation_started']);self.assertEqual(plan['repairs_allowed'],0)
        self.assertEqual(validate_adapter(EXP/'selected',load_manifest(BASE))['weights_sha256'],training['selected_weights_sha256'])
        old=set()
        for version in (1,2,3):old|={r['constant'] for r in records(read_plan(ROOT/f'experiments/coding-curriculum-v{version}/plan.json'))}
        self.assertFalse(old & {r['constant'] for r in rows})


if __name__=='__main__':unittest.main()
