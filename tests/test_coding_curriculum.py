"""Curriculum integrity and host-only decoder controls; no generated execution."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from s_brain.coding_curriculum import records, read_plan, validate_splits, oracle, digest
from s_developer.models import configure_decoding, configure_runtime, install_demo, load_config
from s_developer.providers import provider

PLAN = Path(__file__).resolve().parents[1]/'experiments/coding-curriculum-v1/plan.json'


class CurriculumTests(unittest.TestCase):
    def test_disjoint_constants_prompts_and_completion_hashes(self):
        rows=records(read_plan(PLAN))
        self.assertEqual({s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
                         {'train':240,'validation':8,'test':8})
        for field in ('prompt','completion'):
            sets={s:{r[field] for r in rows if r['split']==s} for s in ('train','validation','test')}
            self.assertFalse(sets['train'] & sets['test'])
            self.assertFalse(sets['train'] & sets['validation'])
            self.assertFalse(sets['validation'] & sets['test'])
        self.assertEqual(digest(rows),digest(records(read_plan(PLAN))))
    def test_followup_uses_fresh_constants_and_explicit_answer_boundary(self):
        plan=read_plan(PLAN.parent.parent/'coding-curriculum-v2/plan.json')
        rows=records(plan)
        self.assertEqual(sum(r['split']=='train' for r in rows),1104)
        original_tests={r['constant'] for r in records(read_plan(PLAN)) if r['split']=='test'}
        self.assertFalse(original_tests & {r['constant'] for r in rows})
        self.assertTrue(all(r['prompt'].endswith('\nAnswer:\n') for r in rows))

    def test_preserved_datasets_and_selected_models_match_the_training_ledger(self):
        from s_brain.checkpoint import load_manifest
        for version in (1,2,3):
            root=PLAN.parent.parent/f'coding-curriculum-v{version}'
            plan=read_plan(root/'plan.json');rows=records(plan)
            archived=[json.loads(line) for line in gzip.decompress((root/'dataset.jsonl.gz').read_bytes()).splitlines()]
            self.assertEqual(archived,rows)
            training=json.loads((root/'training-report.json').read_text())
            self.assertEqual(training['dataset_sha256'],digest(rows))
            self.assertEqual(training['plan_sha256'],digest(plan))
            manifest=load_manifest(root/'checkpoints/selected')
            self.assertEqual(manifest.weights_sha256,training['selected_weights_sha256'])
            self.assertEqual(manifest.step,training['selected_step'])
            self.assertTrue(manifest.extras['independently_initialized'])

    def test_split_leakage_is_rejected(self):
        rows=records(read_plan(PLAN));bad=copy.deepcopy(rows[0]);bad['split']='test'
        with self.assertRaisesRegex(ValueError,'leakage'):validate_splits(rows+[bad])
    def test_oracle_handles_negative_floor_and_zero_without_eval(self):
        self.assertEqual(oracle('floor_divide',7,-21),{'value':-3})
        self.assertEqual(oracle('floor_divide',13,-21),{'value':-2})
        self.assertEqual(oracle('multiply',13,0),{'value':0})
        with self.assertRaises(ValueError):oracle('__import__',7,0)
    def test_decoder_selection_is_host_only_and_portable_guarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3'
            with self.assertRaisesRegex(ValueError,'Install'):configure_decoding(db,'greedy-json')
            install_demo(db)
            configure_decoding(db,'greedy-json')
            self.assertEqual(provider('native',load_config(db)).decoding,'greedy-json')
            configure_runtime(db,'stdlib')
            self.assertEqual(load_config(db)['native_decoding'],'sampled')
            with self.assertRaisesRegex(ValueError,'torch'):configure_decoding(db,'greedy-json')
            with self.assertRaises(ValueError):configure_decoding(db,'shell')
            config=load_config(db);config['native_decoding']='greedy-json'
            (Path(temporary)/'s-developer-host.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'torch'):load_config(db)


if __name__=='__main__':unittest.main()
