"""Real temporary baseline installs; private data never enters fixtures."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from continuum.storage import Store

SOURCE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('upgrade_developer',SOURCE/'scripts'/'upgrade_developer.py')
upgrade=importlib.util.module_from_spec(spec);spec.loader.exec_module(upgrade)


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.app=Path(self.temp.name)/'app';self.app.mkdir()
        # Use a known reviewed source baseline, not arbitrary installed code.
        for name in [*upgrade.KNOWN,'continuum/storage.py']:
            result=subprocess.run(['git','show','efd82545b3ae6a5b5d9ab801877730c4322c4418:'+name],cwd=SOURCE,capture_output=True)
            if result.returncode==0:
                target=self.app/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(result.stdout)
        (self.app/'s-brain-host.json').write_text('{"enabled":false}')
        self.store=Store(self.app/'data'/'continuum.sqlite3')
        self.project=self.store.create_project('Synthetic continuity fixture')

    def test_upgrade_backup_and_rollback_preserve_database_and_config(self):
        original=(self.app/'continuum'/'server.py').read_bytes()
        config=(self.app/'s-brain-host.json').read_bytes()
        plan=upgrade.plan(self.app);self.assertTrue(any(x['change'] for x in plan))
        self.assertEqual((self.app/'continuum'/'server.py').read_bytes(),original)
        result=upgrade.apply(self.app)
        self.assertGreater(result['changed'],0)
        self.assertEqual((self.app/'continuum'/'server.py').read_bytes(),(SOURCE/'continuum'/'server.py').read_bytes())
        self.assertEqual(self.store.get_project(self.project['id'])['name'],'Synthetic continuity fixture')
        backup=Path(result['backup'])
        self.assertEqual(Store(backup/'continuum.sqlite3').get_project(self.project['id']),self.project)
        self.assertEqual((self.app/'s-brain-host.json').read_bytes(),config)
        self.assertEqual(upgrade.apply(self.app)['changed'],0)
        upgrade.rollback(self.app,backup)
        self.assertEqual((self.app/'continuum'/'server.py').read_bytes(),original)
        self.assertFalse((self.app/'web'/'developer.html').exists())
        self.assertEqual((self.app/'s-brain-host.json').read_bytes(),config)

    def test_unknown_local_changes_refuse_before_any_mutation(self):
        target=self.app/'continuum'/'server.py';target.write_text('local custom server')
        with self.assertRaisesRegex(ValueError,'Unrecognized local changes'):upgrade.apply(self.app)
        self.assertEqual(target.read_text(),'local custom server')
        self.assertFalse((self.app/'upgrade-backups').exists())

    def test_interrupted_copy_restores_previous_code(self):
        original=(self.app/'continuum'/'server.py').read_bytes()
        real=upgrade.atomic_copy
        calls=[]
        def fail_once(source,target):
            calls.append(target)
            if len(calls)==3:raise OSError('Synthetic copy failure')
            real(source,target)
        with patch.object(upgrade,'atomic_copy',side_effect=fail_once):
            with self.assertRaisesRegex(OSError,'Synthetic copy failure'):upgrade.apply(self.app)
        self.assertEqual((self.app/'continuum'/'server.py').read_bytes(),original)
        self.assertFalse((self.app/'web'/'developer.html').exists())

    def test_post_upgrade_edits_prevent_rollback_overwrite(self):
        result=upgrade.apply(self.app)
        target=self.app/'web'/'developer.html';target.write_text('new user changes')
        with self.assertRaisesRegex(ValueError,'Post-upgrade edits'):upgrade.rollback(self.app,result['backup'])
        self.assertEqual(target.read_text(),'new user changes')
