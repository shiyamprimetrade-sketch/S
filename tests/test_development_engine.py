"""Orchestration/security unit tests. Mocked runner results are not build evidence."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from continuum.storage import Store
from s_developer.specifications import Specifications
from s_developer.bundles import validate_files, fingerprint
from s_developer.providers import decode_proposal, TemplateProvider, provider, OllamaProvider
from s_developer.sandbox import DockerRunner, prepare_image, docker_env
from s_developer.models import install_demo, load_config, validate_native

ANSWERS={'purpose':'Print a greeting','users':'Me','features':'Show the project name','platform':'Windows',
         'data':'none','constraints':'Offline','acceptance':'Print the approved project description'}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.db=Path(self.temp.name)/'s.sqlite3';self.specs=Specifications(Store(self.db),{'execution_enabled':True})
        self.engine=self.specs.engine
        self.spec=self.specs.save(dict(name='Greeting',request='S develop application',answers=ANSWERS,approve=True))
        self.pid=self.spec['project_id']
    def generate(self,recipe='python-cli',provider='template'):
        rows=self.engine.artifacts(self.pid)
        return self.engine.generate(dict(project_id=self.pid,spec_revision=1,expected_artifact_revision=rows[0]['revision'] if rows else 0,
                                         recipe=recipe,provider=provider,consent=True))
    def build(self,revision,maximum=0):
        return self.engine.build(dict(project_id=self.pid,artifact_revision=revision,repair_provider='template',max_repairs=maximum,consent=True))
    def test_all_starters_generate_export_and_reopen_without_neural_claims(self):
        for recipe in ('static-site','python-cli','c-cli'):
            a=self.generate(recipe)
            self.assertEqual(a['provenance']['provider'],'template')
            self.assertFalse(a['provenance']['independently_generated_neural_code'])
            self.assertEqual(a['source_sha256'],fingerprint(a['files']))
            reopened=Specifications(Store(self.db)).engine.latest(self.pid)
            self.assertEqual(reopened,a)
            value=self.engine.archive(self.pid)
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(value['archive_base64']))) as archive:
                self.assertEqual(set(archive.namelist()),set(a['files'])|{'S-PROVENANCE.json'})
    def test_draft_new_revision_invalidates_generation_and_build(self):
        a=self.generate()
        self.specs.save(dict(project_id=self.pid,expected_revision=1,name='Greeting',request='S develop application',answers=ANSWERS))
        with self.assertRaisesRegex(ValueError,'Specification changed'):self.generate()
        with self.assertRaisesRegex(ValueError,'Specification changed'):self.build(a['revision'])
    def test_action_consent_and_host_opt_in_are_distinct(self):
        with self.assertRaises(PermissionError):
            self.engine.generate(dict(project_id=self.pid,spec_revision=1,expected_artifact_revision=0,recipe='python-cli',provider='template',consent=False))
        a=self.generate();self.engine.config['execution_enabled']=False
        with self.assertRaises(PermissionError):self.build(a['revision'])
    def test_injected_commands_or_provider_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.generate(dict(project_id=self.pid,spec_revision=1,expected_artifact_revision=0,recipe='python-cli',provider='template',consent=True,command='run me'))
        with self.assertRaises(PermissionError):provider('../../model',{})
        with self.assertRaises(PermissionError):provider('native',{})
    def test_no_host_execution_fallback_when_docker_missing(self):
        a=self.generate()
        with patch('s_developer.sandbox.shutil.which',return_value=None),patch('s_developer.sandbox.subprocess.Popen') as popen:
            result=self.build(a['revision'],2)
        self.assertEqual(result['final_status'],'blocked');self.assertEqual(len(result['attempts']),1)
        popen.assert_not_called()
    def test_bounded_repair_preserves_original_and_broken_checkpoints(self):
        a=self.generate();broken=a['files']['app.py'].replace('def describe():','def describe()')
        b=self.engine.edit(dict(project_id=self.pid,expected_artifact_revision=1,path='app.py',content=broken,consent=True))
        results=[{'status':'failed','output':'SyntaxError: expected colon'}, {'status':'toolchain_passed','output':'Synthetic runner result'}]
        with patch('s_developer.engine.DockerRunner.run',side_effect=results) as runner:result=self.build(b['revision'],1)
        self.assertEqual(runner.call_count,2);self.assertEqual(result['final_status'],'toolchain_passed')
        self.assertEqual(len(self.engine.artifacts(self.pid)),3)
        self.assertEqual(result['artifact']['files'],a['files'])
        self.assertEqual(self.engine.artifacts(self.pid)[1]['files']['app.py'],broken)
    def test_unmatched_repair_stops_and_budget_cannot_expand(self):
        a=self.generate()
        with patch('s_developer.engine.DockerRunner.run',return_value={'status':'failed','output':'Unknown failure'}) as runner:
            result=self.build(a['revision'],2)
        self.assertEqual(runner.call_count,1);self.assertEqual(result['final_status'],'repair_stopped')
        for maximum in (-1,3,True):
            with self.assertRaises(ValueError):self.build(a['revision'],maximum)
    def test_stale_source_edit_build_and_rollback_do_not_overwrite(self):
        a=self.generate();self.generate()
        with self.assertRaises(ValueError):self.build(a['revision'])
        with self.assertRaises(ValueError):
            self.engine.edit(dict(project_id=self.pid,expected_artifact_revision=1,path='app.py',content='print(1)',consent=True))
        restored=self.engine.restore(dict(project_id=self.pid,revision=1,expected_artifact_revision=2,consent=True))
        self.assertEqual(restored['revision'],3);self.assertEqual(restored['files'],a['files'])
    def test_generation_failure_does_not_replace_working_source(self):
        a=self.generate()
        with patch('s_developer.engine.provider') as selected:
            selected.return_value.provenance.return_value={'provider':'native','model_sha256':'fixture'}
            selected.return_value.generate.side_effect=ValueError('invalid model JSON')
            with self.assertRaisesRegex(ValueError,'no source replaced'):self.generate(provider='native')
        self.assertEqual(self.engine.latest(self.pid),a)
        self.assertEqual(self.engine.runs(self.pid)[0]['stage'],'generation')
    def test_requirements_changed_during_generation_do_not_publish_stale_source(self):
        def change_requirements(*args):
            self.specs.save(dict(project_id=self.pid,expected_revision=1,name='Changed',request='S develop application',answers=ANSWERS))
            return TemplateProvider().generate(self.spec,'python-cli')
        with patch('s_developer.engine.provider') as selected:
            selected.return_value.generate.side_effect=change_requirements
            with self.assertRaisesRegex(ValueError,'Specification changed'):self.generate()
        self.assertEqual(self.engine.artifacts(self.pid),[])
    def test_native_demo_install_is_local_checksum_verified_and_idempotent(self):
        result=install_demo(self.db);self.assertTrue(result['installed']);self.assertFalse(result['inference_verified'])
        config=load_config(self.db);manifest=validate_native(Path(config['native_checkpoint']))
        self.assertEqual(manifest.weights_sha256,result['weights_sha256'])
        self.assertEqual(install_demo(self.db)['weights_sha256'],result['weights_sha256'])
        weights=Path(config['native_checkpoint'])/'model.safetensors';weights.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):install_demo(self.db)


class BundleAndSandboxTests(unittest.TestCase):
    def test_paths_types_sizes_case_collisions_and_reserved_export_names(self):
        for name in ('../escape.py','/tmp/escape.py','C:/escape.py','a\\b.py','.env','CON.py','a/../../bad.py','S-PROVENANCE.json','a//b.py'):
            with self.assertRaises(ValueError):validate_files({'app.py':'','test_app.py':'',name:'bad'},'python-cli')
        for files in ({'app.py':'','test_app.py':'','APP.py':''},{'app.py':'x'*64001,'test_app.py':''},
                      {'app.py':None,'test_app.py':''},{'app.py':'','test_app.py':'','a.py':'','a.py/b.py':''}):
            with self.assertRaises(ValueError):validate_files(files,'python-cli')
    def test_model_cannot_choose_commands_or_loosen_permissions(self):
        for proposal in ('{"files":{"../escape.py":"x"}}','{"files":{},"command":"shell"}','```json {} ```'):
            with self.assertRaises(ValueError):decode_proposal(proposal,'python-cli')
    def test_container_plan_has_fixed_commands_digest_and_resource_restrictions(self):
        runner=DockerRunner({'python':'python@sha256:'+'a'*64})
        command=runner.command('/usr/bin/docker','s-fixture','python-cli',Path('/tmp/source'),Path('/tmp/driver'))
        for flag in ('--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
                     '--pull=never','--user=65534:65534','--pids-limit=64','--memory=512m','--log-driver=none'):
            self.assertIn(flag,command)
        self.assertNotIn('--privileged',command);self.assertNotIn('/var/run/docker.sock',' '.join(command))
        self.assertEqual(command[-3:],['-I','-S','/driver/run.py'])
        with self.assertRaises(PermissionError):DockerRunner({'python':'python:latest'}).command('docker','s','python-cli',Path('/s'),Path('/d'))
    def test_runtime_preparation_never_downloads_without_explicit_operator_flag(self):
        with patch('s_developer.sandbox.subprocess.run') as run:
            value=prepare_image('python',False);self.assertFalse(value['installed']);run.assert_not_called()
        with self.assertRaises(ValueError):prepare_image('evil/image',True)
    def test_host_credentials_and_docker_endpoint_not_forwarded(self):
        with patch.dict('os.environ',{'S_PRIVATE_TOKEN':'fixture','DOCKER_HOST':'tcp://attacker:2375','HTTPS_PROXY':'fixture'}):
            env=docker_env()
        for key in ('S_PRIVATE_TOKEN','DOCKER_HOST','HTTPS_PROXY'):self.assertNotIn(key,env)
    def test_external_provider_discloses_provenance_and_denies_cloud(self):
        engine=OllamaProvider('local:cloud')
        with patch('s_developer.providers.local_json',return_value={'models':[{'name':'local:cloud','digest':'fixture'}]}):
            with self.assertRaises(PermissionError):engine.provenance()
