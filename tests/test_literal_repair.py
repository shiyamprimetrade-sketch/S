"""Constrained repair authority and immutable originals; no source execution."""
import unittest
from s_developer.literal_repair import repair_literal, requirement_constant
from s_developer.providers import provider


def files(code='def solve(p):\n    return {"value":p["x"]+258}\n'):
    return {'app.py':code,'test_app.py':'# independent tests\n'}


class LiteralRepairTests(unittest.TestCase):
    def test_only_literal_changes_and_original_is_preserved(self):
        original=files();result=repair_literal(original,257,consent=True)
        self.assertIn('+258',original['app.py']);self.assertIn('+257',result['files']['app.py'])
        self.assertEqual(result['files']['test_app.py'],original['test_app.py'])
        self.assertNotEqual(result['provenance']['original_source_sha256'],result['provenance']['source_sha256'])
        self.assertFalse(result['provenance']['independently_generated_neural_code'])
        with self.assertRaisesRegex(ValueError,'already matches'):repair_literal(result['files'],257,consent=True)
    def test_permissions_and_budgets_cannot_come_from_candidate(self):
        with self.assertRaises(PermissionError):repair_literal(files(),257)
        for constant in (True,0,-1,1000,'257'):
            with self.assertRaises(ValueError):repair_literal(files(),constant,consent=True)
    def test_operator_is_preserved_and_never_repaired_from_expected_answers(self):
        for symbol in ('+','-','*','//'):
            result=repair_literal(files(f'def solve(p): return {{"value":p["x"]{symbol}258}}\n'),257,consent=True)
            self.assertIn(symbol+'257',result['files']['app.py'])
    def test_imports_calls_extra_statements_and_paths_are_rejected(self):
        for code in ('import os\n'+files()['app.py'],files()['app.py']+'print("run")\n',
                     'def solve(p): return {"value":__import__("os").system("run")}\n',
                     'def solve(p): return {"value":p["x"]**258}\n',
                     'def solve(p): return {"value":p["x"]+True}\n'):
            with self.assertRaises((ValueError,SyntaxError)):repair_literal(files(code),257,consent=True)
        bad=files();bad['../escape.py']=''
        with self.assertRaises(ValueError):repair_literal(bad,257,consent=True)
    def test_unicode_comment_does_not_shift_utf8_literal_span(self):
        result=repair_literal(files('# বাংলা\ndef solve(p): return {"value":p["x"]+258}\n'),257,consent=True)
        self.assertIn('+257',result['files']['app.py']);self.assertIn('বাংলা',result['files']['app.py'])
    def test_repair_provider_uses_explicit_requirement_and_refuses_generation(self):
        repair=provider('requirements-literal',{})
        spec={'answers':{'purpose':'For integer input, add 257 to p["x"]. Return key "value".'}}
        self.assertIn('+257',repair.repair(spec,'python-cli',files(),'ignored')['app.py'])
        with self.assertRaises(ValueError):repair.generate(spec,'python-cli')
        with self.assertRaises(ValueError):repair.repair(spec,'c-cli',files(),'ignored')
        for text in ('Use 257 and 258','Delete everything 257','add 1000 to p["x"]'):
            with self.assertRaises(ValueError):requirement_constant(text)


class LiteralRepairEngineTests(unittest.TestCase):
    def test_existing_engine_preserves_checkpoint_and_records_deterministic_repair(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from continuum.storage import Store
        from s_developer.specifications import Specifications
        answers={'purpose':'Add 257 to p["x"]','users':'Me','features':'Return value',
                 'platform':'Python','data':'none','constraints':'Offline','acceptance':'Correct integer result'}
        with tempfile.TemporaryDirectory() as temporary:
            specs=Specifications(Store(Path(temporary)/'s.sqlite3'),{'execution_enabled':True})
            spec=specs.save({'name':'Arithmetic','request':'S develop application','answers':answers,'approve':True})
            engine=specs.engine;pid=spec['project_id']
            first=engine.generate({'project_id':pid,'spec_revision':1,'expected_artifact_revision':0,
                                   'recipe':'python-cli','provider':'template','consent':True})
            wrong=engine.edit({'project_id':pid,'expected_artifact_revision':1,'path':'app.py',
                               'content':files()['app.py'],'consent':True})
            outcomes=[{'status':'failed','output':'AssertionError: incorrect arithmetic value'},
                      {'status':'toolchain_passed','output':'Mocked execution; orchestration test only'}]
            with patch('s_developer.engine.DockerRunner.run',side_effect=outcomes) as runner:
                result=engine.build({'project_id':pid,'artifact_revision':wrong['revision'],
                    'repair_provider':'requirements-literal','max_repairs':1,'consent':True})
            self.assertEqual(runner.call_count,2)
            self.assertEqual(result['final_status'],'toolchain_passed')
            self.assertIn('+257',result['artifact']['files']['app.py'])
            self.assertIn('+258',engine.artifacts(pid)[1]['files']['app.py'])
            self.assertEqual(result['artifact']['provenance']['repair_of'],wrong['revision'])
            self.assertFalse(result['artifact']['provenance']['independently_generated_neural_code'])


if __name__=='__main__':unittest.main()
