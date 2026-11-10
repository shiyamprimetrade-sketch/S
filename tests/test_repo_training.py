import tempfile,unittest
from pathlib import Path
from continuum.repo_training import prepare
class SourceTrainingTests(unittest.TestCase):
    def test_split_and_sensitive_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(50):
                (root/f'module_{i}.py').write_text('def function_'+str(i)+'():\n    return '+str(i)+'\n'+'#' * 150)
            (root/'credentials.py').write_text('sk-'+'A'*30+'\n'+'#'*150)
            rows,m=prepare(root)
            self.assertEqual(len(rows),50)
            self.assertIn('credentials.py',m['excluded_sensitive'])
            self.assertTrue(any(r['split']=='evaluation' for r in rows))
            self.assertTrue(any(r['split']=='training' for r in rows))
            self.assertEqual({r['path'] for r in rows if r['split']=='training'} & {r['path'] for r in rows if r['split']=='evaluation'},set())
    def test_no_eligible_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):prepare(Path(tmp))
    def test_invalid_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):prepare(Path(tmp),0)
if __name__=='__main__':unittest.main()
