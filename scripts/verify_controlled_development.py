"""Actual Docker acceptance checks. No mock runner; failure is a failing CI job."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continuum.storage import Store
from s_developer.specifications import Specifications
from s_developer.sandbox import prepare_image

ANSWERS={'purpose':'Display a welcome message','users':'Visitors','features':'Show the project name',
         'platform':'Local test runtime','data':'none','constraints':'Offline',
         'acceptance':'Display approved project description'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--approve-runtime-download',action='store_true');args=parser.parse_args()
    if not args.approve_runtime_download:parser.error('Explicit approval to prepare official runtime images is required')
    images={key:prepare_image(key,True)['image'] for key in ('python','gcc')}
    report={'images':images,'tests':[],'mocked_execution':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        specifications=Specifications(Store(root/'s.sqlite3'),{'execution_enabled':True,'images':images})
        engine=specifications.engine
        def project(recipe):
            spec=specifications.save(dict(name='Welcome fixture',request='S develop website' if recipe=='static-site' else 'S develop application',answers=ANSWERS,approve=True))
            a=engine.generate(dict(project_id=spec['project_id'],spec_revision=1,expected_artifact_revision=0,recipe=recipe,provider='template',consent=True))
            return a
        def build(a,repairs=0):
            result=engine.build(dict(project_id=a['project_id'],artifact_revision=a['revision'],repair_provider='template',max_repairs=repairs,consent=True))
            report['latest_run']=result
            args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
            return result
        for recipe in ('static-site','python-cli','c-cli'):
            a=project(recipe);result=build(a)
            assert result['final_status']=='toolchain_passed',result
            report['tests'].append({'case':recipe,'result':result})
        a=project('python-cli')
        broken=engine.edit(dict(project_id=a['project_id'],expected_artifact_revision=a['revision'],path='app.py',
            content=a['files']['app.py'].replace('def describe():','def describe()'),consent=True))
        repaired=build(broken,1)
        assert len(repaired['attempts'])==2,repaired
        assert repaired['attempts'][0]['result']['status']=='failed',repaired
        assert 'SyntaxError' in repaired['attempts'][0]['result']['output'],repaired
        assert repaired['final_status']=='toolchain_passed',repaired
        assert len(engine.artifacts(a['project_id']))==3
        report['tests'].append({'case':'actual-python-syntax-error-and-one-repair','result':repaired})
        sentinel=root/'host-sentinel.txt';sentinel.write_text('private host fixture')
        a=project('python-cli')
        attack='''import os, socket, unittest
class Boundaries(unittest.TestCase):
    def test_host_file_is_not_mounted(self):
        with self.assertRaises(FileNotFoundError):open(HOST_SENTINEL).read()
    def test_source_mount_is_read_only(self):
        with self.assertRaises(OSError):open('/src/app.py','w').write('escape')
    def test_host_environment_is_not_inherited(self):
        self.assertNotIn('S_HOST_SECRET',os.environ)
    def test_network_is_disabled(self):
        with socket.socket() as s:
            s.settimeout(2)
            with self.assertRaises(OSError):s.connect(('1.1.1.1',443))
'''.replace('HOST_SENTINEL',repr(str(sentinel)))
        secure=engine.edit(dict(project_id=a['project_id'],expected_artifact_revision=1,path='test_app.py',content=attack,consent=True))
        os.environ['S_HOST_SECRET']='synthetic-not-a-real-credential'
        result=build(secure)
        assert result['final_status']=='toolchain_passed',result
        assert sentinel.read_text()=='private host fixture'
        assert engine.latest(a['project_id'])['files']['app.py']==a['files']['app.py']
        report['tests'].append({'case':'real-container-host-path-env-source-write-network-denial','result':result})
        # An untrusted test can return a misleading success code: that is why
        # S labels toolchain status separately from business acceptance.
        a=project('python-cli')
        flood=engine.edit(dict(project_id=a['project_id'],expected_artifact_revision=1,path='test_app.py',
                              content='print("x"*1000000)\n',consent=True))
        result=build(flood)
        assert result['final_status']=='failed',result
        assert result['attempts'][0]['result']['output_limit_exceeded'],result
        report['tests'].append({'case':'real-container-output-budget-enforced','result':result})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for case in report['tests']:
        print(json.dumps({'case':case['case'],'final_status':case['result']['final_status'],
            'checks':[{'status':a.get('result',{}).get('status'),'returncode':a.get('result',{}).get('returncode'),
                       'output':a.get('result',{}).get('output','')[:2000]}
                      for a in case['result']['attempts']]}))
    print('PASS: actual Docker website HTTP, Python tests, C compilation/run, bounded repair and isolation checks')


if __name__=='__main__':main()
