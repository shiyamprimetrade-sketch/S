"""Actual Docker tests of the independent oracle, never neural capability evidence."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from s_developer.capability import CapabilityEvaluator
from s_developer.sandbox import prepare_image

CODE={
 'positive-total':'def solve(p):\n    return {"total":sum(v for v in p["values"] if v>0)}\n',
 'unique-words':'def solve(p):\n    return {"words":sorted(set(p["text"].lower().split()))}\n',
 'palindrome':'def solve(p):\n    s=p["text"].replace(" ","").lower()\n    return {"palindrome":s==s[::-1]}\n',
}


class Fixture:
    id='test-fixture'
    def __init__(self,mode):self.mode=mode
    def provenance(self):return {'provider':'test-fixture','mode':self.mode,'independently_generated_neural_code':False}
    def files(self,code):return {'app.py':code,'test_app.py':'raise RuntimeError("Generated tests must not be used as the host oracle")\n'}
    def generate(self,spec,recipe):
        code=CODE[spec['name']]
        if self.mode=='exit-zero':code='import os\nos._exit(0)\n'
        if self.mode=='constant-answer':code='def solve(p): return {"total":5}\n'
        if self.mode=='repair':code='def solve(p): return {}\n'
        return self.files(code)
    def repair(self,spec,recipe,files,diagnostic):return self.files(CODE[spec['name']])


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--approve-runtime-download',action='store_true');a=p.parse_args()
    if not a.approve_runtime_download:p.error('Explicit official runtime preparation approval required')
    image=prepare_image('python',True)['image']
    config={'execution_enabled':True,'images':{'python':image}}
    reports=[];a.report.parent.mkdir(parents=True,exist_ok=True)
    for mode in ('correct','exit-zero','constant-answer','repair'):
        fixture=Fixture(mode)
        # Only source provider is a labelled fixture. Docker execution and
        # host-side comparisons are real, unmocked and subject to isolation.
        with patch('s_developer.capability.provider',return_value=fixture):
            result=CapabilityEvaluator(config).evaluate('fixture',True,True,1 if mode=='repair' else 0)
        reports.append(result);a.report.write_text(json.dumps(reports,indent=2),encoding='utf-8')
        if mode in ('correct','repair'):assert result['suite_passed'],result
        else:assert not result['suite_passed'] and result['final_tasks_correct']==0,result
        if mode=='repair':assert result['initial_tasks_correct']==0 and result['final_tasks_correct']==3,result
        assert not result['promotion_eligible']
        print(json.dumps({'fixture':mode,'independent_tasks_correct':result['final_tasks_correct'],
                          'initial_tasks_correct':result['initial_tasks_correct'],'assessed':result['tasks_fully_assessed'],
                          'suite_passed':result['suite_passed'],'provider':result['provider'],'image':image}))
    print('PASS: real containers accept correct functions, reject exit-zero/constant-answer cheating, and verify repaired functions')


if __name__=='__main__':main()
