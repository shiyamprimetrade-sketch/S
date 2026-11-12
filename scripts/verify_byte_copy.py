"""Independent no-repair runtime acceptance of frozen byte-copy programs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from s_brain.checkpoint import load_manifest
from s_brain.coding_curriculum import read_plan,records,digest
from s_developer.copy_models import validate_adapter
from s_developer.providers import decode_proposal
from s_developer.sandbox import DockerRunner,prepare_image
from scripts.verify_coding_curriculum import run_cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--approve-runtime-download',action='store_true');p.add_argument('--require-target',action='store_true');a=p.parse_args()
    if not a.approve_runtime_download:p.error('Explicit official runtime preparation approval required')
    if a.report.exists():p.error('Preserve existing execution evidence')
    plan=read_plan(a.experiment/'plan.json');rows=records(plan);tasks=[r for r in rows if r['split']=='test']
    frozen=json.loads((a.experiment/'generated.json').read_text());base=load_manifest(Path(plan['base_checkpoint']))
    assert frozen['plan_sha256']==digest(plan) and frozen['dataset_sha256']==digest(rows)
    assert frozen['base_weights_sha256']==base.weights_sha256==plan['base_weights_sha256']
    assert frozen['repairs_allowed']==plan['repairs_allowed']==0
    assert [m['label'] for m in frozen['models']]==['base-unassisted','random-copy','learned-copy']
    image=prepare_image('python',True)['image'];runner=DockerRunner({'python':image})
    report={'experiment':plan['experiment'],'plan_sha256':digest(plan),'image':image,'repairs_performed':0,
            'automatic_promotion':False,'models':[]}
    for model in frozen['models']:
        if model['label']!='base-unassisted':
            path=a.experiment/('selected' if model['label']=='learned-copy' else 'random-control')
            assert validate_adapter(path,base)['weights_sha256']==model['adapter_sha256']
        assert [t['id'] for t in model['tasks']]==[t['id'] for t in tasks]
        item={'label':model['label'],'tasks':[]}
        for generated,task in zip(model['tasks'],tasks):
            assert generated['prompt_sha256']==digest(task['prompt'])
            row={'id':task['id'],'passed':False,'cases':[]}
            try:files=decode_proposal(generated['raw'],'python-cli')
            except (ValueError,TypeError):assert generated['format_passed'] is False
            else:
                assert generated['format_passed'] is True and files==generated['files'] and digest(files)==generated['source_sha256']
                row['cases']=run_cases(runner,files,plan,task)
                row['passed']=len(row['cases'])==plan['target']['cases_per_task'] and all(c['verdict']['passed'] for c in row['cases'])
            item['tasks'].append(row)
        item['tasks_correct']=sum(t['passed'] for t in item['tasks']);report['models'].append(item)
        a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'label':item['label'],'tasks_correct':item['tasks_correct'],'tasks':len(tasks),'repairs':0}),flush=True)
    candidate=report['models'][-1]['tasks_correct']
    report['target_met']=candidate>=plan['target']['minimum_tasks_correct']
    report['improved_over_controls']=candidate>max(m['tasks_correct'] for m in report['models'][:-1])
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('target_met','improved_over_controls','repairs_performed')}),flush=True)
    if a.require_target and not (report['target_met'] and report['improved_over_controls']):
        raise AssertionError('Predeclared no-repair learned-copy quality target not met; evidence preserved')


if __name__=='__main__':main()
