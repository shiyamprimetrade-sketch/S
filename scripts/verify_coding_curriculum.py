"""Independent semantics of frozen neural outputs, in real restricted containers."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s_brain.checkpoint import load_manifest
from s_brain.coding_curriculum import digest, read_plan, records, oracle, case_inputs
from s_developer.providers import decode_proposal
from s_developer.capability import judge
from s_developer.sandbox import DockerRunner, prepare_image
from s_developer.literal_repair import LiteralRepairProvider


def run_cases(runner, files, plan, task):
    results=[]
    for x in case_inputs(plan,task):
        expected=oracle(task['operation'],task['constant'],x)
        raw=runner.run(files,'python-cli',timeout=10,evaluation_input={'x':x})
        verdict=judge(raw,expected)
        if verdict['status']=='blocked':raise RuntimeError('Isolation unavailable; no host execution fallback')
        results.append({'input':{'x':x},'expected':expected,'verdict':verdict,
                        'returncode':raw.get('returncode'),'output':raw.get('output','')})
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--approve-runtime-download',action='store_true')
    p.add_argument('--approve-requirements-literal-repair',action='store_true')
    a=p.parse_args()
    if not a.approve_runtime_download:p.error('Explicit official runtime preparation approval required')
    if a.report.exists():p.error('Report exists; preserve previous evidence')
    plan=read_plan(a.experiment/'plan.json')
    generated=json.loads((a.experiment/'generated.json').read_text())
    tasks=[r for r in records(plan) if r['split']=='test']
    repair_plan=None
    if a.approve_requirements_literal_repair:
        repair_plan=json.loads((a.experiment/'repair-plan.json').read_text())
        if repair_plan['max_repairs']!=1 or repair_plan['provider']!='requirements-literal':
            raise ValueError('Unsupported repair policy')
    assert generated['plan_sha256']==digest(plan)
    assert generated['dataset_sha256']==digest(records(plan))
    assert [m['label'] for m in generated['models']]==['old-demo','random-control','trained-candidate']
    image=prepare_image('python',True)['image']
    runner=DockerRunner({'python':image})
    report={'experiment':plan['experiment'],'plan_sha256':digest(plan),'image':image,
            'model_output_replay':'required separately; this runner does not attest inference',
            'automatic_promotion':False,'models':[]}
    if repair_plan:report['repair_policy_sha256']=digest(repair_plan)
    for model in generated['models']:
        checkpoint=(Path('s_developer/resources/native-demo') if model['label']=='old-demo' else
                    a.experiment/'checkpoints'/('selected' if model['label']=='trained-candidate' else 'random-control'))
        assert load_manifest(checkpoint).weights_sha256==model['weights_sha256']
        rows=model['cases']
        assert [r['id'] for r in rows]==[r['id'] for r in tasks]
        result={'label':model['label'],'weights_sha256':model['weights_sha256'],'tasks':[]}
        for row,task in zip(rows,tasks):
            assert row['prompt_sha256']==digest(task['prompt'])
            case={'id':task['id'],'format_passed':row['format_passed'],'cases':[],'passed':False}
            try:files=decode_proposal(row['raw'],'python-cli')
            except (ValueError,TypeError):
                assert row['format_passed'] is False
            else:
                assert row['format_passed'] is True and files==row['files'] and digest(files)==row['source_sha256']
                case['cases']=run_cases(runner,files,plan,task)
                case['passed']=all(c['verdict']['passed'] for c in case['cases'])
                case['initial_passed']=case['passed']
                if not case['passed'] and repair_plan:
                    repair=LiteralRepairProvider()
                    try:
                        # Only the original requirement reaches repair. No
                        # case inputs, expected answers or oracle diagnostics.
                        spec={'answers':{'purpose':task['prompt']}}
                        patched=repair.repair(spec,'python-cli',files,'')
                        repaired_cases=run_cases(runner,patched,plan,task)
                        case['repair']={'files':patched,'provenance':repair.provenance(),
                                        'cases':repaired_cases}
                        case['passed']=all(c['verdict']['passed'] for c in repaired_cases)
                    except (ValueError,SyntaxError) as error:
                        case['repair_stopped']=str(error)
            result['tasks'].append(case)
        result['initial_tasks_correct']=sum(r.get('initial_passed',r['passed']) for r in result['tasks'])
        result['tasks_correct']=sum(r['passed'] for r in result['tasks'])
        report['models'].append(result)
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='tasks'}),flush=True)
    candidate=report['models'][-1]['initial_tasks_correct']
    report['target_met']=candidate>=plan['target']['minimum_tasks_correct']
    report['improved_over_controls']=candidate>max(r['initial_tasks_correct'] for r in report['models'][:-1])
    if repair_plan:
        report['repair_target_met']=report['models'][-1]['tasks_correct']>=repair_plan['minimum_final_tasks_correct']
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    # Experiment failure is evidence, not broken infrastructure. CI must verify
    # replay, isolation and oracle integrity; quality is recorded independently.
    print(json.dumps({'target_met':report['target_met'],'improved_over_controls':report['improved_over_controls'],
                      'candidate_tasks_correct':candidate,'final_tasks_correct':report['models'][-1]['tasks_correct'],
                      'repair_target_met':report.get('repair_target_met'),'total_tasks':len(tasks)}),flush=True)


if __name__=='__main__':main()
