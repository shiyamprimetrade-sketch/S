"""Independent semantics of frozen neural outputs, in real restricted containers."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s_brain.checkpoint import load_manifest
from s_brain.coding_curriculum import digest, read_plan, records, oracle
from s_developer.providers import decode_proposal
from s_developer.capability import judge
from s_developer.sandbox import DockerRunner, prepare_image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--approve-runtime-download',action='store_true')
    a=p.parse_args()
    if not a.approve_runtime_download:p.error('Explicit official runtime preparation approval required')
    if a.report.exists():p.error('Report exists; preserve previous evidence')
    plan=read_plan(a.experiment/'plan.json')
    generated=json.loads((a.experiment/'generated.json').read_text())
    tasks=[r for r in records(plan) if r['split']=='test']
    assert generated['plan_sha256']==digest(plan)
    assert generated['dataset_sha256']==digest(records(plan))
    assert [m['label'] for m in generated['models']]==['old-demo','random-control','trained-candidate']
    image=prepare_image('python',True)['image']
    runner=DockerRunner({'python':image})
    report={'experiment':plan['experiment'],'plan_sha256':digest(plan),'image':image,
            'model_output_replay':'required separately; this runner does not attest inference',
            'automatic_promotion':False,'models':[]}
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
                for x in (-21,0,19):
                    expected=oracle(task['operation'],task['constant'],x)
                    raw=runner.run(files,'python-cli',timeout=10,evaluation_input={'x':x})
                    verdict=judge(raw,expected)
                    if verdict['status']=='blocked':raise RuntimeError('Isolation unavailable; no host execution fallback')
                    case['cases'].append({'input':{'x':x},'expected':expected,'verdict':verdict,
                                          'returncode':raw.get('returncode'),'output':raw.get('output','')})
                case['passed']=all(c['verdict']['passed'] for c in case['cases'])
            result['tasks'].append(case)
        result['tasks_correct']=sum(r['passed'] for r in result['tasks'])
        report['models'].append(result)
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='tasks'}),flush=True)
    candidate=report['models'][-1]['tasks_correct']
    report['target_met']=candidate>=plan['target']['minimum_tasks_correct']
    report['improved_over_controls']=candidate>max(r['tasks_correct'] for r in report['models'][:-1])
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    # Experiment failure is evidence, not broken infrastructure. CI must verify
    # replay, isolation and oracle integrity; quality is recorded independently.
    print(json.dumps({'target_met':report['target_met'],'improved_over_controls':report['improved_over_controls'],
                      'candidate_tasks_correct':candidate,'total_tasks':len(tasks)}),flush=True)


if __name__=='__main__':main()
