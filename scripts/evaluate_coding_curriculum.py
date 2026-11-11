"""Freeze actual model output, or replay it byte-for-byte without running source."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from s_brain.checkpoint import load_weights
from s_brain.coding_curriculum import digest, records, read_plan
from s_brain.coding_decode import greedy_json
from s_developer.providers import decode_proposal


def generate(plan, checkpoints):
    torch.set_num_threads(plan['threads'])
    rows = [r for r in records(plan) if r['split'] == 'test']
    result = {'experiment':plan['experiment'], 'plan_sha256':digest(plan),
              'dataset_sha256':digest(records(plan)), 'decoder':plan['decoder'],
              'max_new_tokens':plan['max_new_tokens'], 'semantic_execution':'not_assessed',
              'automatic_promotion':False, 'models':[]}
    for label, checkpoint in checkpoints.items():
        model, manifest = load_weights(checkpoint, 'cpu')
        report = {'label':label, 'weights_sha256':manifest.weights_sha256,
                  'training_steps':manifest.step, 'cases':[]}
        for row in rows:
            raw = greedy_json(model, row['prompt'], plan['max_new_tokens'])
            case = {'id':row['id'], 'prompt_sha256':digest(row['prompt']), 'raw':raw}
            try:
                files = decode_proposal(raw, 'python-cli')
                case.update(format_passed=True, files=files, source_sha256=digest(files))
            except (ValueError, TypeError) as error:
                case.update(format_passed=False, diagnostic=str(error))
            report['cases'].append(case)
        result['models'].append(report)
        print(json.dumps({'model':label, 'weights_sha256':manifest.weights_sha256,
                          'format_passed':sum(r['format_passed'] for r in report['cases']),
                          'tasks':len(rows), 'semantic_execution':'not_assessed'}), flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',type=Path)
    a=p.parse_args()
    if (a.output is None)==(a.verify is None):p.error('Choose exactly --output or --verify')
    if a.output and a.output.exists():p.error('Output exists; preserve previous evaluation')
    root=a.experiment
    plan=read_plan(root/'plan.json')
    checkpoint_root=root/'checkpoints'
    checkpoints={'old-demo':Path('s_developer/resources/native-demo'),
                 'random-control':checkpoint_root/'random-control',
                 'trained-candidate':checkpoint_root/'selected'}
    training=json.loads((root/'training-report.json').read_text())
    if training['plan_sha256']!=digest(plan) or training['dataset_sha256']!=digest(records(plan)):
        raise RuntimeError('Training plan/dataset fingerprint mismatch')
    result=generate(plan,checkpoints)
    if result['models'][-1]['weights_sha256']!=training['selected_weights_sha256']:
        raise RuntimeError('Selected checkpoint differs from the frozen training selection')
    if a.verify:
        if json.loads(a.verify.read_text()) != result:raise RuntimeError('Frozen neural outputs differ from replay')
        print('PASS: all frozen outputs reproduced exactly from checksum-verified neural checkpoints')
    else:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
