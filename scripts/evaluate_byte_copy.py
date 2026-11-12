"""Freeze or replay prompt-only learned-copy generation; never executes source."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from s_brain.checkpoint import load_weights
from s_brain.coding_curriculum import read_plan,records,digest
from s_brain.coding_decode import greedy_json
from s_brain.byte_copy import generate_copy
from s_brain.span_copy import generate_span
from s_developer.copy_models import load_adapter
from s_developer.providers import decode_proposal


def generate(root):
    plan=read_plan(root/'plan.json');rows=records(plan)
    report=json.loads((root/'training-report.json').read_text())
    assert report['plan_sha256']==digest(plan) and report['dataset_sha256']==digest(rows)
    torch.set_num_threads(2)
    base,manifest=load_weights(Path(plan['base_checkpoint']))
    assert manifest.weights_sha256==plan['base_weights_sha256']
    output={'experiment':plan['experiment'],'plan_sha256':digest(plan),'dataset_sha256':digest(rows),
            'base_weights_sha256':manifest.weights_sha256,'generation_mechanism':plan.get('adapter_kind','s-learned-byte-copy'),'repairs_allowed':0,'semantic_execution':'not_assessed',
            'automatic_promotion':False,'models':[]}
    for label,path in [('base-unassisted',None),('random-copy',root/'random-control'),('learned-copy',root/'selected')]:
        adapter,meta=load_adapter(path,manifest) if path else (None,None)
        if label=='learned-copy':assert meta['weights_sha256']==report['selected_weights_sha256']
        model={'label':label,'adapter_sha256':meta['weights_sha256'] if meta else None,'tasks':[]}
        for row in (r for r in rows if r['split']=='test'):
            # Neither the authored completion nor structured constant is passed.
            item={'id':row['id'],'prompt_sha256':digest(row['prompt']),'raw':'','format_passed':False}
            try:
                decoder=generate_span if meta and meta['kind']=='s-learned-span-copy' else generate_copy
                raw=decoder(base,adapter,row['prompt'],plan['max_new_tokens']) if adapter else greedy_json(base,row['prompt'],plan['max_new_tokens'])
                item['raw']=raw;files=decode_proposal(raw,'python-cli');item.update(format_passed=True,files=files,source_sha256=digest(files))
            except (ValueError,TypeError) as error:item['diagnostic']=str(error)
            model['tasks'].append(item)
        output['models'].append(model)
        print(json.dumps({'label':label,'format_passed':sum(t['format_passed'] for t in model['tasks']),'tasks':8,'repairs':0}),flush=True)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if (a.output is None)==(a.verify is None):p.error('Choose exactly --output or --verify')
    if a.output and a.output.exists():p.error('Preserve existing generated outputs')
    result=generate(a.experiment)
    if a.verify:
        if json.loads(a.verify.read_text())!=result:raise RuntimeError('Frozen copy outputs differ from replay')
        print('PASS: all frozen byte-copy outputs replay exactly; no source execution or repairs')
    else:a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
