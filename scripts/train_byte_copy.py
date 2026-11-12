"""Fixed-budget source-pointer/gate training; never executes generated source."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
from s_brain.checkpoint import load_weights,sha256_file
from s_brain.coding_curriculum import read_plan,records,batches,digest,OPERATIONS
from s_brain.byte_copy import ByteCopy,frozen_features,generate_copy
from s_brain.generation_validation import contract_matches,selection_rank
from s_developer.providers import decode_proposal


def supervision(rows,targets):
    # Training annotations only. Inference never calls this function.
    gate=torch.zeros_like(targets,dtype=torch.float32)
    pointers=torch.full_like(targets,-100)
    anchor=json.dumps('p["x"]')[1:-1].encode()
    for i,row in enumerate(rows):
        number=str(row['constant']).encode();prompt=row['prompt'].encode()
        if prompt.count(number)!=1:raise ValueError('Ambiguous training copy annotation')
        source=prompt.index(number)
        output=row['completion'].encode().index(anchor)+len(anchor)+len(OPERATIONS[row['operation']][1])
        start=len(prompt)-1+output
        for j in range(len(number)):gate[i,start+j]=1;pointers[i,start+j]=source+j
    return gate,pointers


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan=read_plan(args.plan);rows=records(plan);train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (args.output/'dataset.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed']);rng=random.Random(plan['seed'])
    base,manifest=load_weights(Path(plan['base_checkpoint']))
    if manifest.weights_sha256!=plan['base_weights_sha256']:raise ValueError('Base weights differ from plan')
    base.requires_grad_(False);base.eval()
    adapter=ByteCopy(**plan['adapter']);optimizer=torch.optim.AdamW(adapter.parameters(),lr=plan['learning_rate'])
    def forward(batch):
        x,y=batches(batch,base.cfg.context);logits,h=frozen_features(base,x)
        lengths=torch.tensor([len(r['prompt'].encode()) for r in batch]);source_length=int(lengths.max())
        source_mask=torch.arange(source_length)[None,:]<lengths[:,None]
        p,scores,g=adapter.distribution(h,logits,h[:,:source_length],x[:,:source_length],source_mask)
        return y,p,scores,g
    def nll(batch):
        with torch.no_grad():
            y,p,_,_=forward(batch);valid=y!=-100
            return float(-p[valid].gather(-1,y[valid,None]).clamp_min(1e-12).log().mean())
    report={'plan_sha256':digest(plan),'dataset_sha256':digest(rows),'base_weights_sha256':manifest.weights_sha256,
            'adapter_parameters':sum(p.numel() for p in adapter.parameters()),'base_parameters':sum(p.numel() for p in base.parameters()),
            'split_counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},'history':[],
            'test_generation_started':False,'source_executed':False,'repairs_allowed':0}
    def save(path,step):
        path=args.output/path;path.mkdir(parents=True,exist_ok=True)
        save_file(adapter.state_dict(),str(path/'adapter.safetensors'))
        meta={'schema':1,'independently_initialized':True,'kind':'s-learned-byte-copy','adapter_config':plan['adapter'],'step':step,
              'base_weights_sha256':manifest.weights_sha256,'weights_sha256':sha256_file(path/'adapter.safetensors'),
              'dataset_sha256':digest(rows),'plan_sha256':digest(plan),'seed':plan['seed'],
              'provenance':'self-authored synthetic; frozen independent SBrain plus independently initialized learned source attention/gate'}
        (path/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');return meta
    save(Path('random-control'),0);best=(-1,-1,float('-inf'));start=time.monotonic()
    for step in range(1,plan['steps']+1):
        adapter.train();selected=rng.choices(train,k=plan['batch_size']);y,p,scores,g=forward(selected)
        target_gate,pointers=supervision(selected,y);valid=y!=-100;copy=pointers!=-100
        loss_nll=-p[valid].gather(-1,y[valid,None]).clamp_min(1e-12).log().mean()
        loss_gate=torch.nn.functional.binary_cross_entropy(g.squeeze(-1)[valid],target_gate[valid])
        loss_pointer=torch.nn.functional.cross_entropy(scores[copy],pointers[copy])
        loss=loss_nll+loss_gate+loss_pointer
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if step%plan['evaluation_interval']==0:
            adapter.eval();val=nll(validation);details=[]
            for row in validation:
                raw=generate_copy(base,adapter,row['prompt'],plan['max_new_tokens']);item={'id':row['id'],'format_passed':False,'contract_matched':False}
                try:files=decode_proposal(raw,'python-cli');item.update(format_passed=True,contract_matched=contract_matches(files,row))
                except (ValueError,TypeError):pass
                details.append(item)
            metrics={'tasks':len(details),'format_passed':sum(r['format_passed'] for r in details),'contract_matched':sum(r['contract_matched'] for r in details),'details':details}
            rank=selection_rank(metrics,val)
            if rank>best:
                best=rank;meta=save(Path('selected'),step);report.update(selected_step=step,selected_weights_sha256=meta['weights_sha256'],selected_validation=metrics)
                save(Path('checkpoints')/f'step-{step:06d}',step)
            item={'step':step,'loss':float(loss.detach()),'completion_nll':float(loss_nll.detach()),'gate_bce':float(loss_gate.detach()),'pointer_ce':float(loss_pointer.detach()),'validation_nll':val,'format':metrics['format_passed'],'contracts':metrics['contract_matched'],'elapsed_seconds':round(time.monotonic()-start,3)}
            report['history'].append(item);(args.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
    assert sha256_file(Path(plan['base_checkpoint'])/'model.safetensors')==manifest.weights_sha256
    assert all(p.grad is None for p in base.parameters())
    print('Complete: frozen base unchanged; no test generation or source execution.',flush=True)


if __name__=='__main__':main()
