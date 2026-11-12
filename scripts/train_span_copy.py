"""Fixed-budget frozen-SBrain learned span/gate training; no source execution."""
import argparse,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
from s_brain.checkpoint import load_weights,sha256_file
from s_brain.coding_curriculum import read_plan,records,batches,digest
from s_brain.byte_copy import frozen_features
from s_brain.span_copy import SpanCopy,generate_span
from s_brain.generation_validation import contract_matches,selection_rank
from s_developer.providers import decode_proposal
from scripts.train_byte_copy import supervision


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=read_plan(a.plan);rows=records(plan);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(a.output/'dataset.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed']);rng=random.Random(plan['seed'])
    base,manifest=load_weights(Path(plan['base_checkpoint']));assert manifest.weights_sha256==plan['base_weights_sha256'];base.requires_grad_(False)
    adapter=SpanCopy(**plan['adapter']);opt=torch.optim.AdamW(adapter.parameters(),lr=plan['learning_rate'])
    def forward(batch):
        x,y=batches(batch,base.cfg.context);_,h=frozen_features(base,x)
        lengths=torch.tensor([len(r['prompt'].encode()) for r in batch]);n=int(lengths.max());mask=torch.arange(n)[None,:]<lengths[:,None]
        starts,ends=adapter.positions(h[:,:n],mask);gate,pointers=supervision(batch,y)
        source_start=[];source_end=[];gate_mask=y!=-100;gate_target=torch.zeros_like(gate)
        for i in range(len(batch)):
            indices=(pointers[i]!=-100).nonzero().flatten();source_start.append(int(pointers[i,indices[0]]));source_end.append(int(pointers[i,indices[-1]]))
            gate_target[i,indices[0]]=1;gate_mask[i,indices[1:]]=False
        gate_logits=adapter.gate(h).squeeze(-1)
        loss=(torch.nn.functional.cross_entropy(starts,torch.tensor(source_start))+torch.nn.functional.cross_entropy(ends,torch.tensor(source_end))+
              torch.nn.functional.binary_cross_entropy_with_logits(gate_logits[gate_mask],gate_target[gate_mask],pos_weight=torch.tensor(16.0)))
        return loss
    report={'plan_sha256':digest(plan),'dataset_sha256':digest(rows),'base_weights_sha256':manifest.weights_sha256,'base_parameters':sum(v.numel() for v in base.parameters()),
            'adapter_parameters':sum(v.numel() for v in adapter.parameters()),'split_counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},'history':[],'test_generation_started':False,'repairs_allowed':0,'source_executed':False}
    def save(name,step):
        root=a.output/name;root.mkdir(parents=True,exist_ok=True);save_file(adapter.state_dict(),str(root/'adapter.safetensors'))
        value={'schema':1,'independently_initialized':True,'kind':'s-learned-span-copy','adapter_config':plan['adapter'],'step':step,'seed':plan['seed'],
               'base_weights_sha256':manifest.weights_sha256,'weights_sha256':sha256_file(root/'adapter.safetensors'),'plan_sha256':digest(plan),'dataset_sha256':digest(rows),
               'provenance':'self-authored synthetic; frozen independently trained SBrain + learned span endpoints/gate; one bounded copy action, no repair'}
        (root/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');return value
    save(Path('random-control'),0);best=(-1,-1,float('-inf'));start=time.monotonic()
    for step in range(1,plan['steps']+1):
        adapter.train();loss=forward(rng.choices(train,k=plan['batch_size']))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1,error_if_nonfinite=True);opt.step()
        if step%plan['evaluation_interval']==0:
            adapter.eval()
            with torch.no_grad():validation_loss=float(forward(val))
            details=[]
            for row in val:
                item={'id':row['id'],'format_passed':False,'contract_matched':False}
                try:
                    raw=generate_span(base,adapter,row['prompt'],plan['max_new_tokens']);files=decode_proposal(raw,'python-cli');item.update(format_passed=True,contract_matched=contract_matches(files,row))
                except (ValueError,TypeError):pass
                details.append(item)
            metrics={'tasks':len(details),'format_passed':sum(r['format_passed'] for r in details),'contract_matched':sum(r['contract_matched'] for r in details),'details':details}
            rank=selection_rank(metrics,validation_loss)
            if rank>best:
                best=rank;meta=save(Path('selected'),step);save(Path('checkpoints')/f'step-{step:06d}',step)
                report.update(selected_step=step,selected_weights_sha256=meta['weights_sha256'],selected_validation=metrics)
            item={'step':step,'loss':float(loss.detach()),'validation_objective':validation_loss,'format':metrics['format_passed'],'contracts':metrics['contract_matched'],'elapsed_seconds':round(time.monotonic()-start,3)}
            report['history'].append(item);(a.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
    assert sha256_file(Path(plan['base_checkpoint'])/'model.safetensors')==manifest.weights_sha256
    assert all(p.grad is None for p in base.parameters())
    print('Complete: base unchanged; no test generation or repair.',flush=True)


if __name__=='__main__':main()
