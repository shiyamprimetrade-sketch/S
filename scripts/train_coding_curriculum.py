"""Reproduce a fixed-budget, independently initialized coding experiment on CPU."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from s_brain.model import Config, SBrain
from s_brain.checkpoint import save_checkpoint
from s_brain.supervised_data import masked_cross_entropy
from s_brain.coding_curriculum import batches, digest, read_plan, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Prevent accidental destruction of earlier experiments.
    args.output.mkdir(parents=True, exist_ok=False)
    plan = read_plan(args.plan)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    rows = records(plan)
    (args.output / 'dataset.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed'])
    rng = random.Random(plan['seed'])
    cfg = Config(**plan['model'])
    model = SBrain(cfg)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'])
    vx, vy = batches(validation, cfg.context)
    def evaluate():
        model.eval()
        with torch.no_grad():
            logits, _ = model(vx)
            return float(masked_cross_entropy(logits, vy))
    fingerprint = digest(rows)
    extras = {'independently_initialized': True, 'source': 'self-authored-synthetic',
              'plan_sha256': digest(plan), 'no_pretrained_weights': True,
              'selection': plan['selection']}
    def save(name, step, tokens, loss, validation_loss):
        return save_checkpoint(args.output / name, model, cfg, step=step,
            train_tokens=tokens, train_loss=loss, validation_loss=validation_loss,
            seed=plan['seed'], dataset_fingerprint=fingerprint, extras=extras)
    best = evaluate()
    save('random-control', 0, 0, best, best)
    report = {'plan_sha256': digest(plan), 'dataset_sha256': fingerprint,
              'parameters': sum(p.numel() for p in model.parameters()),
              'split_counts': {s: sum(r['split'] == s for r in rows) for s in ('train','validation','test')},
              'initial_validation_nll': best, 'history': [], 'selected_step': 0,
              'weight_update_l2_definition': 'Latest update state_dict L2, including tied aliases; not selected-checkpoint distance',
              'selection_frozen_before_test_generation': True}
    tokens = 0
    start = time.monotonic()
    for step in range(1, plan['steps'] + 1):
        model.train()
        selected = rng.choices(train, k=plan['batch_size'])
        x, y = batches(selected, cfg.context)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = masked_cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; training stopped')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        warm = min(1, step / plan['warmup_steps'])
        progress = max(0, step-plan['warmup_steps']) / (plan['steps']-plan['warmup_steps'])
        lr = warm * (plan['minimum_learning_rate'] + (plan['learning_rate']-plan['minimum_learning_rate'])*0.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.step()
        tokens += int((y != -100).sum())
        if step % plan['evaluation_interval'] == 0 or step == plan['steps']:
            val = evaluate()
            row = {'step':step, 'train_nll':float(loss.detach()), 'validation_nll':val,
                   'gradient_norm':float(norm), 'learning_rate':lr,
                   'elapsed_seconds':round(time.monotonic()-start, 3)}
            report['history'].append(row)
            if val < best:
                best = val
                report['selected_step'] = step
                manifest = save('selected', step, tokens, row['train_nll'], val)
                report['selected_weights_sha256'] = manifest.weights_sha256
            report['weight_update_l2'] = math.sqrt(sum(float(((v.detach()-initial[k])**2).sum()) for k,v in model.state_dict().items()))
            (args.output/'training-report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)
    print('Training complete; test generation has not run.', flush=True)


if __name__ == '__main__': main()
