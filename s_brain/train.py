"""Train independent byte-level next-token model; no external teacher needed.

Example: python -m s_brain.train --corpus my_authorized_text.txt --steps 100
"""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import torch
from .model import Config, SBrain

def run():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("s_brain/checkpoints"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.steps < 1 or a.batch < 1 or a.sequence < 8:
        p.error("steps/batch must be positive; sequence must be >= 8")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    data = a.corpus.read_bytes()
    if len(data) < 20*(a.sequence+1):
        p.error("Corpus too small: supply at least 20 training windows")
    cut = int(len(data)*0.9)
    # 90% train, 10% holdout; byte vocabulary avoids tokenizer mismatch.
    cfg = Config(dim=a.dim, layers=a.layers, context=a.sequence)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBrain(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    train, valid = torch.tensor(list(data[:cut]), dtype=torch.long), torch.tensor(list(data[cut:]), dtype=torch.long)
    if len(valid) < a.sequence+2:
        p.error("Validation split too small")
    def batch(source):
        start = torch.randint(0, len(source)-a.sequence, (a.batch,))
        delta = torch.arange(a.sequence)
        x = source[start[:,None]+delta].to(device)
        y = source[start[:,None]+delta+1].to(device)
        return x,y
    a.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for step in range(1,a.steps+1):
        model.train()
        x,y = batch(train)
        opt.zero_grad(set_to_none=True)
        _,loss = model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == a.steps:
            model.eval()
            with torch.inference_mode():
                x,y = batch(valid)
                _,val = model(x,y)
            score = float(val)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={score:.4f}", flush=True)
            if score < best:
                best = score
                # Safe to load only from a trusted local training run.
                torch.save(model.state_dict(), a.output/"weights.pt")
                (a.output/"config.json").write_text(json.dumps(asdict(cfg), indent=2))
                (a.output/"metrics.json").write_text(json.dumps({"step":step,"validation_loss":score,"seed":a.seed}, indent=2))
    print("Saved best validation checkpoint to", a.output)

if __name__ == "__main__":
    run()
