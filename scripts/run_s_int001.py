"""Execute bounded S-INT-001 experiments with all socket connections denied."""
import contextlib
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from s_intelligence.train import run
from s_intelligence.evaluate import run as evaluate


def main():
    output = ROOT / 'artifacts/s-int-001'
    output.mkdir(parents=True, exist_ok=True)
    configurations = [('smoke', 80, 20261110), ('controlled', 240, 20261110),
                      ('controlled', 240, 20261111), ('controlled', 240, 20261112),
                      ('reproducibility', 80, 20261110)]
    metrics = []
    paths = []
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden during native experiment')):
        for label, steps, seed in configurations:
            checkpoint = output / f'{label}-seed-{seed}.pt'
            with contextlib.redirect_stdout(sys.stderr):
                result = run(steps=steps, output=checkpoint, seed=seed, threads=2)
            metrics.append(dict(result, experiment=label))
            paths.append(checkpoint)
            print(f'Completed {label}, seed={seed}, steps={steps}', flush=True)
            gc.collect()
        original = torch.load(paths[0], map_location='cpu', weights_only=True)
        repeated = torch.load(paths[-1], map_location='cpu', weights_only=True)
        exact = all(torch.equal(original['weights'][key], repeated['weights'][key]) for key in original['weights'])
        if not exact:
            raise AssertionError('Repeated seed did not reproduce exact model weights')
        evaluation = evaluate(paths[:-1], output / 'evaluation.json')
    result = {'runs': metrics, 'reproducibility_exact_weights': exact,
              'network': 'All socket.connect calls denied for training, checkpoint reload, evaluation, and generation',
              'evaluated_experiments': len(evaluation['experiments'])}
    (output / 'experiments.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print('S-INT-001 EXPERIMENTS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
