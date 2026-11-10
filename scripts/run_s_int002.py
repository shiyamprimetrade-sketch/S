"""Fixed-budget ASCII experiment. Evaluation never selects a model or training budget."""
import contextlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s_intelligence.train import CORPUS, run
from s_intelligence.evaluate import run as evaluate


def main():
    output = ROOT / 'artifacts/s-int-002'
    if output.exists():
        raise FileExistsError('Preserve existing experiments; use a fresh checkout')
    output.mkdir(parents=True)
    metrics, paths = [], []
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
        for steps, seed in ((80, 20261110), (600, 20261110), (600, 20261111), (600, 20261112)):
            checkpoint = output / f'steps-{steps}-seed-{seed}.pt'
            with contextlib.redirect_stdout(sys.stderr):
                metrics.append(run(steps, checkpoint, seed, 2, ROOT / 'data/training/s-int-002.json'))
            paths.append(checkpoint)
        result = evaluate(paths, output / 'evaluation.json')
    for metric in metrics:
        metric['checkpoint'] = Path(metric['checkpoint']).name
    for experiment in result['experiments']:
        experiment['checkpoint'] = Path(experiment['checkpoint']).name
        shared = [row for row in experiment['records'] if set(row['text']) <= set(CORPUS)]
        experiment['shared_7_record_cross_entropy'] = sum(
            row['trained_loss'] * row['target_characters'] for row in shared
        ) / sum(row['target_characters'] for row in shared)
    report = {'training': metrics, 'evaluation': result,
              'network': 'socket.connect denied throughout training and evaluation',
              'design': 'Fixed 80-step smoke and three 600-step seeds; no evaluation-driven tuning',
              'comparison_limit': 'Old model evaluated 7/12 records; compare shared subset, not unequal aggregate scores. Original benchmark was known during dataset design; not a blind test or causal ablation'}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
