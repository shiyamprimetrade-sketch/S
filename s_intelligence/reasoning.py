"""Frozen short-answer diagnostic using the exact browser generation adapter."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from .browser import CHECKPOINT, generate_answer

DATA = Path(__file__).resolve().parents[1] / 'data/evaluation/native-reasoning.json'


def matches(output, expected):
    return ' '.join(output.strip().rstrip('.').casefold().split()) == expected.casefold()


def run(checkpoint, output, data=DATA):
    path = Path(output)
    if path.exists():
        raise FileExistsError('Preserve existing evaluation evidence')
    document = json.loads(Path(data).read_text(encoding='utf8'))
    rows = []
    categories = defaultdict(lambda: {'correct': 0, 'total': 0})
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
        for case in document['cases']:
            response = generate_answer(case['prompt'], checkpoint)
            passed = matches(response, case['expected'])
            rows.append(dict(case, output=response, correct=passed))
            categories[case['category']]['total'] += 1
            categories[case['category']]['correct'] += int(passed)
    result = {'checkpoint': Path(checkpoint).name,
              'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
              'provenance': document['provenance'], 'cases': rows, 'categories': dict(categories),
              'correct': sum(r['correct'] for r in rows), 'total': len(rows),
              'scoring': 'Whole answer exact match after case, whitespace and final-period normalization; no substring credit',
              'network': 'socket.connect denied', 'generation_seed': 20261110,
              'limitation': 'Small synthetic diagnostic. End-to-end answer failure does not isolate reasoning from language, instruction following or context limits. No training or tuning used these cases.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'correct': result['correct'], 'total': result['total'], 'categories': result['categories']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(CHECKPOINT))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
