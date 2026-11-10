"""Independent character-level evaluation; never used by the training pipeline."""
import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch
from .infer import run as infer
from .model import SModel
from .train import CORPUS

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data/evaluation/s-int-001.json'


def evaluate(checkpoint, data=DEFAULT_DATA):
    document = json.loads(Path(data).read_text(encoding='utf-8-sig'))
    texts = document['texts']
    if not texts or any(not isinstance(t, str) or len(t) < 2 or t in CORPUS for t in texts):
        raise ValueError('Evaluation needs independent, nonempty texts absent from the corpus')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    vocabulary = state['vocabulary']
    indices = {char: i for i, char in enumerate(vocabulary)}
    torch.manual_seed(state['seed'])
    untrained = SModel(len(vocabulary), state['width'], state['context']).eval()
    trained = SModel(len(vocabulary), state['width'], state['context']).eval()
    trained.load_state_dict(state['weights'])
    counts = Counter(CORPUS)
    rows = []
    token_count = 0
    initial_sum = final_sum = unigram_sum = 0.0
    for text in texts:
        missing = sorted(set(text) - set(vocabulary))
        if missing:
            rows.append({'text': text, 'status': 'unsupported_vocabulary', 'missing_characters': missing})
            continue
        ids = torch.tensor([indices[char] for char in text], dtype=torch.long)
        before = after = 0.0
        with torch.no_grad():
            for offset in range(0, len(ids) - 1, state['context']):
                x = ids[offset:offset + state['context']].unsqueeze(0)
                y = ids[offset + 1:offset + 1 + x.shape[1]]
                x = x[:, :len(y)]
                for model, label in ((untrained, 'initial'), (trained, 'final')):
                    logits, _ = model(x)
                    loss = float(torch.nn.functional.cross_entropy(
                        logits[0], y, reduction='sum'))
                    if label == 'initial':
                        before += loss
                    else:
                        after += loss
        size = len(ids) - 1
        unigram = -sum(math.log(counts[c] / len(CORPUS)) for c in text[1:])
        token_count += size
        initial_sum += before
        final_sum += after
        unigram_sum += unigram
        rows.append({'text': text, 'status': 'evaluated', 'target_characters': size,
                     'untrained_loss': before / size, 'trained_loss': after / size,
                     'training_unigram_loss': unigram / size})
    if not token_count:
        raise ValueError('No evaluation text fits this checkpoint vocabulary')
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        generated = infer(checkpoint, prompt='S ', count=80, seed=20261110)
    return {
        'checkpoint': str(checkpoint), 'seed': state['seed'], 'steps': state.get('steps'),
        'evaluation_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'provenance': document['provenance'], 'records': rows,
        'evaluated_texts': sum(r['status'] == 'evaluated' for r in rows),
        'unsupported_texts': sum(r['status'] != 'evaluated' for r in rows),
        'target_characters': token_count,
        'untrained_cross_entropy': initial_sum / token_count,
        'trained_cross_entropy': final_sum / token_count,
        'training_unigram_cross_entropy': unigram_sum / token_count,
        'uniform_cross_entropy': math.log(len(vocabulary)),
        'generated_output': generated,
        'inference_seconds_including_checkpoint_load': time.perf_counter() - start,
        'inference_prompt': 'S ', 'generation_seed': 20261110, 'generated_characters': 80,
        'network': 'socket.connect denied throughout evaluation and inference',
        'limitation': 'Character likelihood on a small synthetic set; not evidence of reasoning or conversational competence',
    }


def run(checkpoints, output, data=DEFAULT_DATA, threads=2):
    torch.set_num_threads(threads)
    with patch('socket.socket.connect', side_effect=AssertionError('Network access forbidden')):
        results = [evaluate(path, data) for path in checkpoints]
    result = {'experiments': results, 'metric': 'next-character natural-log cross-entropy',
              'scope': 'New synthetic text, matched random-initialization and unigram baselines; unsupported records disclosed'}
    path = Path(output)
    if path.exists():
        raise FileExistsError('Choose a new evaluation report path')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', default=str(DEFAULT_DATA))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.data, args.threads)
