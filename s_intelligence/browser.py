"""Bounded local browser adapter for the S-INT-002 experimental checkpoint."""
from importlib.util import find_spec
from pathlib import Path
import threading

MODEL_ID = 's-native:local'
CHECKPOINT = Path(__file__).resolve().parents[1] / 'native-research/S-INT-002/artifacts/s-int-002/steps-80-seed-20261110.pt'
NOTICE = ('Experimental native model: generated text is unreliable; reasoning is not established. '
          'ASCII only; 32-character context; no conversation memory.\n\n')
_LOCK = threading.Lock()


def available():
    return CHECKPOINT.is_file() and find_spec('torch') is not None


def generate_answer(text, checkpoint=CHECKPOINT):
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ValueError('Native Lab needs a nonempty prompt of at most 4096 characters')
    if any(c not in '\n\t' and not ' ' <= c <= '~' for c in text):
        raise ValueError('Native Lab supports ASCII text only; Unicode is not supported')
    if not Path(checkpoint).is_file():
        raise ValueError('Native Lab checkpoint is missing')
    import torch
    from .model import SModel
    # Serialize requests because sampling uses the process-wide PyTorch RNG.
    with _LOCK, torch.random.fork_rng(devices=[]):
        torch.set_num_threads(2)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        chars = state['vocabulary']
        indices = {char: i for i, char in enumerate(chars)}
        prompt = text + '\nAnswer: '
        if any(c not in indices for c in prompt):
            raise ValueError('Native Lab checkpoint does not support the prompt vocabulary')
        model = SModel(len(chars), state['width'], state['context'])
        model.load_state_dict(state['weights'])
        torch.manual_seed(20261110)
        output = model.generate(torch.tensor([[indices[c] for c in prompt]]), count=80)
        raw = ''.join(chars[i] for i in output[0, len(prompt):].tolist())
    return ''.join(c for c in raw if c.isprintable() or c in '\n\t').strip()


def complete(text):
    return NOTICE + (generate_answer(text) or '[No readable text generated]')
