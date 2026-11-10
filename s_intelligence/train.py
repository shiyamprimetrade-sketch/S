"""Reproducible from-scratch CPU training on explicitly synthetic text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from .model import CharTokenizer, SModel

CORPUS = ('S observes a pattern. S tests the pattern. S records the evidence.\n' * 120)


def peak_memory_bytes():
    """Return the process-lifetime peak resident memory, including PyTorch import."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'peak_working_set', 'working_set', 'peak_paged', 'paged',
                    'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.peak_working_set
        return None
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == 'darwin' else peak * 1024
    except ImportError:
        return None


def run(steps=80, output='artifacts/s-native.pt', seed=20261110, threads=2):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Training steps must be a positive integer')
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer from 0 through 2**32 - 1')
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('CPU threads must be a positive integer')
    path = Path(output)
    metrics_path = path.with_suffix('.metrics.json')
    if path.exists() or metrics_path.exists():
        raise FileExistsError('Choose a new checkpoint path; existing experiments are preserved')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    tok = CharTokenizer(CORPUS)
    ids = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    cut = int(len(ids) * .9)
    train, validation = ids[:cut], ids[cut:]
    model = SModel(len(tok.chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)

    def sample(data, generator=None, batch=16):
        starts = torch.randint(0, len(data) - model.context, (batch,), generator=generator)
        return (torch.stack([data[i:i + model.context] for i in starts]),
                torch.stack([data[i + 1:i + model.context + 1] for i in starts]))

    evaluation_rng = torch.Generator().manual_seed(seed + 1)
    train_batch = sample(train, evaluation_rng)
    validation_batch = sample(validation, evaluation_rng)

    def losses():
        model.eval()
        with torch.no_grad():
            return [float(model(x, y)[1]) for x, y in (train_batch, validation_batch)]

    initial_train, initial_validation = losses()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    model.train()
    for _ in range(steps):
        x, y = sample(train)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is not finite')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    duration = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    final_train, final_validation = losses()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'weights': model.state_dict(), 'vocabulary': tok.chars,
                'context': model.context, 'width': 64, 'seed': seed,
                'steps': steps, 'initialization': 'random_from_scratch',
                'corpus_sha256': hashlib.sha256(CORPUS.encode()).hexdigest()}, path)
    result = {
        'steps': steps, 'seed': seed, 'threads': threads,
        'initial_training_loss': initial_train, 'final_training_loss': final_train,
        'initial_validation_loss': initial_validation, 'final_validation_loss': final_validation,
        'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_seconds': duration, 'training_cpu_seconds': cpu_seconds,
        'cpu_core_equivalents': cpu_seconds / duration,
        'cpu_percent_of_machine': 100 * cpu_seconds / duration / (os.cpu_count() or 1),
        'peak_process_resident_bytes': peak_memory_bytes(),
        'checkpoint': str(path), 'checkpoint_bytes': path.stat().st_size,
        'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'corpus_sha256': hashlib.sha256(CORPUS.encode()).hexdigest(),
        'vocabulary': tok.chars, 'torch': torch.__version__,
        'device': 'cpu', 'initialization': 'random_from_scratch',
        'evaluation': 'Same frozen train and validation batches before and after training',
        'limitation': 'Repeated synthetic text; validation shares training patterns, not independent generalization',
    }
    metrics_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--output', default='artifacts/s-native.pt')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.steps, args.output, args.seed, args.threads)
