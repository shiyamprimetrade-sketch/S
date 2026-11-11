"""Bounded, unconstrained native byte-model JSON decoding."""
import json


def greedy_json(model, prompt, max_new_tokens=200):
    """Unconstrained greedy generation; stop only on a complete JSON object.

    No wrapper, template, filtering, repair, sampled retries, or source execution.
    Invalid prefixes remain invalid and exhaust the explicit generation budget.
    """
    import torch
    from .kv_cache import prefill, append
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
        raise ValueError('Generation budget must be 1–1024 tokens')
    encoded = prompt.encode()
    if not encoded or len(encoded) > model.cfg.context:
        raise ValueError('Prompt exceeds context; truncation forbidden')
    output = bytearray()
    logits, state = prefill(model, torch.tensor([list(encoded)], dtype=torch.long))
    for _ in range(max_new_tokens):
        token = int(logits[0, -1].argmax())
        output.append(token)
        try:
            value = json.loads(output.decode('utf-8'))
            if isinstance(value, dict):
                return output.decode('utf-8')
        except (UnicodeDecodeError, ValueError):
            pass
        logits, state = append(model, torch.tensor([[token]], dtype=torch.long), state)
    return output.decode('utf-8', errors='replace')
