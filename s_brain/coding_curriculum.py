"""Public synthetic arithmetic curriculum, distinct from S's original coding suite.

Training completions are authored fixtures, never model-generated claims. Test
answers are computed by a separate host oracle and are never fed to a model.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROTOCOL = 'Return JSON {"files":{"app.py":"...","test_app.py":"..."}}. app.py must define solve(p). Use stdlib only.\n'
OPERATIONS = {'add': ('Add', '+'), 'subtract': ('Subtract', '-'),
              'multiply': ('Multiply', '*'), 'floor_divide': ('Floor-divide', '//')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def contract(operation, constant, style):
    if operation not in OPERATIONS or not 1 <= constant <= 999 or style not in range(5):
        raise ValueError('Unsupported curriculum coordinate')
    verb, symbol = OPERATIONS[operation]
    descriptions = {
        'add': f'add {constant} to p["x"]',
        'subtract': f'subtract {constant} from p["x"]',
        'multiply': f'multiply p["x"] by {constant}',
        'floor_divide': f'floor-divide p["x"] by {constant}',
    }
    action = descriptions[operation]
    return [f'For integer input, {action}. Return a JSON object with key "value".',
            f'Implement this operation: {action}; output {{"value": result}}.',
            f'Given integer p["x"], compute p["x"]{symbol}{constant}. Return {{"value": result}}.',
            f'Take integer p["x"] and {action}. Put the result in key "value".',
            f'Please {action} for integer input p. Respond with the result under "value".'][style]


def records(plan):
    rows = []
    for constant in range(plan['constant_range'][0], plan['constant_range'][1] + 1):
        split = ('validation' if constant in plan['validation_constants'] else
                 'test' if constant in plan['test_constants'] else 'train')
        for operation in plan['operations']:
            symbol = OPERATIONS[operation][1]
            completion = json.dumps({'files': {
                'app.py': f'def solve(p):\n    return {{"value":p["x"]{symbol}{constant}}}\n',
                'test_app.py': plan.get('test_file_comment', '# Evaluated by independent host cases.\n')}}, separators=(',', ':'))
            defaults = {'train': [0, 1, 2], 'validation': [3], 'test': [4]}
            for style in plan.get('prompt_styles', defaults)[split]:
                prompt = PROTOCOL + contract(operation, constant, style) + plan.get('prompt_suffix', '')
                rows.append({'id': f'{operation}-{constant}-{style}', 'split': split,
                             'operation': operation, 'constant': constant, 'style': style,
                             'prompt': prompt, 'completion': completion,
                             'provenance': 'self-authored-synthetic', 'license': 'CC0-1.0'})
    validate_splits(rows)
    return rows


def validate_splits(rows):
    """Both exact prompts and whole completions must be split-disjoint."""
    for field in ('prompt', 'completion'):
        owners = {}
        for row in rows:
            key = hashlib.sha256(row[field].encode()).hexdigest()
            previous = owners.setdefault(key, row['split'])
            if previous != row['split']:
                raise ValueError(f'{field} leakage between {previous} and {row["split"]}')


def oracle(operation, constant, value):
    if operation == 'add': result = value + constant
    elif operation == 'subtract': result = value - constant
    elif operation == 'multiply': result = value * constant
    elif operation == 'floor_divide': result = value // constant
    else: raise ValueError('Unknown operation')
    return {'value': result}


def case_inputs(plan, task):
    if plan.get('case_strategy') == 'arithmetic-boundaries':
        k = task['constant']
        return [-21, 0, 19, k-1, k, -k-1, 2*k]
    return [-21, 0, 19]


def batches(rows, context):
    """Full records, completion-only targets. No silent context truncation."""
    import torch
    sequences = [(r['prompt'].encode(), r['completion'].encode()) for r in rows]
    length = max(len(p) + len(c) - 1 for p, c in sequences)
    if length > context:
        raise ValueError('Full curriculum record exceeds model context')
    inputs = torch.zeros((len(rows), length), dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for i, (prompt, completion) in enumerate(sequences):
        sequence = torch.tensor(list(prompt + completion), dtype=torch.long)
        inputs[i, :len(sequence)-1] = sequence[:-1]
        targets[i, len(prompt)-1:len(sequence)-1] = sequence[len(prompt):]
    return inputs, targets


def read_plan(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
