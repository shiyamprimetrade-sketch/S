"""Free-running validation diagnostics; parses source but never executes it."""
import ast
import json
from .coding_decode import greedy_json
from .coding_curriculum import OPERATIONS
from s_developer.providers import decode_proposal


def contract_matches(files, row):
    """Restricted arithmetic contract AST equality, not a runtime success claim.

    Expected validation source is authored independently of model generation.
    This metric is selection-only; held-out acceptance uses isolated execution.
    """
    expected = json.loads(row['completion'])['files']['app.py']
    try:
        actual = ast.dump(ast.parse(files['app.py']), include_attributes=False)
        wanted = ast.dump(ast.parse(expected), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return actual == wanted


def generation_metrics(model, rows, budget):
    outputs = []
    for row in rows:
        raw = greedy_json(model, row['prompt'], budget)
        result = {'id': row['id'], 'format_passed': False, 'contract_matched': False}
        try:
            files = decode_proposal(raw, 'python-cli')
            result['format_passed'] = True
            result['contract_matched'] = contract_matches(files, row)
        except (ValueError, TypeError):
            pass
        outputs.append(result)
    return {'tasks': len(rows), 'format_passed': sum(r['format_passed'] for r in outputs),
            'contract_matched': sum(r['contract_matched'] for r in outputs),
            'source_executed': False, 'tasks_detail': outputs}


def selection_rank(metrics, nll):
    """Generated contract matches outrank formatting; loss is only a tie breaker."""
    return metrics['contract_matched'], metrics['format_passed'], -float(nll)


def semantic_token_weights(rows, targets, weight):
    """Give operation/constant bytes explicit gradient weight, retaining all code.

    No outputs/answers are inserted by this function. It affects training loss
    only; generation stays unconstrained and validation NLL stays unweighted.
    """
    import torch
    if not isinstance(weight, (int, float)) or not 1 <= weight <= 32:
        raise ValueError('Semantic weight must be between 1 and 32')
    weights = (targets != -100).to(torch.float32)
    anchor = json.dumps('p["x"]')[1:-1].encode()
    for i, row in enumerate(rows):
        completion = row['completion'].encode()
        offset = completion.index(anchor) + len(anchor)
        length = len(OPERATIONS[row['operation']][1]) + len(str(row['constant']))
        start = len(row['prompt'].encode()) - 1 + offset
        weights[i, start:start+length] = weight
    return weights
