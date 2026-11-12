"""One permissioned arithmetic-literal repair; never executes candidate source."""
import ast
import re
from .bundles import validate_files, fingerprint


SYMBOLS={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.FloorDiv:'//'}


def repair_literal(files, approved_constant, *, consent=False):
    if consent is not True:raise PermissionError('Explicit literal-repair consent required')
    if type(approved_constant) is not int or not 1<=approved_constant<=999:
        raise ValueError('Approved arithmetic constant must be an integer between 1 and 999')
    files=validate_files(files,'python-cli')
    tree=ast.parse(files['app.py'])
    try:
        expr=tree.body[0].body[0].value.values[0]
        old=expr.right.value;symbol=SYMBOLS[type(expr.op)]
        if type(old) is not int or not 1<=old<=999:raise ValueError('Unsupported generated literal')
        skeleton=ast.parse(f'def solve(p):\n    return {{"value":p["x"]{symbol}{old}}}\n')
        if ast.dump(tree,include_attributes=False)!=ast.dump(skeleton,include_attributes=False):
            raise ValueError('Candidate is outside the single-expression repair grammar')
    except (AttributeError,IndexError,KeyError,TypeError) as error:
        raise ValueError('Candidate is outside the single-expression repair grammar') from error
    if old==approved_constant:raise ValueError('Literal already matches; no speculative repair')
    node=expr.right
    lines=files['app.py'].encode().splitlines(keepends=True)
    if node.lineno!=node.end_lineno:raise ValueError('Unsupported literal span')
    start=sum(map(len,lines[:node.lineno-1]))+node.col_offset
    end=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    raw=files['app.py'].encode()
    patched=dict(files);patched['app.py']=(raw[:start]+str(approved_constant).encode()+raw[end:]).decode()
    patched=validate_files(patched,'python-cli')
    return {'files':patched,'provenance':{'provider':'requirements-literal',
        'method':'deterministic approved-requirement literal repair; operation unchanged',
        'independently_generated_neural_code':False,'original_source_sha256':fingerprint(files),
        'source_sha256':fingerprint(patched),'changed_path':'app.py','old_constant':old,
        'approved_constant':approved_constant,'repairs_used':1}}


def requirement_constant(text):
    """Recognize one explicit arithmetic constant; ambiguous prose is refused."""
    if not isinstance(text,str) or len(text)>8000:raise ValueError('Unsupported arithmetic requirement')
    numbers=re.findall(r'(?<!\w)\d+(?!\w)',text)
    if len(numbers)!=1:raise ValueError('Exactly one explicit arithmetic constant is required')
    phrase=r'(?:add\s+\d+\s+to\s+p\["x"\]|subtract\s+\d+\s+from\s+p\["x"\]|multiply\s+p\["x"\]\s+by\s+\d+|floor-divide\s+p\["x"\]\s+by\s+\d+)'
    if not re.search(phrase,text,re.IGNORECASE):raise ValueError('No supported explicit arithmetic requirement')
    value=int(numbers[0])
    if not 1<=value<=999:raise ValueError('Arithmetic requirement constant is outside the repair budget')
    return value


class LiteralRepairProvider:
    id='requirements-literal'
    def __init__(self):self.last=None
    def provenance(self):
        return self.last or {'provider':self.id,'independently_generated_neural_code':False,
                             'method':'repair only; copies one approved arithmetic literal, never chooses the operation'}
    def generate(self,spec,recipe):raise ValueError('Requirements-literal is repair-only; select a source generator')
    def repair(self,spec,recipe,files,diagnostic):
        if recipe!='python-cli':raise ValueError('Literal repair supports Python arithmetic only')
        constant=requirement_constant(spec['answers']['purpose'])
        result=repair_literal(files,constant,consent=True)
        self.last=result['provenance']
        return result['files']
