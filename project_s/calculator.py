import ast, operator
from decimal import Decimal, localcontext

def calculate(expression):
    if not isinstance(expression,str) or len(expression)>200: raise ValueError('Invalid expression')
    tree=ast.parse(expression,mode='eval')
    def run(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return Decimal(ast.get_source_segment(expression,n))
        if isinstance(n,ast.UnaryOp) and type(n.op) in (ast.UAdd,ast.USub): return run(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        if isinstance(n,ast.BinOp) and type(n.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div): return {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}[type(n.op)](run(n.left),run(n.right))
        raise ValueError('Unsupported expression')
    with localcontext() as ctx:
        ctx.prec=40
        return str(run(tree.body))

if __name__=='__main__':
    while True:
        try: print(calculate(input('Expression: ')))
        except (EOFError,KeyboardInterrupt): break
        except Exception as e: print('Error:',e)
