import ast

from decimal import (
    Decimal,
    localcontext,
    ROUND_HALF_UP
)


class ArithmeticVerifier:

    def verify(self, report):

        expression = (
            report["expression"]
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
        )

        tree = ast.parse(
            expression, mode="eval"
        )

        precision = report["precision"]

        with localcontext() as ctx:

            ctx.prec = 150
            ctx.rounding = ROUND_HALF_UP

            def solve(node):

                if isinstance(node, ast.Constant):

                    return Decimal(
                        ast.get_source_segment(
                            expression, node
                        )
                    )

                if isinstance(node, ast.UnaryOp):

                    value = solve(node.operand)

                    if isinstance(node.op, ast.USub):
                        return -value

                    if isinstance(node.op, ast.UAdd):
                        return value

                    raise ValueError("Unsupported unary")

                if not isinstance(node, ast.BinOp):
                    raise ValueError("Unsupported node")

                a = solve(node.left)
                b = solve(node.right)

                if isinstance(node.op, ast.Add):
                    return a + b

                if isinstance(node.op, ast.Sub):
                    return a - b

                if isinstance(node.op, ast.Mult):
                    return a * b

                if isinstance(node.op, ast.Div):

                    return (a / b).quantize(
                        Decimal(1).scaleb(-precision)
                    )

                if isinstance(node.op, ast.FloorDiv):
                    return Decimal(int(a) // int(b))

                if isinstance(node.op, ast.Mod):
                    return Decimal(int(a) % int(b))

                raise ValueError("Unsupported operation")

            expected = solve(tree.body)

            actual = Decimal(report["result"])

            return {
                "verified": actual == expected,
                "method": "independent_decimal_ast"
            }
