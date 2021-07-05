
from gpflow.kernels import RBF
from gpflow.kernels.linears import Linear
from gpflow.kernels.statics import White
from kernel_discovery.description.simplify import simplify
from kernel_discovery.description.transform import ast_to_kernel, ast_to_text, kernel_to_ast
from typing import Any, Dict, List, Optional
from anytree import Node
from kernel_discovery.expansion.grammar import expand_kernel


def expand_asts(asts: List[Node], grammar_kwargs: Optional[Dict[str, Any]] = None):

    expanded_kernels = {}
    for ast in asts:
        kernel = ast_to_kernel(ast)
        for candidate in expand_kernel(kernel, grammar_kwargs):
            ast = kernel_to_ast(candidate)
            ast = simplify(ast)
            expanded_kernels[ast_to_text(ast)] = ast

    return list(expanded_kernels.values())


if __name__ == '__main__':

    # unit test
    def test_expand_asts():
        k_linear = Linear()
        k_white = White()
        k_rbf = RBF()

        ast_linear = Node(type(k_linear))
        ast_white = Node(type(k_white))
        ast_rbf = Node(type(k_rbf))

        res_should_be = expand_kernel(
            k_linear) + expand_kernel(k_white) + expand_kernel(k_rbf)

        expanded_kernels = [ast_to_text(ast) for ast in expand_asts(
            [ast_linear, ast_white, ast_rbf])]

        assert set(expanded_kernels) == {ast_to_text(
            simplify(kernel_to_ast(k))) for k in res_should_be}

    test_expand_asts()
