
from gpflow.kernels import RBF
from gpflow.kernels.linears import Linear
from gpflow.kernels.statics import Constant, White
from kernel_discovery.description.simplify import simplify, distribution
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
            simplified = simplify(ast)
            simplified_text = ast_to_text(simplified)
            expanded_kernels[simplified_text] = simplified
            
    print(len(expanded_kernels))

    return list(expanded_kernels.values())


if __name__ == '__main__':

    from kernel_discovery.description.utils import pretty_ast
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
        
    def print_expand():
        
        k = Constant() * RBF()
        
        ast = kernel_to_ast(k)
        
        expanded = expand_asts([ast])
        
        for ast in expanded:
            print(ast_to_text(ast))
        
    # test_expand_asts()
    
    print_expand()
