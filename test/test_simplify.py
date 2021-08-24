import numpy as np
from anytree import Node
from anytree import LevelOrderIter
from kernel_discovery.kernel import Periodic, Product, RBF, Sum, White, Linear, Polynomial, ChangePoints
from kernel_discovery.description.transform import kernel_to_ast, ast_to_kernel
from kernel_discovery.description.simplify import distribution, merge_rbfs, replace_white_product, simplify


def are_asts_equal(ast1, ast2):

    for a, b in zip(LevelOrderIter(ast1), LevelOrderIter(ast2)):
        # check if they have the same name
        if a.name != b.name:
            return False
        # check if they have same parameters
        if not hasattr(a, "parameters") and not hasattr(b, "parameters"):
            continue
        if hasattr(a, "parameters") and not hasattr(b, "parameters"):
            return False
        elif not hasattr(a, "parameters") and hasattr(b, "parameters"):
            return False
        elif not np.array_equal(a.parameters, b.parameters):
            return False
    return True

def test_distribution():
    k = (RBF() + White() * Linear()) * Polynomial()
    ast = kernel_to_ast(k)

    # Level 1.
    ast_should_be = Node(Sum, full_name='Sum')

    # Level 2.
    p1 = Node(Product, full_name='Product', parent=ast_should_be)
    p2 = Node(Product, full_name='Product', parent=ast_should_be)

    # Level 3.1
    Node(Polynomial, parent=p1)
    Node(RBF, parent=p1)

    # Level 3.2
    Node(Polynomial, parent=p2)
    Node(White, parent=p2)
    Node(Linear, parent=p2)

    assert are_asts_equal(distribution(ast), ast_should_be)

def test_merge_rbfs():

    k = ((RBF() * RBF() + RBF() + RBF() + White() * Linear())) * \
        Polynomial() * RBF() * RBF()

    ast = kernel_to_ast(k)
    merged_rbf_ast = merge_rbfs(ast)

    ast_should_be = Node(Product, full_name='Product')

    p1 = Node(Sum, full_name='Sum', parent=ast_should_be)
    Node(Polynomial, full_name='Polynomial', parent=ast_should_be)
    Node(RBF, full_name='RBF', parent=ast_should_be)

    Node(RBF, full_name='RBF', parent=p1)
    Node(RBF, full_name='RBF', parent=p1)
    p2 = Node(Product, full_name='Product', parent=p1)
    Node(RBF, full_name='RBF', parent=p1)

    Node(White, full_name='White', parent=p2)
    Node(Linear, full_name='Linear', parent=p2)

    assert are_asts_equal(merged_rbf_ast, ast_should_be)

def test_replace_white_products():

    k1 = Product([White(), White(), White(), White()])
    ast1 = kernel_to_ast(k1)
    ast1_should_be = Node(White)
    assert are_asts_equal(ast1_should_be, replace_white_product(ast1))

    k2 = White() + White()
    ast2 = kernel_to_ast(k2)
    ast2_should_be = Node(Sum)
    Node(White, parent=ast2_should_be)
    Node(White, parent=ast2_should_be)
    assert are_asts_equal(ast2_should_be, replace_white_product(ast2))

    k3 = Product([White(), White(), White(), White(),
                    RBF(), Linear(), Polynomial()]) + RBF()

    ast3 = kernel_to_ast(k3)
    ast3_should_be = Node(Sum)
    p1 = Node(Product, parent=ast3_should_be)
    Node(RBF, parent=ast3_should_be)

    Node(White, parent=p1)
    Node(Linear, parent=p1)
    Node(Polynomial, parent=p1)

    assert are_asts_equal(ast3_should_be, replace_white_product(ast3))

    k = RBF() * White() + Periodic() * White()

    ast = kernel_to_ast(k)

    ast_should_be = Node(Sum, full_name='Sum')
    Node(White, full_name='White', parent=ast_should_be)
    Node(White, full_name='White', parent=ast_should_be)

    assert are_asts_equal(ast_should_be, replace_white_product(ast))

def test_simplify():

    k = (Linear(variance=2.) + RBF(lengthscales=2.)) * Periodic() * RBF() + \
        White() * Linear() * Periodic() * White()

    ast = kernel_to_ast(k)

    ast_should_be = Node(Sum, full_name='Sum')

    prod1 = Node(Product, full_name='Product', parent=ast_should_be)
    Node(Periodic, full_name='Periodic', parent=prod1)
    Node(Linear, full_name='Linear', parent=prod1)
    Node(RBF, full_name='RBF', parent=prod1)

    prod2 = Node(Product, full_name='Product', parent=ast_should_be)
    Node(Periodic, full_name='Periodic', parent=prod2)
    Node(RBF, full_name='RBF', parent=prod2)

    prod3 = Node(Product, full_name='Product', parent=ast_should_be)
    Node(White, full_name='White', parent=prod3)
    Node(Linear, full_name='Linear', parent=prod3)

    assert are_asts_equal(ast_should_be, simplify(ast))
    assert not are_asts_equal(ast, simplify(ast))
    
def test_simplified_has_parameters():
    k = Linear(variance=2.) + RBF(lengthscales=2.)
    ast = kernel_to_ast(k, include_param=True)
    
    simplified_ast = simplify(ast)
    
    assert are_asts_equal(ast, simplified_ast)