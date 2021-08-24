from kernel_discovery.description.transform import ast_to_kernel, kernel_to_ast, ast_to_text
from kernel_discovery.kernel import BASE_KERNELS, COMBINATION_KERNELS

from gpflow.kernels import Linear, White, RBF, Polynomial, Product, Sum
from anytree import LevelOrderIter, Node
from random import choice
from gpflow.utilities import print_summary

def test_kernel_to_ast():

    kernel = (RBF() + White() * Linear()) * Polynomial()

    ast_kernel = kernel_to_ast(kernel)

    ast_manual = Node(Product)
    sum_ = Node(Sum, parent=ast_manual)
    Node(Polynomial, parent=ast_manual)

    Node(RBF, parent=sum_)
    prod = Node(Product, parent=sum_)

    Node(White, parent=prod)
    Node(Linear, parent=prod)

    for a, b in zip(LevelOrderIter(ast_kernel), LevelOrderIter(ast_manual)):
        assert a.name == b.name, "Not the same tree"

def test_ast_to_kernel():

    available_kernels = list(BASE_KERNELS.values())

    root = Node(COMBINATION_KERNELS['sum'])

    for _ in range(5):
        Node(choice(available_kernels), parent=root)

    prod = Node(COMBINATION_KERNELS['product'], parent=root)

    for _ in range(5):
        Node(choice(available_kernels), parent=prod)

    sum_ = Node(COMBINATION_KERNELS['sum'], parent=prod)

    for _ in range(5):
        Node(choice(available_kernels), parent=sum_)

    asts = [root, prod, sum_]
    kernels = [ast_to_kernel(ast) for ast in asts]

    for ast, kernel in zip(asts, kernels):
        kernel_ast = kernel_to_ast(kernel)
        for a, b in zip(LevelOrderIter(ast), LevelOrderIter(kernel_ast)):
            assert a.name == b.name, 'Not the same'

def test_ast_to_text():

    ast_manual = Node(Product)

    sum_ = Node(Sum, parent=ast_manual)
    Node(Polynomial, parent=ast_manual)

    Node(RBF, parent=sum_)
    prod = Node(Product, parent=sum_)

    Node(White, parent=prod)
    Node(Linear, parent=prod)

    ast_str = ast_to_text(ast_manual)

    assert ast_str == '(linear * white + squaredexponential) * polynomial'

def test_has_parameters():
    kernel = (RBF(variance=2., lengthscales=4.) + White(variance=1.5)
                * Linear(variance=5.)) * Polynomial(variance=1., offset=1)

    ast_kernel = kernel_to_ast(kernel, include_param=True)

    ast_manual = Node(Product)
    sum_ = Node(Sum, parent=ast_manual)
    Node(Polynomial, parent=ast_manual)

    Node(RBF, parent=sum_)
    prod = Node(Product, parent=sum_)

    Node(White, parent=prod)
    Node(Linear, parent=prod)

    for a, b in zip(LevelOrderIter(ast_kernel), LevelOrderIter(ast_manual)):
        assert a.name == b.name, "Not the same tree"

    kernel_after = ast_to_kernel(ast_kernel)

    print_summary(kernel_after)