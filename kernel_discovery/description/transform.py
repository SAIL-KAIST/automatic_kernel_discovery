from typing import Callable, Optional

from gpflow.utilities.traversal import parameter_dict
from numpy.core.fromnumeric import var

from kernel_discovery.kernel import BASE_KERNELS, COMBINATION_KERNELS

from anytree import Node, node
from kernel_discovery.kernel import Sum, Product, Kernel, White, Linear, RBF, Periodic, Polynomial, Constant

def kernel_to_ast(kernel: Kernel, parent: Optional[Node]=None, include_param=False) -> Node:
    
    n = Node(type(kernel), parent=parent, full_name=kernel.name)
    
    if isinstance(kernel, tuple(BASE_KERNELS.values())):
        if include_param:
            n.parameters = [param.numpy() for param in kernel.parameters]
        return n
    elif isinstance(kernel, tuple(COMBINATION_KERNELS.values())):
        for child in kernel.kernels:
            kernel_to_ast(child, parent=n, include_param=include_param)
            
    return n

def ast_to_kernel(node: Node, init_func: Callable=None) -> Kernel:
    
    if node.is_leaf:
        kernel = node.name()
        if hasattr(node, 'parameters'):
            for k_param, n_param in zip(kernel.parameters, node.parameters):
                k_param.assign(n_param)
        else:
            if init_func is not None:
                params = init_func(node.name)
                for k_param, param in zip(kernel.parameters, params):
                    k_param.assign(param)
        return kernel
    
    return node.name([ast_to_kernel(child) for child in node.children])


def ast_to_text(node: Node) -> str:
    
    if node.name is Sum:    
        sum_str = ' + '.join(sorted(ast_to_text(child) for child in node.children))
        
        if node.parent is not None and node.parent.name is Product:
            return f'({sum_str})'
        
        return sum_str
    
    if node.name is Product:
        return ' * '.join(sorted(ast_to_text(child) for child in node.children))
    
    return node.name.__name__.lower()


if __name__ == "__main__":
    
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
        kernel = (RBF(variance=2., lengthscales=4.) + White(variance=1.5) * Linear(variance=5.)) * Polynomial(variance=1., offset=1)
        
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
    
    # test_kernel_to_ast()
    
    # test_ast_to_kernel()
    
    # test_ast_to_text()
    
    test_has_parameters()
    
    # all tests passed