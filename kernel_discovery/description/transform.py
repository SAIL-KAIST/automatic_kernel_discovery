from typing import Callable, Optional

from gpflow.utilities.traversal import parameter_dict
from numpy.core.fromnumeric import var

from kernel_discovery.kernel import BASE_KERNELS, COMBINATION_KERNELS

from anytree import Node, node
from kernel_discovery.kernel import Sum, Product, Kernel, White, Linear, RBF, Periodic, Polynomial, Constant


def kernel_to_ast(kernel: Kernel, parent: Optional[Node] = None, include_param=False) -> Node:

    n = Node(type(kernel), parent=parent, full_name=kernel.name)

    if isinstance(kernel, tuple(BASE_KERNELS.values())):
        if include_param:
            n.parameters = [param.numpy() for param in kernel.parameters]
        return n
    elif isinstance(kernel, tuple(COMBINATION_KERNELS.values())):
        for child in kernel.kernels:
            kernel_to_ast(child, parent=n, include_param=include_param)

    return n


def ast_to_kernel(node: Node, init_func: Callable = None) -> Kernel:

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
        sum_str = ' + '.join(sorted(ast_to_text(child)
                             for child in node.children))

        if node.parent is not None and node.parent.name is Product:
            return f'({sum_str})'

        return sum_str

    if node.name is Product:
        return ' * '.join(sorted(ast_to_text(child) for child in node.children))

    return node.name.__name__.lower()
