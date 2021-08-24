import numpy as np
from anytree import Node
from copy import deepcopy

from gpflow.kernels.statics import Constant
from kernel_discovery.description.transform import kernel_to_ast, ast_to_kernel
from kernel_discovery.kernel import Periodic, Product, RBF, Sum, White, Linear, Polynomial, ChangePoints


def simplify(node: Node):

    to_simplify = deepcopy(node)
    return replace_white_product(
        merge_rbfs(
            distribution(
                to_simplify
            )
        )
    )


def replace_white_product(node: Node):
    node = deepcopy(node)
    _replace_white_product(node)
    return node


def _replace_white_product(node: Node):

    if node.is_leaf:
        return

    if node.name is Product:
        white_children = [
            child for child in node.children if child.name is White]
        if white_children:
            nonstationary_childen = [child for child in node.children
                                     if child.name in [Linear, Polynomial]]
            new_kids = [white_children[0]] + nonstationary_childen
            if len(new_kids) == 1:
                if node.is_root:
                    node.name = new_kids[0].name
                    try:
                        node.full_name = new_kids[0].full_name
                    except AttributeError:
                        pass
                else:
                    new_kids[0].parent = node.parent
                    node.parent = None
                node.children = []
            else:
                node.children = new_kids

    for child in node.children:
        _replace_white_product(child)


def merge_rbfs(node: Node):

    node = deepcopy(node)
    _merge_rbfs(node)
    return node


def _merge_rbfs(node: Node):

    if node.is_leaf:
        return

    if node.name is Product:
        rbf_children = [child for child in node.children if child.name is RBF]
        other_children = [
            child for child in node.children if child.name is not RBF]

        new_kids = other_children + rbf_children[:1]
        if len(new_kids) == 1:
            if node.is_root:
                node.name = new_kids[0].name
                try:
                    node.full_name = new_kids[0].full_name
                except AttributeError:
                    pass
            else:
                new_kids[0].parent = node.parent
                node.parent = None
            node.children = []
        else:
            node.children = new_kids

    for child in node.children:
        _merge_rbfs(child)


def distribution(node: Node):

    node = deepcopy(node)
    _distribution(node)
    return kernel_to_ast(ast_to_kernel(node), include_param=True)


def _distribution(node: Node):

    if node.is_leaf:
        return

    if node.name is Product:
        sum_to_distribute = [
            child for child in node.children if child.name is Sum]

        if sum_to_distribute:
            sum_to_distr = sum_to_distribute[0]
            children_to_distribute_to = [
                child for child in node.children if child is not sum_to_distr]

            node.name = Sum
            node.full_name = 'Sum'
            node.children = []

            for child in sum_to_distr.children:
                new_prod = Node(name=Product, full_name='Product', parent=node)

                new_kids = [deepcopy(child)
                            for child in children_to_distribute_to]
                if child.name is Product:
                    new_kids.extend([deepcopy(c) for c in child.children])
                else:
                    new_kids += [child]

                for kid in new_kids:
                    kid.parent = new_prod

    for child in node.children:
        _distribution(child)


def extract_envelop(node: Node):

    node = deepcopy(node)
    _extract_envelop(node)
    return node


def _extract_envelop(node: Node):

    if node.is_leaf:
        node.name = Constant
        node.parameters = [np.array(1.)]
        return

    for child in node.children:
        _extract_envelop(child)


