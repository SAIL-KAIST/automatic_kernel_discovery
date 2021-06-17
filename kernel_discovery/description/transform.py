
from typing import Optional


from anytree import Node
from gpflow.kernels import Kernel

def kernel_to_ast(kernel: Kernel, parent: Optional[Node]=None) -> Node:
    
    n = Node(type(kernel), parent=parent, full_name=kernel.name)
    
    # TODO: traverse all the children
    

def ast_to_kernel(node: Node, build=False) -> Kernel:
    
    # TODO: implement this
    raise NotImplementedError


def ast_to_text(node) -> str:
    
    # TODO: implement this
    raise NotImplementedError