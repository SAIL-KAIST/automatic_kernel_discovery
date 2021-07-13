
from copy import deepcopy
from operator import le
from re import sub
from typing import Optional, List
import itertools
from gpflow.kernels import Kernel

from kernel_discovery.kernel import Sum, Product

from kernel_discovery.kernel import BASE_KERNELS

IMPLEMENTED_BASE_KERNEL_NAMES = [
    'constant', 'linear', 'periodic', 'rbf', 'white']
_IMPLEMENTED_BASE_KERNEL_NAMES = [BASE_KERNELS[name]
                                  for name in IMPLEMENTED_BASE_KERNEL_NAMES]


def _expand_single(kernel):
    
    kernel_candidates = []
    
    for base_kernel in _IMPLEMENTED_BASE_KERNEL_NAMES:
    
        kernel_candidates.extend([
            kernel + base_kernel(),
            kernel * base_kernel(),
            kernel * (base_kernel() + BASE_KERNELS['constant']())
        ])
    
    return kernel_candidates
    

def expand_kernel(kernel: Kernel, grammar_kwargs=None) -> List[Kernel]:

    kernel_candidates: List[Kernel] = [kernel]

    kernel_candidates.extend(base_kernel()
                             for base_kernel in _IMPLEMENTED_BASE_KERNEL_NAMES)

    # expand single
    kernel_candidates.extend(_expand_single(kernel))

    # expand combination
    kernel_candidates.extend(_expand_combination(kernel))

    return kernel_candidates


def _expand_combination(kernel: Kernel) -> List[Kernel]:

    if isinstance(kernel, (Sum, Product)):
        
        operands = kernel.kernels
        zero_one_pairs = ((0, 1), ) * len(operands)
        all_subsets= itertools.product(*zero_one_pairs)
        # all_subsets where len(operands) == 3: (0, 0, 0), (0, 0, 1), (0, 1, 0) and so one 
        
        results = []
        for subset in all_subsets:
            if (not sum(subset)==0) and (not sum(subset) == len(operands)):
                unexpand = [deepcopy(op) for i, op in enumerate(operands) if not subset[i]]
                to_be_expanded = [deepcopy(op) for i, op in enumerate(operands) if subset[i]]
                
                if len(to_be_expanded) >= 1:
                    # make new kernel without unepanded operands
                    k = deepcopy(kernel)
                    k.kernels = to_be_expanded
                    expansion = _expand_single(kernel)
                else:
                    expansion = expand_kernel(to_be_expanded[0])
                
                for expanded in expansion:
                    k = deepcopy(kernel)
                    k.kernels = [expanded] + unexpand
                    results.append(k)
        
        return results
            
    return []


if __name__ == "__main__":

    from typing import Callable
    from kernel_discovery.kernel import Linear
    from kernel_discovery.description.transform import kernel_to_ast, ast_to_text
    # unit test

    def test_expand_kernel():

        assert isinstance(expand_kernel, Callable)

        expanded_1 = [ast_to_text(kernel_to_ast(k))
                      for k in expand_kernel(Linear())]
        expanded_2 = [ast_to_text(kernel_to_ast(k))
                      for k in expand_kernel(Linear())]

        assert set(expanded_1) == set(expanded_2)

    test_expand_kernel()
