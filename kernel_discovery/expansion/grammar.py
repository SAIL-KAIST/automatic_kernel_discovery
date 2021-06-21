
from typing import Optional, List
from gpflow.kernels import Kernel

from kernel_discovery.kernel import Sum, Product

from kernel_discovery.kernel import BASE_KERNELS

IMPLEMENTED_BASE_KERNEL_NAMES = ['constant', 'linear', 'periodic', 'rbf', 'white']
_IMPLEMENTED_BASE_KERNEL_NAMES = [BASE_KERNELS[name] for name in IMPLEMENTED_BASE_KERNEL_NAMES]

def expand_kernel(kernel: Kernel, grammar_kwargs=None) -> List[Kernel]:
    
    kernel_candidates: List[Kernel] = [kernel]
    
    kernel_candidates.extend(base_kernel() for base_kernel in _IMPLEMENTED_BASE_KERNEL_NAMES)
    
    for base_kernel in _IMPLEMENTED_BASE_KERNEL_NAMES:
        
        kernel_candidates.extend([
            kernel + base_kernel(),
            kernel * base_kernel(),
            kernel * (base_kernel() + BASE_KERNELS['constant']())
        ])
    
    kernel_candidates.extend(_expand_combination(kernel))
    
    return kernel_candidates
        
def _expand_combination(kernel: Kernel) -> List[Kernel]:
    
    if isinstance(kernel, (Sum, Product)):
        return kernel.kernels
    
    return []


if __name__ == "__main__":
    
    from typing import Callable
    from kernel_discovery.kernel import Linear
    from kernel_discovery.description.transform import kernel_to_ast, ast_to_text
    # unit test
    def test_expand_kernel():
        
        assert isinstance(expand_kernel, Callable)
        
        expanded_1 = [ast_to_text(kernel_to_ast(k)) for k in expand_kernel(Linear())]
        expanded_2 = [ast_to_text(kernel_to_ast(k)) for k in expand_kernel(Linear())]
        
        assert set(expanded_1) == set(expanded_2)
        
    test_expand_kernel()
        