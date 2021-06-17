
from typing import Optional, List
from gpflow.kernels import Kernel

from kernel_discovery.kernel import BASE_KERNELS

IMPLEMENTED_BASE_KERNEL_NAMES = ['constant', 'linear', 'periodic', 'rbf', 'white']
_IMPLEMENTED_BASE_KERNEL_NAMES = [BASE_KERNELS[name] for name in IMPLEMENTED_BASE_KERNEL_NAMES]

def expand_kernel(kernel: Kernel, base_kernels_to_exclude: Optional[List[str]]=None) -> List[Kernel]:
    
    if base_kernels_to_exclude is None:
        base_kernels_to_exclude = []
        