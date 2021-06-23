from typing import Dict

import gpflow

from gpflow.kernels import (
    Constant,
    Kernel,
    RBF,
    Linear,
    Periodic as gpflow_Periodic,
    RationalQuadratic,
    Polynomial,
    Sum,
    Product,
    White
)

class Periodic(gpflow_Periodic):
    
    def __init__(self, variance=1., lengthscales=1., period=1.):
        base_kernel = RBF(variance=variance, lengthscales=lengthscales)
        super().__init__(base_kernel, period=period)

BASE_KERNELS : Dict[str, Kernel] = {
    'constant': Constant,
    'rbf': RBF,
    'linear': Linear,
    'periodic': Periodic,
    'rationalquadratic': RationalQuadratic,
    'polynomial': Polynomial,
    'white': White
}

COMBINATION_KERNELS: Dict[str, Kernel] = {
    'sum': Sum,
    'product': Product
}

