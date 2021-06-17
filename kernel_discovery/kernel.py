from typing import Dict

import gpflow

from gpflow.kernels import (
    Constant,
    RBF,
    Linear,
    Periodic,
    RationalQuadratic,
    Sum,
    Product,
    White
)

BASE_KERNELS : Dict[str, gpflow.kernels.Kernel] = {
    'constant': Constant,
    'rbf': RBF,
    'linear': Linear,
    'periodic': Periodic,
    'rationalquadratic': RationalQuadratic,
    'white': White
}

COMBINATION_KERNELS: Dict[str, gpflow.kernels.Kernel] = {
    'sum': Sum,
    'product': Product
}

