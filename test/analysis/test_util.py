from re import I
from gpflow.kernels import RBF
import numpy as np
from kernel_discovery.analysis.util import compute_mean_var
from kernel_discovery.kernel import RBF

def test_compute_mean_var():
    
    x = np.linspace(0, 5, 100)[:, None]
    x_extrap = np.linspace(0, 5.5, 300)[:, None]
    y = np.sin(x)

    kernel = RBF() + RBF(lengthscales=0.5)
    component = RBF()

    mean, var = compute_mean_var(
        x, x_extrap, y, kernel, component, noise=0.1)