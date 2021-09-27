from gpflow import likelihoods
import pytest

import numpy as np
from kernel_discovery.kernel import Periodic
from gpflow.models.gpr import GPR
from gpflow.optimizers.scipy import Scipy

def test_periodic_kernel():
    
    x = np.linspace(0, 1)[:,None]
    y = np.sin(x)
    
    period = Periodic(variance=0.5, lengthscales=0.5, period=5.)
    
    model = GPR(data=(x, y), kernel=period, noise_variance=0.1)
    
    optimizer = Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables)
    
    
    