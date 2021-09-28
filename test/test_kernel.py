from gpflow import likelihoods
import pytest

import numpy as np
from kernel_discovery.kernel import Periodic
from gpflow.models.gpr import GPR
from gpflow.optimizers.scipy import Scipy
from kernel_discovery.plot.plot_gp import plot_gp_regression

def test_periodic_kernel():
    
    x = np.linspace(0, 4)[:,None]
    y = np.sin(x)
    
    period = Periodic(variance=0.5, lengthscales=0.5, period=5.)
    
    model = GPR(data=(x, y), kernel=period, noise_variance=0.1)
    
    optimizer = Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables)
    x_extrap = np.linspace(np.min(x), np.max(x), 100)[:,None]
    mean, covar = model.predict_y(x_extrap)
    fig, ax = plot_gp_regression(x, 
                       y,
                       x_extrap,
                       mean.numpy().squeeze(), 
                       covar.numpy().squeeze())
    fig.savefig("dummy.png")

if __name__ == "__main__":
    test_periodic_kernel()
    
    