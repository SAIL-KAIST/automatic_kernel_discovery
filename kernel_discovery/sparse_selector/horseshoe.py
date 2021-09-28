from site import main
from gpflow.likelihoods.base import Likelihood
import numpy as np
import tensorflow as tf
from gpflow.models.svgp import SVGP
from gpflow.likelihoods import Gaussian

from kernel_discovery.discovery import BaseDiscovery

from kernel_discovery.sparse_selector.structural_sgp import StructuralSVGP
from kernel_discovery.sparse_selector.base_model import HorseshoeSelector

from kernel_discovery.kernel import RBF


class HorseshoeDiscovery(BaseDiscovery):
    
    def __init__(self, 
                 n_inducings,
                 kernel_order=2,
                 ) -> None:
        super().__init__()
        self.n_inducings = n_inducings
        self.kernel_order = kernel_order
        
    def generate_kernels(self):
        return [RBF()]
    
    def sample_inducing_points(self, x):
        return x[:self.n_inducings]
    
    def discover(self, x, y):
        # TODO: implement this
        # parameters
        n_iter = int(1e3)
        lr= 1e-3
        batch_size = 128
        
        kernels = self.generate_kernels()
        gps = []
        for kernel in kernels:
            gp = SVGP(kernel=kernel, 
                      likelihood=None, 
                      inducing_variable=self.sample_inducing_points(x),
                      q_mu=np.random.randn(self.n_inducings, 1))
            gps.append(gp)
        
        selector = HorseshoeSelector(dim=len(gps))
        model = StructuralSVGP(gps, selector, likelihood=Gaussian())
        
        # data
        train_iter = make_data_iteration(x, y, batch_size)
        
        optimizer = tf.optimizers.Adam(lr=lr)
        
        train_loss = model.training_loss_closure(train_iter)
        
        @tf.function
        def optimize_step():
            optimizer.minimize(train_loss, model.trainable_variables)
            model.update_tau_lambda()
        
        for i in range(n_iter):
            optimize_step()
        

def make_data_iteration(x, y, batch_size=128, shuffle=False):
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.repeat().shuffle(buffer_size=1024, seed=123)
    
    data_iter = iter(dataset.batch(batch_size=batch_size))
    return data_iter
