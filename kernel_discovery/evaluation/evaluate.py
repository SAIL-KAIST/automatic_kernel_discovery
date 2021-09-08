from kernel_discovery.preprocessing import DataShape, get_datashape
from operator import mod
from typing import List
import gpflow
import numpy as np
import logging

from anytree import Node
from gpflow import optimizers
from kernel_discovery.description.transform import ast_to_kernel, kernel_to_ast
from kernel_discovery.description.utils import pretty_ast
from gpflow.models.gpr import GPR
from gpflow.optimizers.scipy import Scipy
from kernel_discovery.kernel import RBF, Linear, Periodic, Polynomial, init_kernel

import ray
from ray.tune.integration.mlflow import mlflow_mixin
import mlflow

def nll(model: GPR):

    return model.training_loss()


def bic(model: GPR):

    num_parameters = len(list(model.parameters))
    return 2. * nll(model) + num_parameters * np.log(model.data[0].shape[0])

def init_noise_variance(datashape_y, sd=1.):
    
    if np.random.rand() < 0.5:
        log_var = np.random.normal(loc=datashape_y.std - np.log(10), scale=sd)
    else:
        log_var = np.random.normal(loc=0., scale=sd)
    
    return np.exp(log_var)

class BaseEvaluator(object):

    def __init__(self, score='bic') -> None:
        if score == 'bic':
            self.score = bic
        elif score == 'nll':
            self.score = nll

        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__()

    def evaluate(self, x, y, asts: List[Node]):

        raise NotImplementedError

    def evaluate_single(self, x, y, ast):

        datashape_x, datashape_y = get_datashape(x), get_datashape(y)

        # initialize hyperparameter function w.r.t. each kernel type
        def init_func(kernel_type):
            return init_kernel(kernel_type, datashape_x, datashape_y, sd=1.)

        kernel = ast_to_kernel(ast, init_func=init_func)

        model = GPR(data=(x, y), kernel=kernel, noise_variance=init_noise_variance(datashape_y))

        optimizer = Scipy()
        try:
            opt_logs = optimizer.minimize(
                model.training_loss, model.trainable_variables, options=dict(maxiter=500))
        except:
            self.logger.error(
                f"Error occured when optimized: \n {pretty_ast(ast)}")
            optimized_ast = kernel_to_ast(model.kernel, include_param=True)
            noise_variance = model.likelihood.variance.numpy()
            return optimized_ast, noise_variance, np.Inf

        optimized_ast = kernel_to_ast(model.kernel, include_param=True)
        noise_variance = model.likelihood.variance.numpy()
        return optimized_ast, noise_variance, self.score(model).numpy()


class LocalEvaluator(BaseEvaluator):

    def evaluate(self, x, y, asts: List[Node]):

        for i, ast in enumerate(asts):
            optimized_ast, noise, score = self.evaluate_single(x, y, ast)
            # TODO: return model parameters not optimized_model
            yield optimized_ast, noise, score
            self.logger.info(
                f"{i + 1}/{len(asts)}  Score: {score:.3f} for \n {pretty_ast(ast)}")


class ParallelEvaluator(BaseEvaluator):
    """
    Run parallel jobs on Ray
    """

    def __init__(self, cluster=None) -> None:
        super().__init__()
        self.cluster = cluster
        if cluster:
            ray.init(address=cluster)
        else:
            # if no cluster is specified, make a local one
            ray.init()

    def evaluate(self, x, y, asts: List[Node]):

        # x, y are shared for all jobs. This will reduce network traffic overhead
        x_id = ray.put(x)
        y_id = ray.put(y)

        @ray.remote
        def remote_evaluate(ast):
            x, y = ray.get(x_id), ray.get(y_id)
            return self.evaluate_single(x, y, ast)

        refs = []
        for ast in asts:
            obj_ref = remote_evaluate.remote(ast)
            refs.append(obj_ref)

        return ray.get(refs)
    
    
class TuneEvaluation(BaseEvaluator):
    
    def evaluate(self, x, y, asts: List[Node]):
        
        def train_fn(config):
            x, y, ast = config["x"], config["y"], config["ast"]
            
            


