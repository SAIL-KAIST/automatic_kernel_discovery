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


if __name__ == '__main__':
    
    def test_on_msft():
        from kernel_discovery.io import retrieve
        from kernel_discovery.preprocessing import preprocessing
        from kernel_discovery.description.plot import compute_mean_var, plot_gp
        from kernel_discovery.description import ast_to_kernel
        x, y, ticker = retrieve('MSFT', '2021-01-01', '2021-03-01')

        x, y = preprocessing(x, y, rescale_x_to_upper_bound=None)
        
        evaluator = LocalEvaluator()
        ast = Node(gpflow.kernels.RBF)
        optimized_ast, noise, score = evaluator.evaluate_single(x, y, ast)
        print(optimized_ast)
        print(noise)
        print(score)
        x_extrap = np.linspace(np.min(x), np.max(x), 100)[:,None]
        x_extrap = x
        kernel = ast_to_kernel(optimized_ast)
        mean, var = compute_mean_var(x, x_extrap, y, kernel=kernel, component=kernel, noise=noise)
        fig, ax = plot_gp(x, y, x_extrap, mean, var)
        fig.savefig("dummy.png")
        
        
    
    def test_local_evaluator():

        x = np.array([[0], [1], [2]]).astype(float)
        y = np.array([[0], [1], [2]]).astype(float)

        # test single
        evaluator = LocalEvaluator()
        ast = Node(gpflow.kernels.White)
        evaluator.evaluate_single(x, y, ast)

        # test multiple
        asts = [Node(k) for k in [gpflow.kernels.Linear,
                                  gpflow.kernels.White, gpflow.kernels.RBF]]
        evaluator.evaluate(x, y, asts)

    def test_parallel_evaluator_single_machine():

        x = np.array([[0], [1], [2]]).astype(float)
        y = np.array([[0], [1], [2]]).astype(float)

        # test single
        evaluator = ParallelEvaluator()
        ast = Node(gpflow.kernels.White)
        evaluator.evaluate_single(x, y, ast)

        # test multiple
        asts = [Node(k) for k in [gpflow.kernels.Linear,
                                  gpflow.kernels.White, gpflow.kernels.RBF]]
        result = evaluator.evaluate(x, y, asts)
        print(result)

    # test_local_evaluator()
    # test_parallel_evaluator_single_machine()
    test_on_msft()
