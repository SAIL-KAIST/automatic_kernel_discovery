import numpy as np
import gpflow
from anytree import Node
from kernel_discovery.evaluation.evaluate import LocalEvaluator, ParallelEvaluator
from kernel_discovery.io import retrieve
from kernel_discovery.preprocessing import preprocessing
from kernel_discovery.description.plot import compute_mean_var, plot_gp
from kernel_discovery.description import ast_to_kernel

def test_evaluate_on_msft():
    
    x, y, ticker = retrieve('MSFT', '2021-01-01', '2021-03-01')

    x, y = preprocessing(x, y, rescale_x_to_upper_bound=None)
    
    evaluator = LocalEvaluator()
    ast = Node(gpflow.kernels.RBF)
    optimized_ast, noise, score = evaluator.evaluate_single(x, y, ast)
    x_extrap = np.linspace(np.min(x), np.max(x), 100)[:,None]
    x_extrap = x
    kernel = ast_to_kernel(optimized_ast)
    mean, var = compute_mean_var(x, x_extrap, y, kernel=kernel, component=kernel, noise=noise)
    fig, ax = plot_gp(x, y, x_extrap, mean, var)
    
    

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