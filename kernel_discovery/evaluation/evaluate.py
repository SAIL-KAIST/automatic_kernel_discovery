from typing import List
import gpflow
import numpy as np
import logging

from anytree import Node
from gpflow import optimizers
from kernel_discovery.description.transform import ast_to_kernel
from kernel_discovery.description.utils import pretty_ast
from gpflow.models.gpr import GPR
from gpflow.optimizers.scipy import Scipy

def nll(model:GPR):
    
    return model.training_loss()

def bic(model:GPR):
    
    num_parameters = len(list(model.parameters))
    return 2. * nll(model) + num_parameters * np.log(model.data[0].shape[0])

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
    
    def initialize_hyperparmeters(self, kernel):
        # TODO: implement this. hyperparameters are intialized according to its data shape
        return kernel
    
    
    def evaluate_single(self, x, y, ast):
        kernel = ast_to_kernel(ast)
        kernel = self.initialize_hyperparmeters(kernel)
        model = GPR(data=(x, y), kernel=kernel)
        
        optimizer = Scipy()
        try:
            opt_logs = optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        except:
            self.logger.error(f"Error occured when optimized: \n {pretty_ast(ast)}")
            return model, np.Inf
        
        return model, self.score(model)
        
        
class LocalEvaluator(BaseEvaluator):
    
    def evaluate(self, x, y, asts: List[Node]):
        
        for i, ast in enumerate(asts):
            optimized_model, score = self.evaluate_single(x, y, ast)
            print(f"{i + 1}/{len(asts)}  Score: {score:.3f} for \n {pretty_ast(ast)}")
            # yield ast, optimized_model, score
            # self.logger.info(f"{i + 1}/{len(asts)}  Score: {score:.3f} for \n {pretty_ast(ast)}")
            
    

class ParallelEvaluation(BaseEvaluator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
    def evaluate(x, y, asts: List[Node]):
        return super().evaluate(y, asts)
    

if __name__ == '__main__':
    
    def test_local_evaluator():
        
        x = np.array([[0], [1], [2]]).astype(float)
        y = np.array([[0], [1], [2]]).astype(float)
        
        # test single
        evaluator = LocalEvaluator()
        ast = Node(gpflow.kernels.White)
        evaluator.evaluate_single(x, y, ast)
        
        # test multiple
        asts = [Node(k) for k in [gpflow.kernels.Linear, gpflow.kernels.White, gpflow.kernels.RBF]]
        evaluator.evaluate(x, y, asts)
        
        
        
        
    test_local_evaluator()