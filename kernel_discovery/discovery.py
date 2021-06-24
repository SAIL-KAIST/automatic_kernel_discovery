from kernel_discovery.description.transform import ast_to_kernel
from kernel_discovery.expansion.grammar import IMPLEMENTED_BASE_KERNEL_NAMES
import logging
from typing import Any, Dict, Optional
from gpflow import kernels

import numpy as np
from gpflow.kernels import White

from kernel_discovery.preprocessing import preprocessing
from kernel_discovery.description import kernel_to_ast, ast_to_text
from kernel_discovery.expansion.expand import expand_asts
from kernel_discovery.evaluation.evaluate import LocalEvaluator, ParallelEvaluator



class BaseDiscovery(object):
    
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def discover(self):
        raise NotImplementedError
    

class ABCDiscovery(BaseDiscovery):
    
    def __init__(
        self, 
        x, 
        y,
        search_depth: int = 6,
        rescale_x_to_upper_bound: Optional[float]=None,
        max_kernels_per_depth: Optional[int]=1,
        find_n_best: int=1,
        full_initial_base_kernel_expansion: bool=False,
        early_stopping_min_rel_delta: Optional[float]=None,
        gammar_kwargs: Optional[Dict[str, Any]]=None
        ) -> None:
        
        super().__init__(x, y)
        
        self.search_depth = search_depth
        self.rescale_x_to_upper_bound = rescale_x_to_upper_bound
        self.max_kernels_per_depth = max_kernels_per_depth
        self.find_n_best = find_n_best
        self.full_initial_base_kernel_expansion = full_initial_base_kernel_expansion
        self.early_stopy_min_rel_delta = early_stopping_min_rel_delta
        self.gammar_kwargs = gammar_kwargs
        
        self.start_ast = kernel_to_ast(White(), include_param=True)
        
        # init either local or cluster evaluator
        self.evaluator = ParallelEvaluator()
    
    def get_n_best(self, scored_kernels: Dict[str, Dict[str, Any]]):
        return sorted(scored_kernels, key=lambda kernel: scored_kernels[kernel]['score'])[:self.find_n_best]
    
    def discover(self):
        
        x, y = preprocessing(self.x, self.y, rescale_x_to_upper_bound=self.rescale_x_to_upper_bound)
        
        self.logger.info(f'Starting the kernel discovery with base kernels `{IMPLEMENTED_BASE_KERNEL_NAMES}`')
        
        stopping_reason = f"Depth `{self.search_depth} - 1`: Reached maximum depth"
        scored_kernels = {
            ast_to_text(self.start_ast): {
                'ast': self.start_ast,
                'noise': 0.,
                'score': np.Inf,
                'depth': 0
            }
        }
        
        best_previous_kernels = self.get_n_best(scored_kernels)
        
        
        for depth in range(self.search_depth):
            
            best_previous_kernels = self.get_n_best(scored_kernels)
            
            to_expand = [scored_kernels[kernel_name]['ast'] for kernel_name in best_previous_kernels]
            new_asts = expand_asts(to_expand)
            
            unscored_asts = [ast for ast in new_asts if ast_to_text(ast) not in scored_kernels]
            
            if not unscored_asts:
                stopping_reason = f"Depth `{depth}`: Empty search space, no new asts found"
            
            for optimized_ast, noise, score in self.evaluator.evaluate(x, y, unscored_asts):
                scored_kernels[ast_to_text(optimized_ast)] = {
                    'ast': optimized_ast,
                    'noise':noise,
                    'score': score,
                    'depth':depth
                }
                
        self.logger.info(f'Finish model search, stopping reason was: \n\n\t{stopping_reason}\n')
        
        return {
            **{kernel_name: scored_kernels[kernel_name] for kernel_name in self.get_n_best(scored_kernels)},
            'terminate_reason': stopping_reason
        }
        
    

class HorseshoeDiscovery(BaseDiscovery):
    
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
    
    def discover(self):
        # TODO: implement this
        pass
    

if __name__ == "__main__":
    
    # unit test
    from kernel_discovery.description.describe import describe
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    
    discovery = ABCDiscovery(x, y)
    
    results = discovery.discover()
    desc = describe(list(results.values())[0]['ast'])
    print(results)
    print(desc)
    