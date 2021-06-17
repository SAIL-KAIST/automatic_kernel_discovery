import logging
from typing import Any, Dict, Optional

import numpy as np
from gpflow.kernels import White

from kernel_discovery.preprocessing import preprocessing
from kernel_discovery.description import kernel_to_ast, ast_to_text

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
        
        self.search_depth = search_depth,
        self.rescale_x_to_upper_bound = rescale_x_to_upper_bound
        self.max_kernels_per_depth = max_kernels_per_depth
        self.find_n_best = find_n_best
        self.full_initial_base_kernel_expansion = full_initial_base_kernel_expansion
        self.early_stopy_min_rel_delta = early_stopping_min_rel_delta
        self.gammar_kwargs = gammar_kwargs
        
        self.start_ast = kernel_to_ast(White(1))
        
    
    def discover(self):
        
        x, y = preprocessing(self.x, self.y, rescale_x_to_upper_bound=self.rescale_x_to_upper_bound)
        
        
        # self.logger.info(f'Starting the kernel discovery with base kernels `{}`')
        
        scored_kernels = {
            ast_to_text(self.start_ast): {
                'ast': self.start_ast,
                'params': {},
                'score': np.Inf,
                'depth': 0
            }
        }
        for depth in range(self.search_depth):
            
            best_previous_kernel = None
            
            # check if early stopping
            
            
            new_asts = expand_asts()
        
    

class HorseshoeDiscovery(BaseDiscovery):
    
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
    
    def discover(self):
        # TODO: implement this
        pass