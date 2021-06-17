from typing import List

from anytree import Node

class BaseEvaluator(object):
    
    
    def evaluate(self, x, y, asts: List[Node]):
        
        raise NotImplementedError
    
    
    def evaluate_single(self, x, y, ast):
        # TODO: implement this
        pass
    
    
class LocalEvaluator(BaseEvaluator):
    
    def evaluate(self, x, y, asts: List[Node]):
        
        for i, ast in enumerate(asts):
            self.evaluate_single(x, y, ast)
            
    

class ParallelEvaluation(BaseEvaluator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
    def evaluate(x, y, asts: List[Node]):
        return super().evaluate(y, asts)