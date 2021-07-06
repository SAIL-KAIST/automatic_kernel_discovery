from kernel_discovery.kernel import Periodic
from kernel_discovery.description.transform import kernel_to_ast
import numpy as np
from anytree import Node


from kernel_discovery.description.plot import Result
from kernel_discovery.description.describe import produce_summary

# def generate_

def make_all_figures_and_report(x: np.array, y: np.array, kernel: Node, noise: np.array):
    
    result = Result(x, y, kernel, noise, root="./figure/")
    result.process()
    
    for component in result.components:
        component.make_description()
        
    produce_summary("Some name", result)
    
    

if __name__ == '__main__':

    from kernel_discovery.kernel import Linear, RBF
    dataset_name = "Some Name"
    k = Linear() * RBF() + Periodic()
    prod = kernel_to_ast(k)
    
    x = np.linspace(0,1,100)[:,None]
    y = np.sin(x)
    make_all_figures_and_report(x, y, prod, noise=np.array(0.1))
    
    
        
    
    