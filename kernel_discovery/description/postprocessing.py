import numpy as np
from anytree import Node


from kernel_discovery.description.plot import component_stats, order_by_mae_reduction, cummulative_plots, checking_stats
from kernel_discovery.description.describe import produce_summary, 


        
    
        

def make_all_figures_and_report(x: np.array, y: np.array, kernel: Node, noise: np.arrary):
    
    order, evaluated_data = order_by_mae_reduction(x, y, kernel, noise)
    component_data = component_stats(x, y, kernel, noise, order)
    cumulative_data = cummulative_plots(x, y, kernel, noise, order)
    checking_data = checking_stats(x, y, kernel, noise, order)
    
    evaluated_data.update(component_data)
    evaluated_data.update(cumulative_data)
    evaluated_data.update(checking_data)
    
    