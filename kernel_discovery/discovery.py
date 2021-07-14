import re
from yfinance import ticker
from kernel_discovery.expansion.grammar import IMPLEMENTED_BASE_KERNEL_NAMES
import logging
from typing import Any, Dict, Optional
from gpflow import kernels

import numpy as np
from gpflow.kernels import White

from kernel_discovery.preprocessing import get_datashape, preprocessing
from kernel_discovery.description import kernel_to_ast, ast_to_text
from kernel_discovery.expansion.expand import expand_asts
from kernel_discovery.evaluation.evaluate import LocalEvaluator, ParallelEvaluator
from collections import defaultdict

class BaseDiscovery(object):

    def __init__(self) -> None:

        self.logger = logging.getLogger(self.__class__.__name__)

    def discover(self, x, y):
        raise NotImplementedError


class ABCDiscovery(BaseDiscovery):

    def __init__(
        self,
        search_depth: int = 6,
        rescale_x_to_upper_bound: Optional[float] = None,
        max_kernels_per_depth: Optional[int] = 1,
        find_n_best: int = 1,
        full_initial_base_kernel_expansion: bool = False,
        early_stopping_min_rel_delta: Optional[float] = None,
        gammar_kwargs: Optional[Dict[str, Any]] = None,
        run_local: Optional[bool]=True,
        num_restarts: int = 3
    ) -> None:

        super().__init__()

        self.search_depth = search_depth
        self.rescale_x_to_upper_bound = rescale_x_to_upper_bound
        self.max_kernels_per_depth = max_kernels_per_depth
        self.find_n_best = find_n_best
        self.full_initial_base_kernel_expansion = full_initial_base_kernel_expansion
        self.early_stopy_min_rel_delta = early_stopping_min_rel_delta
        self.gammar_kwargs = gammar_kwargs
        self.num_restarts = num_restarts

        self.start_ast = kernel_to_ast(White(), include_param=True)

        # init either local or cluster evaluator
        if run_local:
            self.evaluator = LocalEvaluator()
        else:
            self.evaluator = ParallelEvaluator()

    def get_n_best(self, scored_kernels: Dict[str, Dict[str, Any]]):
        return sorted(scored_kernels, key=lambda kernel: scored_kernels[kernel]['score'])[:self.find_n_best]

    def discover(self, x, y):

        x, y = preprocessing(
            x, y, rescale_x_to_upper_bound=self.rescale_x_to_upper_bound)

        self.logger.info(
            f'Starting the kernel discovery with base kernels `{IMPLEMENTED_BASE_KERNEL_NAMES}`')

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
            
            # 1. get best models
            best_previous_kernels = self.get_n_best(scored_kernels)

            # 2. greedily expand from the best one
            to_expand = [scored_kernels[kernel_name]['ast']
                         for kernel_name in best_previous_kernels]
            new_asts = expand_asts(to_expand)

            unscored_asts = [ast for ast in new_asts if ast_to_text(
                ast) not in scored_kernels]
            

            if not unscored_asts:
                stopping_reason = f"Depth `{depth}`: Empty search space, no new asts found"

            # 3. optimize candidate kernels (with hyperparameter restarts)
            result = defaultdict(list)
            for _ in range(self.num_restarts):
                for optimized_ast, noise, score in self.evaluator.evaluate(x, y, unscored_asts):
                    result[ast_to_text(optimized_ast)].append({
                        'ast': optimized_ast,
                        'noise': noise,
                        'score': score,
                        'depth': depth
                    })
            # get the best score among multiple restarts
            result = {k: sorted(result[k], key=lambda x: x['score'])[0]  for k in result.keys()}
            scored_kernels.update(result)
            
            self.logger.info(f"Search depth {depth + 1}/{self.search_depth} is finished")

        self.logger.info(
            f'Finish model search, stopping reason was: \n\n\t{stopping_reason}\n')

        return {
            **{kernel_name: scored_kernels[kernel_name] for kernel_name in self.get_n_best(scored_kernels)},
            'terminate_reason': stopping_reason
        }


class HorseshoeDiscovery(BaseDiscovery):

    def __init__(self) -> None:
        super().__init__()

    def discover(self, x, y):
        # TODO: implement this
        pass


if __name__ == "__main__":

    # unit test
    from kernel_discovery.description.describe import describe
    from kernel_discovery.io import retrieve

    ticker = 'MSFT'
    start = '2021-01-01'
    end = '2021-07-01'
    x, y, ticker = retrieve(ticker_name=ticker, start=start, end=end)

    discovery = ABCDiscovery(run_local=False)

    results = discovery.discover(x, y)
    print(results)
