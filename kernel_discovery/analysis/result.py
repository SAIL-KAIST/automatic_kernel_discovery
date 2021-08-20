import numpy as np
import tensorflow as tf
from anytree import Node
from gpflow.kernels import Sum, Kernel
from sklearn.model_selection import KFold

from kernel_discovery.description import kernel_to_ast, simplify, ast_to_kernel, extract_envelop
from kernel_discovery.analysis.util import (gaussian_conditional, compute_mean_var, get_gradient, get_monotonic)
from kernel_discovery.description.transform import ast_to_kernel
from kernel_discovery.description.plot import plot_gp, sample_plot_gp

left_extend = 0.
right_extend = 0.1
num_interp_points = 1000

class Result():
    
    def __init__(self, x: np.array, y: np.array, ast: Node, noise:np.array) -> None:
        self.x = x
        self.y = y
        
        def extract(input_ast, func):
            transformed_ast = func(input_ast)
            kernel = ast_to_kernel(transformed_ast)
            return kernel.kernels if isinstance(kernel, Sum) else [kernel]
        
        self.complete_kernel = ast_to_kernel(ast)
        
        self.kernels = extract(ast, simplify)
        self.envelop_kernels = extract(ast, extract_envelop)
        self.noise = noise
        
        self.n_components = len(self.kernels)
        
        self.components = []
        
        # some general stats for sum kernel
        self.mav_data = np.mean(np.abs(y))
        
        # # misc i.e. plot
        x_left = np.min(x) - (np.max(x) - np.min(x)) * left_extend
        x_right = np.max(x) + (np.max(x) - np.min(x)) * right_extend
        self.x_range = np.linspace(x_left, x_right, num_interp_points)[:, None]
        self.xrange_no_extrap = np.linspace(np.min(x), np.max(x), num_interp_points)[:, None]

    
    def order_by_mae_reduction(self):
        """Decide order of each component by MAE"""
        
        print("Decide order of each component by MAE")
        n_folds = 10
        kf = KFold(n_splits=n_folds)

        idx = []
        maes = np.zeros((self.n_components, 1))
        mae_reductions = np.zeros((self.n_components, 1))
        mav_data = np.mean(np.abs(self.y))

        # stack
        cumulative_kernels = []

        def validate_errors(kernel, x_train, y_train, x_val, y_val):
            K_train = kernel.K(x_train)
            if self.noise is not None:
                K_train = K_train + \
                    tf.eye(x_train.shape[0], dtype=K_train.dtype) * self.noise

            K_train_val = kernel.K(x_train, x_val)
            K_val = kernel.K(x_val)
            y_predict, _ = gaussian_conditional(Kmn=K_train_val,
                                                Kmm=K_train,
                                                Knn=K_val,
                                                f=y_train,
                                                full_cov=False)

            error = np.mean(np.abs(y_val - y_predict))
            return error

        previous_mae = mav_data
        for i in range(self.n_components):
            best_mae = np.Inf
            for j in range(self.n_components):
                if j not in idx:
                    # pop first
                    cumulative_kernels.append(self.kernels[j])
                    current_kernel = Sum(cumulative_kernels)
                    this_mae = 0.
                    # train 10-fold cross-validate
                    for train_idx, val_idx in kf.split(self.x):
                        x_train, y_train = self.x[train_idx], self.y[train_idx]
                        x_val, y_val = self.x[val_idx], self.y[val_idx]
                        error = validate_errors(
                            current_kernel, x_train, y_train, x_val, y_val)
                        this_mae += error
                    this_mae /= n_folds

                    if this_mae < best_mae:
                        best_j = j
                        best_mae = this_mae

                    cumulative_kernels.pop()

            maes[i] = best_mae
            mae_reductions[i] = (1. - best_mae/previous_mae) * 100
            previous_mae = best_mae
            idx += [best_j]
            cumulative_kernels.append(self.kernels[best_j])
        
        # create components by the order
        assert len(self.components) == 0
        for i in idx:
            component = {}
            component["kernel"] = kernel_to_ast(self.kernels[i])
            component["envelop"] = kernel_to_ast(self.envelop_kernels[i])
            component["mae"] = float(maes[i].squeeze())
            component["mae_reduction"] = float(mae_reductions[i].squeeze())
            
            self.components += [component]
        assert len(self.components) == self.n_components
        
        return self.components
        
    def full_posterior_plot(self):
        
        print("Plot full posterior")
        
        complete_mean, complete_var = compute_mean_var(self.x, 
                                                       self.x_range, 
                                                       self.y, 
                                                       kernel=self.complete_kernel, 
                                                       component=self.complete_kernel, 
                                                       noise=self.noise)

        # plot raw data
        raw_fig, _ = plot_gp(self.x, self.y, self.x_range, complete_mean, complete_var, data_only=True)
        print(f"Plot raw data")

        # plot full posterior
        fit_fig, _= plot_gp(self.x, self.y, self.x_range, complete_mean, complete_var)
        print((f"Post full posterior"))
        
        # plot sample from full posterior
        complete_mean, complete_covar = compute_mean_var(self.x, 
                                                         self.x_range, 
                                                         self.y, 
                                                         kernel=self.complete_kernel, 
                                                         component=self.complete_kernel,
                                                         noise=self.noise, 
                                                         full_cov=True)
        sample_fig, _ = sample_plot_gp(self.x, self.x_range, complete_mean, complete_covar)
        print(f"Post sample")
        
        return raw_fig, fit_fig, sample_fig
    

class DownstreamAnalysis(Result):
    
    def __init__(self, x, y, ast, noise) -> None:
        super().__init__(x, y, ast, noise)
    
    def load_components(self, components):
        if not len(components) == self.n_components:
            raise ValueError("Unsucessfully load component. Reason: input components has different length")
        
        self.components = components
    
    def analyze(self):
        raise NotImplemented
    
class IndividualAnalysis(DownstreamAnalysis):
    
    def __init__(self, x, y, ast, noise) -> None:
        super().__init__(x, y, ast, noise)
    
    def analyze(self):
        print("Compute individual component statistics and generate individual plots")
        components = self.components
        if len(components) == 0:
            raise RuntimeError("Components have not been loaded yet. Run `load_components()` first!")

        list_figs = []

        for i, component in enumerate(components):
            
            comp = ast_to_kernel(component["kernel"])
            envelop_comp = ast_to_kernel(component["envelop"])
            mean, var = compute_mean_var(self.x,
                                         self.xrange_no_extrap, 
                                         self.y, 
                                         kernel=self.complete_kernel,
                                         component=comp, 
                                         noise=self.noise)
            
            # this is for compute some statistic
            d_mean, d_var = compute_mean_var(self.x, self.x, self.y, self.complete_kernel, comp, self.noise)

            # computes some statistics
            component["snr"] = 10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))
            component["var"] = (1 - np.var(self.y - d_mean) / np.var(self.y)) * 100
            envelop_diag = envelop_comp.K_diag(self.x).numpy()
            component["monotonic"] = get_monotonic(d_mean, envelop_diag)
            component["gradient"] = get_gradient(self.x, d_mean, envelop_diag)

            if self.n_components > 1:
                
                excluded_kernel = Sum([k.kernel for k in self.components if k is not component])
                removed_mean, _ = compute_mean_var(self.x, 
                                                self.x, 
                                                self.y, 
                                                kernel=self.complete_kernel, 
                                                component=excluded_kernel,
                                                noise=self.noise)
                fit_fig, _ = plot_gp(self.x, removed_mean, self.xrange_no_extrap, mean, var)
                print(f"Plot posterior of component {i+1}/{len(self.components)}")

                mean, covar = compute_mean_var(self.x, 
                                           self.x_range,
                                           self.y, 
                                           kernel=self.complete_kernel, 
                                           component=comp, 
                                           noise=self.noise,
                                           full_cov=True)
                extrap_fig, _ = plot_gp(self.x, removed_mean, self.x_range, mean, np.diag(covar))
                print(f"Plot posterior of component {i+1}/{len(self.components)} with extrapolation")

                sample_fig, _ = sample_plot_gp(self.x, self.x_range, mean, covar)
                print(f"Plot sample for component {i+1}/{len(self.components)}")            

                list_figs += [(fit_fig, extrap_fig, sample_fig)]
            
            return components, list_figs
    
        

if __name__ == '__main__':
    # unit test
    from gpflow.kernels import RBF
    from gpflow.models.gpr import GPR
    
    def test_order_by_mae():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        result = Result(x, y, kernel, noise=np.array(0.1))
        result.order_by_mae_reduction()
        
    # test_order_by_mae()