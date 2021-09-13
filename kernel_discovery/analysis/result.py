import numpy as np
import tensorflow as tf
from anytree import Node
from gpflow.kernels import Sum, Kernel
from sklearn.model_selection import KFold
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram

from kernel_discovery.description import kernel_to_ast, simplify, ast_to_kernel, extract_envelop
from kernel_discovery.analysis.util import (gaussian_conditional,
                                            compute_mean_var, 
                                            get_gradient, 
                                            get_monotonic,
                                            compute_cholesky)
from kernel_discovery.analysis.mmd import mmd_test, mmd_plot
from kernel_discovery.description.transform import ast_to_kernel
from kernel_discovery.description.plot import plot_gp, sample_plot_gp, two_band_plot

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
        
        self.kernels = self.complete_kernel.kernels if isinstance(self.complete_kernel, Sum) else [self.complete_kernelc]
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
            component["kernel"] = kernel_to_ast(self.kernels[i], include_param=True)
            component["envelop"] = kernel_to_ast(self.envelop_kernels[i], include_param=True)
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
            component["var"] = (1 - np.var(self.y - d_mean.reshape(self.y.shape)) / np.var(self.y)) * 100
            envelop_diag = envelop_comp.K_diag(self.x).numpy()
            component["monotonic"] = get_monotonic(d_mean, envelop_diag)
            component["gradient"] = get_gradient(self.x, d_mean, envelop_diag)

            if self.n_components == 1:
                removed_mean = self.y
            else:
                excluded_kernel = Sum([ast_to_kernel(k["kernel"]) for k in self.components if k is not component])
                removed_mean, _ = compute_mean_var(self.x, 
                                                self.x, 
                                                self.y, 
                                                kernel=self.complete_kernel, 
                                                component=excluded_kernel,
                                                noise=self.noise)
                removed_mean = self.y - removed_mean.reshape(self.y.shape)
                
            fit_fig, _ = plot_gp(self.x, removed_mean, self.xrange_no_extrap, mean, var, has_data=False)
            print(f"Plot posterior of component {i+1}/{len(self.components)}")

            mean, covar = compute_mean_var(self.x, 
                                        self.x_range,
                                        self.y, 
                                        kernel=self.complete_kernel, 
                                        component=comp, 
                                        noise=self.noise,
                                        full_cov=True)
            extrap_fig, _ = plot_gp(self.x, removed_mean, self.x_range, mean, np.diag(covar), has_data=False)
            print(f"Plot posterior of component {i+1}/{len(self.components)} with extrapolation")

            sample_fig, _ = sample_plot_gp(self.x, self.x_range, mean, covar)
            print(f"Plot sample for component {i+1}/{len(self.components)}")            

            list_figs += [(fit_fig, extrap_fig, sample_fig)]
            
        return list_figs
class CummulativeAnalysis(DownstreamAnalysis):
    
    def __init__(self, x, y, ast, noise) -> None:
        super().__init__(x, y, ast, noise)
    
    def analyze(self):
        print("Compute cummulative statistics and cummulative plots")
        
        residual = self.y
        accumulate_kernels = []
        figs = []
        for i, component in enumerate(self.components):
            current_kernel = ast_to_kernel(component["kernel"])
            accumulate_kernels.append(current_kernel)
            cumm_kernel = Sum(accumulate_kernels)

            # plot no extrapolation
            mean, var = compute_mean_var(self.x, 
                                         self.xrange_no_extrap, 
                                         self.y, 
                                         kernel=self.complete_kernel, 
                                         component=cumm_kernel, 
                                         noise=self.noise)
            cumm_fit_fig, _ = plot_gp(self.x, self.y, self.xrange_no_extrap, mean, var)
            print(f"Plot sum of components up to component {i+1}/{self.n_components}")

            # plot with extrapolation
            mean, covar = compute_mean_var(self.x, 
                                           self.x_range, 
                                           self.y, 
                                           kernel=self.complete_kernel, 
                                           component=cumm_kernel, 
                                           noise=self.noise, 
                                           full_cov=True)
            cumm_extrap_fig, _ = plot_gp(self.x, self.y, self.x_range, mean, np.diag(covar))
            print(f"Plot sum of components up to component {i+1}/{self.n_components} with extrapolation.")
            

            # plot random sample with extrapolation
            cumm_sample_fig, _ = sample_plot_gp(self.x, self.x_range, mean, covar)
            print(f"Plot sample for sum of components up to component {i+1}/{self.n_components} with extrapolation.")

            d_mean, d_var = compute_mean_var(self.x, 
                                             self.x, 
                                             self.y, 
                                             kernel=self.complete_kernel, 
                                             component=cumm_kernel, 
                                             noise=self.noise)

            # gather statistics here
            component["cum_snr"] = 10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))
            component["cum_var"] = (1. - np.var(self.y - d_mean.reshape(self.y.shape)) / np.var(self.y)) * 100
            component["cum_res_var"] = (1. - np.var(self.y - d_mean.reshape(self.y.shape)) / np.var(residual)) * 100

            # residual plot
            residual = self.y - np.reshape(d_mean, self.y.shape)
            
            anti_res_fig = None
            if i < self.n_components - 1:
                anti_kernels = [
                    comp["kernel"] for comp in self.components 
                    if comp["kernel"] not in accumulate_kernels]
                anti_kernels = [ast_to_kernel(ast) for ast in anti_kernels]
                sum_anti_kernel = Sum(anti_kernels)

                mean, var = compute_mean_var(self.x, 
                                             self.xrange_no_extrap, 
                                             residual, 
                                             kernel=self.complete_kernel, 
                                             component=sum_anti_kernel, 
                                             noise=self.noise)
                anti_res_fig, _ = plot_gp(self.x, residual, self.xrange_no_extrap, mean, var, has_data=False)
                print(f"Plot residual after component {i+1}/{self.n_components}.")

            figs += [(cumm_fit_fig, cumm_extrap_fig, cumm_sample_fig, anti_res_fig)]
        
        return figs

class ModelCheckingAnalysis(DownstreamAnalysis):
    
    def __init__(self, x, y, ast, noise) -> None:
        super().__init__(x, y, ast, noise)
        
    def analyze(self, n_samples=1000):
        """
        see https://github.com/jamesrobertlloyd/gpss-research/blob/2a64958a018f1668f7b8eedf33c4076a63af7868/source/matlab/checking_stats.m
        """
        print("Perform model check")

        complete_sigma = self.complete_kernel.K(self.x)
        if self.noise is not None:
            complete_sigma = complete_sigma + tf.eye(self.x.shape[0], dtype=complete_sigma.dtype) * self.noise


        figs = []
        for i, component in enumerate(self.components):
            current_kernel = ast_to_kernel(component["kernel"])
            decomp_sigma = current_kernel.K(self.x)
            data_mean, data_covar = gaussian_conditional(Kmn=decomp_sigma,
                                                        Kmm=complete_sigma,
                                                        Knn=decomp_sigma,
                                                        f=self.y,
                                                        full_cov=True)
            
            
            random_indices = np.random.permutation(self.x.shape[0])
            x_post = self.x[random_indices]
            y_data_post = self.y[random_indices]
            
            decomp_sigma_post = current_kernel.K(x_post)
            
            data_mean_post, _ = gaussian_conditional(Kmn=decomp_sigma_post,
                                                  Kmm=complete_sigma,
                                                  Knn=decomp_sigma_post,
                                                  f=self.y)
            
            
            L_sigma_post = compute_cholesky(decomp_sigma_post)
            y_post = (y_data_post - data_mean_post[:,None]) + L_sigma_post.transpose() @ np.random.randn(L_sigma_post.shape[0],1)
            
            
            # 1. MMD test and plot
            print("Run MMD test")
            random_indices = np.random.permutation(self.x.shape[0])
            x_data, y_data = self.x[random_indices], self.y[random_indices]
            A = np.hstack([x_data, y_data])
            B = np.hstack([x_post, y_post])
            
            # standarized A, B
            A_std = np.std(A, axis=0) 
            A_std = np.tile(A_std, (A.shape[0], 1))
            A = A / A_std
            B = B / A_std
            
            mmd_fig, ax, mmd_value, mmd_p_value = mmd_test(A, B, n_shuffle=n_samples)
            component["mmd_p_value"] = mmd_p_value
            ax.set_title(f"MMD two sample test plot for component {i}")
                    
            
            
            # 2. make qq plot
            print("Make QQ plot")
            prior_L = compute_cholesky(decomp_sigma)
            post_L = compute_cholesky(data_covar)
            qq_fig, ax, qq_d_max, qq_d_min = make_qqplot(data_mean, prior_L=prior_L, post_L=post_L)
            ax.set_title(f"QQ uncertainty plot for component {i}")
            component["qq_d_max"] = qq_d_max
            component["qq_d_min"] = qq_d_min
            
            # make a grid
            x_data = np.sort(self.x)
            x_data_delta = x_data[1:] - x_data[:-1]
            min_delta = np.min(x_data_delta)
            multiples = x_data_delta / min_delta
            rounded_multiples = np.round(multiples * 10)/10
            if np.all(rounded_multiples == 1):
                x_grid = self.x
                grid_distance = x_grid[1] - x_grid[0]
                num_el = len(self.y)
            else:
                if np.all(rounded_multiples == np.round(rounded_multiples)):
                    num_el = int(np.sum(rounded_multiples) + 1)
                else:
                    num_el = len(self.x)
                x_grid = np.linspace(np.min(self.x), np.max(self.x), num_el)[:, None]
                grid_distance = x_grid[1] - x_grid[0]
                
            decomp_sigma_grid_x = current_kernel.K(self.x, x_grid)
            decomp_sigma_grid = current_kernel.K(x_grid)
            data_mean_grid, data_covar_grid = gaussian_conditional(Kmn=decomp_sigma_grid_x,
                                                                   Kmm=complete_sigma,
                                                                   Knn=decomp_sigma_grid,
                                                                   f=self.y,
                                                                   full_cov=True)
            
            # 3. make acf plot
            print("Make ACF plot")
            prior_L = compute_cholesky(decomp_sigma_grid)
            post_L = compute_cholesky(data_covar_grid)
            acf_fig, ax, acf_min_loc, acf_min = make_acf_plot(data_mean_grid,
                                                                              prior_L=prior_L,
                                                                              post_L=post_L,
                                                                              grid_distance=grid_distance,
                                                                              samples=n_samples)
            ax.set_title(f"ACF uncertainty plot for component {i}")
            ax.set_ylabel("Correlation coefficient")
            ax.set_xlabel("Lag")
            component["acf_min_loc"] = acf_min_loc
            component["acf_min"] = acf_min
            
            # 4. make periodogram plot 
            print("Make periodogram plot")
            pxx_fig, ax, pxx_max_loc, pxx_max = make_peridogram(data_mean_grid,
                                                                prior_L=prior_L,
                                                                post_L=post_L,
                                                                samples=n_samples)
            ax.set_title(f"Periodogram uncertainty plot for component {i}")
            ax.set_ylabel("Power / frequency")
            ax.set_xlabel("Normalized frequency")
            component["pxx_max_loc"] = pxx_max_loc
            component["pxx_max"] = pxx_max
            
            figs += [(mmd_fig, qq_fig, acf_fig, pxx_fig)]
        
        return figs

def compute_quantile(samples: np.array, ):
    
    n, n_samples = samples.shape
    
    quantile_values = np.linspace(0, 1, n+2)[1:-1]
    assert len(quantile_values) == n
    quantiled = np.quantile(samples, quantile_values)
    
    qq = np.zeros((n, n_samples))
    qq_d_max = np.zeros((n_samples, ))
    qq_d_min = np.zeros((n_samples, ))
    
    for i in range(n_samples):
        a = np.sort(samples[:, i])
        qq[:, i] = a
        difference = a - quantiled
        qq_d_max[i] = np.max(difference)
        qq_d_min[i] = np.min(difference)
        
    mean = np.mean(qq, axis=1)
    high = np.quantile(qq, 0.95, axis=1)
    low = np.quantile(qq, 0.05, axis=1)
    
    return quantiled, qq_d_max, qq_d_min, mean, high, low

def make_qqplot(data_mean, prior_L, post_L, samples=1000):

    if len(data_mean.shape) == 1:
        data_mean = data_mean[:, None]

    n = data_mean.shape[0]
    prior_samples = np.matmul(prior_L.transpose(), np.random.randn(n, samples))
    prior_quantiled, prior_qq_d_max, prior_qq_d_min, prior_mean, prior_high, prior_low = compute_quantile(prior_samples)
    
    post_samples = data_mean + np.matmul(post_L.transpose(), np.random.randn(n, samples)) # sample from 
    post_quantiled, post_qq_d_max, post_qq_d_min, post_mean, post_high, post_low = compute_quantile(post_samples)
    
    qq_d_max = np.mean(prior_qq_d_max > post_qq_d_max + 1e-5 * np.max(post_qq_d_max) * np.random.randn(*post_qq_d_max.shape))
    qq_d_min = np.mean(prior_qq_d_min < post_qq_d_min + 1e-5 * np.max(post_qq_d_min) * np.random.randn(*post_qq_d_min.shape))
    
    fig, ax = two_band_plot(prior_quantiled, prior_mean, prior_high, prior_low, post_mean, post_high, post_low)
    ax.scatter(prior_quantiled, post_quantiled, color='k')
    
    return fig, ax, qq_d_max, qq_d_min


def compute_acf(samples: np.array):
    n, n_samples = samples.shape
    acf_values = acf(samples[:, 0], n - 1)
    acf_values = np.zeros((acf_values.size, n_samples))
    acf_min_loc = np.zeros((n_samples,))
    acf_min = np.zeros((n_samples,))
    
    for i in range(n_samples):
        acf_values[:, i] = acf(samples[:,i], n-1)
        acf_min_loc[i] = np.argmin(acf_values[:, i])
        acf_min[i] = np.min(acf_values[:,i])
    
    return acf_values, acf_min_loc, acf_min

def make_acf_plot(data_mean, prior_L, post_L, grid_distance, samples=1000):
    
    n = data_mean.shape[0]
    prior_samples = np.matmul(prior_L.transpose(), np.random.randn(n, samples))
    prior_acf, prior_acf_min_loc, prior_acf_min = compute_acf(prior_samples)
    post_samples = np.matmul(post_L.transpose(), np.random.randn(n, samples))
    post_acf, post_acf_min_loc, post_acf_min = compute_acf(post_samples)

    acf_min = np.mean(prior_acf_min < post_acf_min + 1e-5 * np.max(post_acf_min) * np.random.randn(*post_acf_min.shape))
    acf_min_loc = np.mean(prior_acf_min_loc < post_acf_min_loc + 1e-5 * np.max(post_acf_min_loc) * np.random.randn(*post_acf_min_loc.shape))

    fig, ax = two_band_plot(x=np.arange(1, post_acf.shape[0]+1) * grid_distance,
                  mean_1=np.mean(prior_acf, axis=1),
                  high_1=np.quantile(prior_acf, 0.95, axis=1),
                  low_1=np.quantile(prior_acf, 0.05, axis=1),
                  mean_2=np.mean(post_acf, axis=1),
                  high_2=np.quantile(post_acf, 0.95, axis=1),
                  low_2=np.quantile(post_acf, 0.05, axis=1))
    
    return fig, ax, acf_min_loc, acf_min

def compute_periodogram(samples: np.array):
    
    _, n_samples = samples.shape
    
    pxx = 10. * np.log10(periodogram(samples[:, 0])[0])
    pxx = np.zeros((pxx.size, n_samples))
    pxx_max_loc = np.zeros((n_samples, ))
    pxx_max = np.zeros((n_samples,))
    for i in range(n_samples):
        pxx[:, i] = 10. * np.log10(periodogram(samples[:, i])[0]) 
        pxx_max_loc[i] = np.argmax(pxx[:, i])
        pxx_max[i] = np.max(pxx[:, i])
    
    return pxx, pxx_max_loc, pxx_max
    
def make_peridogram(data_mean, prior_L, post_L, samples=1000):
    
    n = data_mean.shape[0]
    prior_samples = np.matmul(prior_L.transpose(), np.random.randn(n, samples))
    prior_pxx, prior_pxx_max_loc, prior_pxx_max = compute_periodogram(prior_samples)
    post_samples = np.matmul(post_L.transpose(), np.random.randn(n, samples))
    post_pxx, post_pxx_max_loc, post_pxx_max = compute_periodogram(post_samples)

    pxx_max = np.mean(prior_pxx_max > post_pxx_max + 1e-5 * np.max(post_pxx_max) * np.random.randn(*post_pxx_max.shape))
    pxx_max_loc = np.mean(prior_pxx_max_loc > post_pxx_max_loc + 1e-5 * np.max(post_pxx_max_loc) * np.random.randn(*post_pxx_max_loc.shape))
    
    fig, ax = two_band_plot(x=np.linspace(0,1, prior_pxx.shape[0]),
                  mean_1=np.mean(prior_pxx, axis=1),
                  high_1=np.quantile(prior_pxx, 0.95, axis=1),
                  low_1=np.quantile(prior_pxx, 0.05, axis=1),
                  mean_2=np.mean(post_pxx, axis=1),
                  high_2=np.quantile(post_pxx, 0.95, axis=1),
                  low_2=np.quantile(post_pxx, 0.05, axis=1))
    
    return fig, ax, pxx_max_loc, pxx_max        

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
        
    def test_sample_plot_gp():
    
        x = np.linspace(0, 5, 100)
        y = np.sin(x) + 0.01 * np.random.randn(100)
        x_extrap = np.linspace(0, 5.5, 200)
        mean = np.sin(x_extrap)
        var = np.eye(200)*0.1

        fig, _ = sample_plot_gp(x, x_extrap, mean, var)

        fig.savefig("dummy.png")
        
        
    # test_order_by_mae()
    test_sample_plot_gp()