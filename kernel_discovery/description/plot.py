"""
Python implemetation of https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/matlab/component_stats_and_plots.m
"""
import logging
from typing import List
import numpy as np
from numpy.core.defchararray import array
import tensorflow as tf
from anytree import Node
from kernel_discovery.description.transform import ast_to_kernel
from kernel_discovery.description.simplify import simplify, extract_envelop
from kernel_discovery.kernel import Sum, Kernel
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt

logger = logging.getLogger("plot.py")
num_interp_points = 2000
left_extend = 0.
right_extend = 0.1
env_thresh = 0.99


def plot_gp(x: np.array, y: np.array, x_extrap: np.array, mean, var, data_only=False, has_data=True):
    """

    Args:
        x (np.array): input x
        y (np.array): output y
        mean ([type]): GP predictive mean
        var ([type]): GP predictive variance
        data_only (bool, optional): Include data to the plot. Defaults to False.
        has_data (bool, optional): Not include data. Defaults to True.
    """

    color = 'blue'
    alpha = 0.6
    lw = 1.2

    x = x.squeeze()
    y = y.squeeze()
    x_extrap = x_extrap.squeeze()

    plt.figure()
    if not data_only:
        plt.fill_between(x_extrap,
                         mean + 2 * np.sqrt(var),
                         mean - 2 * np.sqrt(var),
                         color=color, alpha=alpha)

    if has_data:
        plt.plot(x, y, 'k.')

    if not data_only:
        plt.plot(x_extrap, mean, color=color, lw=lw)


def sample_plot_gp(x, x_range, mean, covar):

    lw = 1.2
    jitter = 1e-6
    num_samples = 4
    n = x_range.shape[0]
    L = np.linalg.cholesky(covar + jitter * np.eye(n))

    samples = [mean + L @ np.random.randn(n) for _ in range(num_samples)]

    plt.figure()
    for sample in samples:
        plt.plot(x_range, sample, lw=lw)


def gaussian_conditional(Kmn, Lmm, Knn, f, full_cov=False):

    A = tf.linalg.triangular_solve(Lmm, Kmn, lower=True)
    mean = tf.linalg.matmul(A, f, transpose_a=True)
    covar = Knn - tf.linalg.matmul(A, A, transpose_a=True)
    if not full_cov:
        covar = tf.linalg.diag_part(covar)

    return mean.numpy().squeeze(), covar.numpy().squeeze()


def compute_mean_var(x: np.array,
                     x_extrap: np.array,
                     y: np.array,
                     kernel: Kernel,
                     component: Kernel,
                     noise=None,
                     full_cov=False):
    """
    Compute predictive mean and variance where 
    Args:
        x (np.array): data x
        x_extrap (np.array): extrapolation data
        y (np.array): data y
        kernel (Kernel): full kernel function
        component (Kernel): a component in the full kernel function
        noise (np.array): Gaussian noise. Defaults to None.
        full_cov (bool, optional): Whether to return full covariance or not. Defaults to False.

    Returns:
        tuple[np.array, np.array]: predictive mean and variance
    """

    sigma = kernel.K(x)
    if noise is not None:
        sigma = sigma + noise * tf.eye(x.shape[0], dtype=sigma.dtype)

    comp_sigma_star = component.K(x, x_extrap)
    comp_sigma_star2 = component.K(x_extrap, x_extrap)

    L_sigma = tf.linalg.cholesky(sigma)

    mean, var = gaussian_conditional(Kmn=comp_sigma_star,
                                     Lmm=L_sigma,
                                     Knn=comp_sigma_star2,
                                     f=y,
                                     full_cov=full_cov)

    return mean, var


def get_monotonic(data_mean: np.array, envelop_diag: np.array):

    diff = data_mean[1:] - data_mean[:-1]
    activated = np.logical_and(
        envelop_diag[:-1] > env_thresh, envelop_diag[1:] > env_thresh)
    diff_thresh = diff[activated]

    if len(diff_thresh) == 0:
        return 0
    else:
        if np.all(diff_thresh > 0):
            return 1
        elif np.all(diff_thresh < 0):
            return -1
        else:
            return 0


def get_gradient(x, data_mean: np.array, envelop_diag: np.array):

    activated = envelop_diag > env_thresh
    data_mean_thresh = data_mean[activated]
    x_thresh = x[activated]
    if len(data_mean_thresh) == 0:
        return 0.
    else:
        return (data_mean_thresh[-1] - data_mean_thresh[0]) / (x_thresh[-1] - x_thresh[0])


def component_stats(x: np.array, y: np.array, kernel: Node, noise: np.array, order: List):

    x_left = np.min(x) - (np.max(x) - np.min(x)) * left_extend
    x_right = np.max(x) + (np.max(x) - np.min(x)) * right_extend
    x_range = np.linspace(x_left, x_right, num_interp_points)[:, None]
    xrange_no_extrap = np.linspace(
        np.min(x), np.max(x), num_interp_points)[:, None]

    # make sure to simplify kernel which returns sum of product kernels
    kernel = simplify(kernel)
    envelop = extract_envelop(kernel)
    kernel = ast_to_kernel(kernel)
    envelop = ast_to_kernel(envelop)

    if isinstance(kernel, Sum):
        components = kernel.kernels
        envelop_components = envelop.kernels
    else:
        components = [kernel]
        evelop_components = [envelop]

    n_components = len(components)
    assert len(order) == n_components
    components = [components[i] for i in order]

    complete_mean, complete_var = compute_mean_var(
        x, x_range, y, kernel=kernel, component=kernel, noise=noise)

    # plot raw data
    plot_gp(x, y, x_range, complete_mean, complete_var, data_only=True)
    logger.info("Plot raw data")

    # plot full posterior
    plot_gp(x, y, x_range, complete_mean, complete_var, data_only=True)
    logger.info("Plot full posterior")

    # plot sample from full posterior
    complete_mean, complete_covar = compute_mean_var(
        x, x_range, y, kernel=kernel, component=kernel, noise=noise, full_cov=True)
    sample_plot_gp(x, x_range, complete_mean, complete_covar)
    logger.info("Plot sample")

    if len(components) == 1:
        return

    # some statistics includeing SNR, var, monotonic, gradient
    SNRs, vars, monotonic, gradient = [], [], [], []

    for i, (comp, envelop_comp) in enumerate(zip(components, envelop_components)):

        mean, var = compute_mean_var(
            x, xrange_no_extrap, y, kernel, comp, noise)

        # this is for compute some statistic
        d_mean, d_var = compute_mean_var(x, x, y, kernel, comp, noise)

        # computes some statistics
        SNRs += [10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))]
        vars += [(1 - np.var(y - d_mean) / np.var(y)) * 100]
        envelop_diag = envelop_comp.K_diag(x).numpy()
        monotonic += [get_monotonic(d_mean, envelop_diag)]
        gradient += [get_gradient(x, d_mean, envelop_diag)]

        excluded_kernel = Sum([k for k in components if k is not comp])
        removed_mean, _ = compute_mean_var(
            x, x, y, kernel, component=excluded_kernel, noise=noise)
        plot_gp(x, removed_mean, xrange_no_extrap, mean, var)
        logger.info(f"Plot posterior of component {i+1}/{len(components)}")

        mean, covar = compute_mean_var(
            x, x_range, y, kernel, comp, noise, full_cov=True)
        plot_gp(x, removed_mean, x_range, mean, np.diag(covar))
        logger.info(
            f"Plot posterior of component {i+1}/{len(components)} with extrapolation")

        sample_plot_gp(x, x_range, mean, covar)
        logger.info(f"Plot sample for component {i+1}/{len(components)}")

    return {"SNRs": SNRs,
            "vars": vars,
            "monotonic": monotonic,
            "gradient": gradient}


def cummulative_plots(x: np.array, y: np.array, kernel: Node, noise: np.array, order: List):

    x_left = np.min(x) - (np.max(x) - np.min(x)) * left_extend
    x_right = np.max(x) + (np.max(x) - np.min(x)) * right_extend
    x_range = np.linspace(x_left, x_right, num_interp_points)[:, None]
    xrange_no_extrap = np.linspace(
        np.min(x), np.max(x), num_interp_points)[:, None]

    kernel = simplify(kernel)
    kernel = ast_to_kernel(kernel)

    if isinstance(kernel, Sum):
        components = kernel.kernels
    else:
        components = [kernel]

    n_components = len(components)
    assert len(order) == n_components
    components = [components[i] for i in order]

    if len(components) == 1:
        return

    # some statistics: cummulative SNR, cumlutive vars, cumulativate residual vars
    cum_SNRs, cum_vars, cum_res_vars = [], [], []

    residual = y
    accumulate_kernels = []
    for i, comp in enumerate(components):

        accumulate_kernels.append(comp)
        current_kernel = Sum(accumulate_kernels)

        # plot no extrapolation
        mean, var = compute_mean_var(
            x, xrange_no_extrap, y, kernel=kernel, component=current_kernel, noise=noise)
        plot_gp(x, y, xrange_no_extrap, mean, var)
        logger.info(
            f"Plot sum of components up to component {i+1}/{len(components)}")

        # plot with extrapolation
        mean, covar = compute_mean_var(
            x, x_range, y, kernel=kernel, component=current_kernel, noise=noise, full_cov=True)
        plot_gp(x, y, x_range, mean, np.diag(covar))
        logger.info(
            f"Plot sum of components up to component {i+1}/{len(components)} with extrapolation")

        # plot random sample with extrapolation
        sample_plot_gp(x, x_range, mean, covar)
        logger.info(
            f"Plot sample for sum of components up to component {i+1}/{len(components)} with extrapolation")

        d_mean, d_var = compute_mean_var(
            x, x, y, kernel=kernel, component=current_kernel, noise=noise)

        # gather statistics here
        cum_SNRs += [10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))]
        cum_vars += [(1. - np.var(y - d_mean) / np.var(y)) * 100]
        cum_res_vars += [(1. - np.var(y - d_mean) / np.var(residual)) * 100]

        # residual plot
        residual = y - np.reshape(d_mean, y.shape)
        if i < len(components) - 1:
            anti_kernels = [
                k for k in components if k not in accumulate_kernels]
            sum_anti_kernel = Sum(anti_kernels)

            mean, var = compute_mean_var(
                x, xrange_no_extrap, residual, kernel=kernel, component=sum_anti_kernel, noise=noise)
            plot_gp(x, residual, xrange_no_extrap, mean, var)
            logger.info(
                f"Plot residual after component {i+1}/{len(components)}")

    return {"cum_SNRs": cum_SNRs,
            "cum_vars": cum_vars,
            "cum_res_vars": cum_res_vars}


def order_by_mae_reduction(x: np.array, y: np.array, kernel: Node, noise: np.array):

    kernel = simplify(kernel)
    kernel = ast_to_kernel(kernel)

    if isinstance(kernel, Sum):
        components = kernel.kernels
    else:
        components = [kernel]

    # 10 folds cross validates
    n_folds = 10
    kf = KFold(n_splits=n_folds)

    idx = []
    maes = np.zeros((len(components), 1))
    mae_reductions = np.zeros((len(components), 1))
    mav_data = np.mean(np.abs(y))

    # stack
    cumulative_kernels = []

    def validate_errors(kernel, x_train, y_train, x_val, y_val):
        K_train = kernel.K(x_train)
        if noise is not None:
            K_train = K_train + \
                tf.eye(x_train.shape[0], dtype=K_train.dtype) * noise

        K_train_val = kernel.K(x_train, x_val)
        K_val = kernel.K(x_val)

        y_predict, _ = gaussian_conditional(Kmn=K_train_val,
                                            Lmm=tf.linalg.cholesky(K_train),
                                            Knn=K_val,
                                            f=y_train,
                                            full_cov=False)

        error = np.mean(np.abs(y_val - y_predict))
        return error

    previous_mae = mav_data
    for i in range(len(components)):
        best_mae = np.Inf
        for j in range(len(components)):
            if j not in idx:
                # pop first
                cumulative_kernels.append(components[j])
                current_kernel = Sum(cumulative_kernels)
                this_mae = 0.
                for train_idx, val_idx in kf.split(x):
                    x_train, y_train = x[train_idx], y[train_idx]
                    x_val, y_val = x[val_idx], y[val_idx]
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
        cumulative_kernels.append(components[best_j])

    mae_data = {"MAEs": maes,
                "MAE_reductions": mae_reductions,
                "MAV_data": mav_data}
    return idx, mae_data


def checking_stats(x: np.array, y: np.array, kernel: Node, noise: np.array, order: List):

    # TODO: see https://github.com/jamesrobertlloyd/gpss-research/blob/2a64958a018f1668f7b8eedf33c4076a63af7868/source/matlab/checking_stats.m

    kernel = simplify(kernel)
    kernel = ast_to_kernel(kernel)

    if isinstance(kernel, Sum):
        components = kernel.kernels
    else:
        components = [kernel]

    n_components = len(components)
    assert len(order) == n_components
    components = [components[i] for i in order]

    complete_sigma = kernel.K(x)
    if noise is not None:
        complete_sigma = complete_sigma + \
            tf.eye(x.shape[0], dtype=complete_sigma.dtype) * noise

    L_sigma = tf.linalg.cholesky(complete_sigma)

    for comp in components:

        decomp_sigma = comp.K(x)
        data_mean, data_covar = gaussian_conditional(Kmn=decomp_sigma,
                                                     Lmm=L_sigma,
                                                     Knn=decomp_sigma,
                                                     f=y,
                                                     full_cov=True)


class Component():
    
    def __init__(self, kernel) -> None:
        
        self.kernel = kernel
        self.envelop = None
        
        self.snr = None
        self.var = None
        self.monotonic = None
        self.gradient = None
        
        self.cum_snr = None
        self.cum_var = None
        self.cum_res_var = None
        
        self.mae = None
        self.mae_reduction = None
        
    

class Result():
    
    def __init__(self, x: np.array, y: np.array, kernel: Node, noise:np.array, root) -> None:
        self.root = root
        self.x = x
        self.y = y
        
        kernel = simplify(kernel)
        envelop = extract_envelop(kernel)
        
        kernel = ast_to_kernel(kernel)
        envelop = ast_to_kernel(envelop)
        
        self.complete_kernel = kernel
        
        self.kernels = kernel.kernels if isinstance(kernel, Sum) else [kernel]
        self.envelop_kernels = envelop.kernels if isinstance(envelop, Sum) else [envelop]
        self.noise = noise
        
        self.n_components = len(self.kernels)
        
        self.components = []
        
        self.logger = logging.getLogger(__class__.__name__)
        
        # TODO: generate unique id to save to root directory
        
        # some general stats for sum kernel
        self.mav_data = np.mean(np.abs(y))
        
        # misc i.e. plot
        x_left = np.min(x) - (np.max(x) - np.min(x)) * left_extend
        x_right = np.max(x) + (np.max(x) - np.min(x)) * right_extend
        self.x_range = np.linspace(x_left, x_right, num_interp_points)[:, None]
        self.xrange_no_extrap = np.linspace(np.min(x), np.max(x), num_interp_points)[:, None]

    
    def process(self):
        
        self.order_by_mae_reduction()
        self.individual_component_stats_and_plots()
        
    
    def order_by_mae_reduction(self):
        """Decide order of each component by MAE"""
        
        self.logger.info("Decide order of each component by MAE")
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
                                                Lmm=tf.linalg.cholesky(K_train),
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
            component = Component(self.kernels[i])
            component.envelop = self.envelop_kernels[i]
            component.mae = maes[i]
            component.mae_reduction = mae_reductions[i]
            # add to the list
            self.components += [component]
        
        assert len(self.components) == self.n_components
        
    def individual_component_stats_and_plots(self):
        
        self.logger.info("Compute individual component statistics and generate individual plots")
        
        complete_mean, complete_var = compute_mean_var(self.x, 
                                                       self.x_range, 
                                                       self.y, 
                                                       kernel=self.complete_kernel, 
                                                       component=self.complete_kernel, 
                                                       noise=self.noise)

        # plot raw data
        plot_gp(self.x, self.y, self.x_range, complete_mean, complete_var, data_only=True)
        self.logger.info("Plot raw data")

        # plot full posterior
        plot_gp(self.x, self.y, self.x_range, complete_mean, complete_var, data_only=True)
        self.logger.info("Plot full posterior")

        # plot sample from full posterior
        complete_mean, complete_covar = compute_mean_var(self.x, 
                                                         self.x_range, 
                                                         self.y, 
                                                         kernel=self.complete_kernel, 
                                                         component=self.complete_kernel,
                                                         noise=self.noise, 
                                                         full_cov=True)
        sample_plot_gp(self.x, self.x_range, complete_mean, complete_covar)
        self.logger.info("Plot sample")

        if len(self.components) == 1:
            return

        # some statistics includeing SNR, var, monotonic, gradient
        SNRs, vars, monotonic, gradient = [], [], [], []

        for i, component  in enumerate(self.components):
            comp = component.kernel
            envelop_comp = component.envelop
            mean, var = compute_mean_var(self.x,
                                         self.xrange_no_extrap, 
                                         self.y, 
                                         kernel=self.complete_kernel,
                                         component=comp, 
                                         noise=self.noise)

            # this is for compute some statistic
            d_mean, d_var = compute_mean_var(self.x, self.x, self.y, self.complete_kernel, comp, self.noise)

            # computes some statistics
            SNRs += [10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))]
            vars += [(1 - np.var(self.y - d_mean) / np.var(self.y)) * 100]
            envelop_diag = envelop_comp.K_diag(self.x).numpy()
            monotonic += [get_monotonic(d_mean, envelop_diag)]
            gradient += [get_gradient(self.x, d_mean, envelop_diag)]

            excluded_kernel = Sum([k.kernel for k in self.components if k is not component])
            removed_mean, _ = compute_mean_var(self.x, 
                                               self.x, 
                                               self.y, 
                                               kernel=self.complete_kernel, 
                                               component=excluded_kernel,
                                               noise=self.noise)
            plot_gp(self.x, removed_mean, self.xrange_no_extrap, mean, var)
            logger.info(f"Plot posterior of component {i+1}/{len(self.components)}")

            mean, covar = compute_mean_var(self.x, 
                                           self.x_range,
                                           self.y, 
                                           kernel=self.complete_kernel, 
                                           component=comp, 
                                           noise=self.noise,
                                           full_cov=True)
            plot_gp(self.x, removed_mean, self.x_range, mean, np.diag(covar))
            logger.info(
                f"Plot posterior of component {i+1}/{len(self.components)} with extrapolation")

            sample_plot_gp(self.x, self.x_range, mean, covar)
            logger.info(f"Plot sample for component {i+1}/{len(self.components)}")

        # return {"SNRs": SNRs,
        #         "vars": vars,
        #         "monotonic": monotonic,
        #         "gradient": gradient}
    
if __name__ == '__main__':

    from kernel_discovery.kernel import RBF
    from kernel_discovery.description.transform import kernel_to_ast

    def test_plot_gp():

        x = np.linspace(0, 5, 100)
        y = np.sin(x) + 0.01 * np.random.randn(100)
        x_extrap = np.linspace(0, 5.5, 200)
        mean = np.sin(x_extrap)
        var = np.ones(200)*0.1

        plot_gp(x, y, x_extrap, mean, var)

        plt.savefig("dummy.png")

    def test_sample_plot_gp():

        x = np.linspace(0, 5, 100)
        y = np.sin(x) + 0.01 * np.random.randn(100)
        x_extrap = np.linspace(0, 5.5, 200)
        mean = np.sin(x_extrap)
        var = np.eye(200)*0.1

        sample_plot_gp(x, x_extrap, mean, var)

        plt.savefig("dummy.png")

    def test_compute_mean_var():

        x = np.linspace(0, 5, 100)[:, None]
        x_extrap = np.linspace(0, 5.5, 300)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        component = RBF()

        mean, var = compute_mean_var(
            x, x_extrap, y, kernel, component, noise=0.1)

        plot_gp(x.squeeze(), y.squeeze(), x_extrap.squeeze(),
                mean.squeeze(), var.squeeze())

        plt.savefig('dummy_2.png')

    def test_component_and_stat():

        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)

        result = component_stats(
            x, y, kernel=kernel, noise=np.array(0.1), order=[1, 0])

        print(result)
        plt.savefig('dummy_3.png')

    def test_cummulative_plot():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)

        result = cummulative_plots(
            x, y, kernel=kernel, noise=np.array(0.1), order=[1, 0])
        print(result)

    def test_order_mae():

        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)

        idx, mae_data = order_by_mae_reduction(
            x, y, kernel=kernel, noise=np.array(0.1))
        print(idx)
        print(mae_data)
        
    def test_result_object():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)
        result = Result(x, y, kernel, noise=np.array(0.1), root="./figure")
        result.process()

    # test_plot_gp()
    # test_sample_plot_gp()
    # test_compute_mean_var()
    # test_component_and_stat()
    # test_cummulative_plot()
    # test_order_mae()
    test_result_object()
