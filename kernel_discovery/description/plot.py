"""
Python implemetation of https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/matlab/component_stats_and_plots.m
"""

import os

from numpy import random
from numpy.core.fromnumeric import var
from numpy.lib.function_base import quantile
from kernel_discovery.description.describe import ProductDesc
import logging
from typing import Callable, List
import numpy as np
from numpy.core.defchararray import array
import tensorflow as tf
from anytree import Node
from kernel_discovery.description.transform import ast_to_kernel, ast_to_text, kernel_to_ast
from kernel_discovery.description.simplify import simplify, extract_envelop
from kernel_discovery.kernel import Sum, Kernel
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist

from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram

import uuid


import matplotlib.pyplot as plt

logger = logging.getLogger("plot.py")
num_interp_points = 2000
left_extend = 0.
right_extend = 0.1
env_thresh = 0.99

GP_FIG_SIZE = (6,4)
FIGURE_EXT = "png" # figure extension
SAVEFIG_KWARGS = dict(dpi=250)


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

    fig, ax = plt.subplots(figsize=GP_FIG_SIZE)
    if not data_only:
        ax.fill_between(x_extrap,
                         mean + 2 * np.sqrt(var),
                         mean - 2 * np.sqrt(var),
                         color=color, alpha=alpha)

    if has_data:
        ax.plot(x, y, 'k.')

    if not data_only:
        ax.plot(x_extrap, mean, color=color, lw=lw)
        
    return fig, ax


def sample_plot_gp(x, x_range, mean, covar):

    lw = 1.2
    jitter = 1e-6
    num_samples = 3
    n = x_range.shape[0]
    L = np.linalg.cholesky(covar + jitter * np.eye(n))

    samples = [mean + L @ np.random.randn(n) for _ in range(num_samples)]

    fig, ax = plt.subplots(figsize=GP_FIG_SIZE)
    for sample in samples:
        ax.plot(x_range, sample, lw=lw)
    
    return fig, ax


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
    
class Component():
    
    _FIT = "fit_{i}.{ext}" # string format with order and extension
    _EXTRAP = "extrap_{i}.{ext}"
    _SAMPLE = "sample_{i}.{ext}"
    _CUM_FIT = "cum_fit_{i}.{ext}"
    _CUM_EXTRAP = "cum_extrap_{i}.{ext}"
    _CUM_SAMPLE = "cum_sample_{i}.{ext}"
    _ANTI_RES = "anti_res_{i}.{ext}"
    
    def __init__(self, kernel) -> None:
        
        self.kernel = kernel
        self.x = None
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
        
        
    def listing_figures(self, i, ext):
        """Generate all figure names"""
        self.i = i
        self.fit = self._FIT.format(i=i, ext=ext)
        self.extrap = self._EXTRAP.format(i=i, ext=ext)
        self.sample =  self._SAMPLE.format(i=i, ext=ext)
        self.cum_fit = self._CUM_FIT.format(i=i, ext=ext)
        self.cum_extrap = self._CUM_EXTRAP.format(i=i, ext=ext)
        self.cum_sample = self._CUM_SAMPLE.format(i=i, ext=ext)
        self.anti_res = self._ANTI_RES.format(i=i, ext=ext)
    
        
    def make_description(self):
        prod = kernel_to_ast(self.kernel)
        assert self.monotonic is not None
        descriptor = ProductDesc(prod, self.x, self.monotonic, self.gradient)
        summary, full_desc, extrap_desc = descriptor.translate()
        self.summary = summary
        self.full_desc = ".\n".join(full_desc)
        self.extrap_desc = ".\n".join(extrap_desc)
        
    def __repr__(self) -> str:
        kernel_str = ast_to_text(kernel_to_ast(self.kernel))
        
        return f"kernel = [{kernel_str}] \n \t SNR: {self.snr:.2f}, var: {self.var:.2f}, monotonic: {self.monotonic}, \n\t cum SNR: {self.cum_snr:.2f}, " + \
            f"cum var: {self.cum_var:.2f}, cum residual var: {self.cum_res_var:.2f}, \n\t MAE: {self.mae:.2f}, MAE reduction: {self.mae_reduction:.2f}"
        

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
        
        # create unique directory under the working directory
        # TODO: it'd be better if this info is passed to this object
        self.uuid = str(uuid.uuid1())
        self.save_dir = os.path.join(self.root, self.uuid)
        os.makedirs(self.save_dir)
        self.logger.info(f"Create a directory [{self.save_dir}]")
        
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
        self.cummulative_stats_and_plots()
        self.checking_stats()
        
    
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
            component.x = self.x
            component.envelop = self.envelop_kernels[i]
            component.mae = maes[i].squeeze()
            component.mae_reduction = mae_reductions[i].squeeze()
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
        fig, _ = plot_gp(self.x, self.y, self.x_range, complete_mean, complete_var, data_only=True)
        file_name = os.path.join(self.save_dir, "raw.png")
        fig.savefig(file_name)
        self.logger.info(f"Plot raw data and save at [{file_name}]")

        # plot full posterior
        fig, _= plot_gp(self.x, self.y, self.x_range, complete_mean, complete_var)
        file_name = os.path.join(self.save_dir, "fit.png")
        fig.savefig(file_name)
        self.logger.info(f"Plot full posterior and save at [{file_name}]")

        # plot sample from full posterior
        complete_mean, complete_covar = compute_mean_var(self.x, 
                                                         self.x_range, 
                                                         self.y, 
                                                         kernel=self.complete_kernel, 
                                                         component=self.complete_kernel,
                                                         noise=self.noise, 
                                                         full_cov=True)
        fig, _ = sample_plot_gp(self.x, self.x_range, complete_mean, complete_covar)
        file_name = os.path.join(self.save_dir, "sample.png")
        fig.savefig(file_name)
        self.logger.info(f"Plot sample and save at [{file_name}]")

        for i, component  in enumerate(self.components):
            
            # allocate figure name
            component.listing_figures(i, ext=FIGURE_EXT)
            
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
            component.snr = 10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))
            component.var = (1 - np.var(self.y - d_mean) / np.var(self.y)) * 100
            envelop_diag = envelop_comp.K_diag(self.x).numpy()
            component.monotonic = get_monotonic(d_mean, envelop_diag)
            component.gradient = get_gradient(self.x, d_mean, envelop_diag)

            if self.n_components > 1:
                excluded_kernel = Sum([k.kernel for k in self.components if k is not component])
                removed_mean, _ = compute_mean_var(self.x, 
                                                self.x, 
                                                self.y, 
                                                kernel=self.complete_kernel, 
                                                component=excluded_kernel,
                                                noise=self.noise)
                fig, _ = plot_gp(self.x, removed_mean, self.xrange_no_extrap, mean, var)
                file_name = os.path.join(self.save_dir, component.fit)
                fig.savefig(file_name)
                self.logger.info(f"Plot posterior of component {i+1}/{len(self.components)}. Figure was saved at [{file_name}]")

                mean, covar = compute_mean_var(self.x, 
                                           self.x_range,
                                           self.y, 
                                           kernel=self.complete_kernel, 
                                           component=comp, 
                                           noise=self.noise,
                                           full_cov=True)
                fig, _ = plot_gp(self.x, removed_mean, self.x_range, mean, np.diag(covar))
                file_name = os.path.join(self.save_dir, component.extrap)
                fig.savefig(file_name)
                self.logger.info(
                    f"Plot posterior of component {i+1}/{len(self.components)} with extrapolation. Figure was saved at [{file_name}]")

                fig, _ = sample_plot_gp(self.x, self.x_range, mean, covar)
                file_name = os.path.join(self.save_dir, component.sample)
                fig.savefig(file_name)
                self.logger.info(f"Plot sample for component {i+1}/{len(self.components)}. Figure was saved at [{file_name}]")            

    def cummulative_stats_and_plots(self):
        
        
        
        self.logger.info("Compute cummulative statistics and cummulative plots")
        
        residual = self.y
        accumulate_kernels = []
        for i, component in enumerate(self.components):

            accumulate_kernels.append(component.kernel)
            current_kernel = Sum(accumulate_kernels)

            # plot no extrapolation
            mean, var = compute_mean_var(self.x, 
                                         self.xrange_no_extrap, 
                                         self.y, 
                                         kernel=self.complete_kernel, 
                                         component=current_kernel, 
                                         noise=self.noise)
            fig, _ = plot_gp(self.x, self.y, self.xrange_no_extrap, mean, var)
            file_name = os.path.join(self.save_dir, component.cum_fit)
            fig.savefig(file_name)
            self.logger.info(f"Plot sum of components up to component {i+1}/{self.n_components}. Figure was saved at [{file_name}]")

            # plot with extrapolation
            mean, covar = compute_mean_var(self.x, 
                                           self.x_range, 
                                           self.y, 
                                           kernel=self.complete_kernel, 
                                           component=current_kernel, 
                                           noise=self.noise, 
                                           full_cov=True)
            fig, _ = plot_gp(self.x, self.y, self.x_range, mean, np.diag(covar))
            file_name = os.path.join(self.save_dir, component.cum_extrap)
            fig.savefig(file_name)
            self.logger.info(f"Plot sum of components up to component {i+1}/{self.n_components} with extrapolation. Figure was saved at [{file_name}]")
            

            # plot random sample with extrapolation
            fig, _ = sample_plot_gp(self.x, self.x_range, mean, covar)
            file_name = os.path.join(self.save_dir, component.cum_sample)
            fig.savefig(file_name)
            self.logger.info(f"Plot sample for sum of components up to component {i+1}/{self.n_components} with extrapolation. Figure was save at [{file_name}]")

            d_mean, d_var = compute_mean_var(self.x, 
                                             self.x, 
                                             self.y, 
                                             kernel=self.complete_kernel, 
                                             component=current_kernel, 
                                             noise=self.noise)

            # gather statistics here
            component.cum_snr = 10 * np.log10(np.sum(d_mean**2) / np.sum(d_var))
            component.cum_var = (1. - np.var(self.y - d_mean) / np.var(self.y)) * 100
            component.cum_res_var = (1. - np.var(self.y - d_mean) / np.var(residual)) * 100

            # residual plot
            residual = self.y - np.reshape(d_mean, self.y.shape)
            if i < self.n_components - 1:
                anti_kernels = [
                    comp.kernel for comp in self.components if comp.kernel not in accumulate_kernels]
                sum_anti_kernel = Sum(anti_kernels)

                mean, var = compute_mean_var(self.x, 
                                             self.xrange_no_extrap, 
                                             residual, 
                                             kernel=self.complete_kernel, 
                                             component=sum_anti_kernel, 
                                             noise=self.noise)
                fig, _ = plot_gp(self.x, residual, self.xrange_no_extrap, mean, var)
                file_name = os.path.join(self.save_dir, component.anti_res)
                fig.savefig(file_name)
                self.logger.info(f"Plot residual after component {i+1}/{self.n_components}. Figure was saved at [{file_name}]")

    def checking_stats(self):
        
        # TODO: see https://github.com/jamesrobertlloyd/gpss-research/blob/2a64958a018f1668f7b8eedf33c4076a63af7868/source/matlab/checking_stats.m
        self.logger.info("Perform model check")

        complete_sigma = self.complete_kernel.K(self.x)
        if self.noise is not None:
            complete_sigma = complete_sigma + tf.eye(self.x.shape[0], dtype=complete_sigma.dtype) * self.noise

        L_sigma = tf.linalg.cholesky(complete_sigma)

        for component in self.components:

            decomp_sigma = component.kernel.K(self.x)
            data_mean, data_covar = gaussian_conditional(Kmn=decomp_sigma,
                                                        Lmm=L_sigma,
                                                        Knn=decomp_sigma,
                                                        f=self.y,
                                                        full_cov=True)
            
            samples = 1000
            random_indices = np.random.permutation(self.x.shape[0])
            x_post = self.x[random_indices]
            y_data_post = self.y[random_indices]
            
            decomp_sigma_post = component.kernel.K(x_post)
            
            data_mean_post = gaussian_conditional(Kmn=decomp_sigma_post,
                                                  Lmm=L_sigma,
                                                  Knn=decomp_sigma_post,
                                                  f=self.y)
            
            # y_post =(y_data_post - data_mean_post) # TODO: random sample
            
            # random_indices = np.random.permutation(self.x.shape[0])
            # x_data, y_data = self.x[random_indices], self.y[random_indices]
            # A = np.hstack([x_data, y_data])
            # B = np.hstack([x_post, y_post])
            
            # TODO: standarized A, B
            
            # mmd_value, p_value = mmd_test(A, B, n_shuffle=samples)


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

    n = data_mean.shape[0]
    prior_samples = np.matmul(prior_L.transpose(), np.random.randn(n, samples))
    prior_quantiled, prior_qq_d_max, prior_qq_d_min, prior_mean, prior_high, prior_low = compute_quantile(prior_samples)
    
    post_samples = data_mean + np.matmul(post_L.transpose(), np.random.randn(n, samples)) # sample from 
    post_quantiled, post_qq_d_max, post_qq_d_min, post_mean, post_high, post_low = compute_quantile(post_samples)
    
    qq_d_max = np.mean(prior_qq_d_max > post_qq_d_max + 1e-5 * np.max(post_qq_d_max) * np.random.randn(*post_qq_d_max.shape))
    qq_d_min = np.mean(prior_qq_d_min < post_qq_d_min + 1e-5 * np.max(post_qq_d_min) * np.random.randn(*post_qq_d_min.shape))
    
    plt.figure()
    plt.scatter(prior_quantiled, post_quantiled)
    plt.fill_between(prior_quantiled, prior_high, prior_low, color='blue', alpha=0.4)
    plt.plot(prior_quantiled, prior_mean)
    plt.plot(prior_quantiled, post_mean)
    plt.plot(prior_quantiled, post_high)
    plt.plot(prior_quantiled, post_low)
    plt.savefig("dummy.png")
    
    return qq_d_max, qq_d_min


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

    two_band_plot(x=np.arange(1, post_acf.shape[0]+1) * grid_distance,
                  mean_1=np.mean(prior_acf, axis=1),
                  high_1=np.quantile(prior_acf, 0.95, axis=1),
                  low_1=np.quantile(prior_acf, 0.05, axis=1),
                  mean_2=np.mean(post_acf, axis=1),
                  high_2=np.quantile(post_acf, 0.95, axis=1),
                  low_2=np.quantile(post_acf, 0.05, axis=1))
    
    return acf_min_loc, acf_min

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
    
    two_band_plot(x=np.linspace(0,1, prior_pxx.shape[0]),
                  mean_1=np.mean(prior_pxx, axis=1),
                  high_1=np.quantile(prior_pxx, 0.95, axis=1),
                  low_1=np.quantile(prior_pxx, 0.05, axis=1),
                  mean_2=np.mean(post_pxx, axis=1),
                  high_2=np.quantile(post_pxx, 0.95, axis=1),
                  low_2=np.quantile(post_pxx, 0.05, axis=1))
    
    return pxx_max_loc, pxx_max

def two_band_plot(x, mean_1, high_1, low_1, mean_2, high_2, low_2):
    
    plt.figure()
    
    # first band
    plt.plot(x, mean_1)
    plt.fill_between(x, high_1, low_1, color='blue', alpha=0.4)
    
    # second band
    plt.plot(x, mean_2)
    plt.plot(x, high_2, "--")
    plt.plot(x, low_2, "--")
    
    plt.savefig("dummy.png")
    
    
    
    pass

def distmat(x):
    return cdist(x, x)

def sigma_estimation(x, y):
    
    D = distmat(np.vstack([x, y]))
    tri_indices = np.tril_indices(D.shape[0], -1)
    tri = D[tri_indices]
    med = np.median(tri)
    
    if med <= 0:
        med = np.mean(tri)
        
    if med < 1e-12:
        med = 1e-2
    
    return med
    
def permutation_test(matrix, n1, n2, n_shuffle=1000):
    
    a00 = 1./ (n1*(n1-1))
    a11 = 1./ (n2 *(n2-1))
    a01 = -1./ (n1 * n2)
    
    n = n1 + n2
    
    pi = np.zeros(n, dtype=np.int8)
    pi[n1:] = 1
    
    larger = 0.
    for sample_n in range(1 + n_shuffle):
        count = 0.
        for i in range(n):
            for j in range(i, n):
                mij = matrix[i,j] + matrix[j,i]
                if pi[i] == pi[j] == 0:
                    count += a00 * mij
                elif pi[i] == pi[j] == 1:
                    count += a11 * mij
                else:
                    count += a01 * mij
        
        if sample_n == 0:
            statistic = count
        elif statistic <= count:
            larger+= 1
        
        np.random.shuffle(pi)
        
    return larger / n_shuffle

def mmd_test(x, y, sigma=None, n_shuffle=1000):
    
    m = x.shape[0]
    H = np.eye(m) - (1./m) * np.ones((m,m))
    
    Dxx = distmat(x)
    Dyy = distmat(y)
    
    if sigma:
        Kx = np.exp(-Dxx/ (2.*sigma**2))
        Ky = np.exp(-Dyy / (2.*sigma**2))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x, y)
        Kx = np.exp(-Dxx / (2.*sx *sx))
        Ky = np.exp(-Dyy/ (2.*sy * sy))
    
    distance = distmat(np.vstack([x, y]))
    kernel = np.exp(-distance / (sxy * sxy))
    Kxy = kernel[:m, m:]
    
    mmdval = np.mean(Kx) + np.mean(Ky) - 2 * np.mean(Kxy)
    
    p_value = permutation_test(kernel, n1=x.shape[0], n2=y.shape[0], n_shuffle=n_shuffle)
    return mmdval, p_value

if __name__ == '__main__':

    from kernel_discovery.kernel import RBF, White
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
        
    def test_result_object():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)
        result = Result(x, y, kernel, noise=np.array(0.1), root="./figure")
        result.process()
        
        for component in result.components:
            print(component)
            
    def test_mmd():
        x = np.random.randn(100,2)
        y = np.random.randn(200,2)
        value = mmd_test(x, y)
        print(value)
        
    def test_compute_quantile():
        samples = np.random.randn(2, 1000)
        result = compute_quantile(samples)
        print(result)
        
    def test_make_qq_plot():
        data_mean = np.random.randn(100,1)
        x = np.linspace(0,10, 100)[:, None]
        kernel = RBF() + RBF(lengthscales=0.5) + White(variance=0.1)
        K = kernel.K(x)
        L_post = tf.linalg.cholesky(K).numpy()
        kernel = RBF() + White(variance=0.1)
        L_prior = tf.linalg.cholesky(kernel.K(x)).numpy()
        make_qqplot(data_mean, L_prior, L_post)
        
    def test_compute_acf():
        x = np.random.randn(10, 1000)
        result = compute_acf(x)
        print(result)
        
    def test_make_acf():
        data_mean = np.random.randn(100,1)
        x = np.linspace(0,10, 100)[:, None]
        grid_distance = x[1] - x[0]
        kernel = RBF() + RBF(lengthscales=0.5) + White(variance=0.1)
        K = kernel.K(x)
        L_post = tf.linalg.cholesky(K).numpy()
        kernel = RBF() + White(variance=0.1)
        L_prior = tf.linalg.cholesky(kernel.K(x)).numpy()
        make_acf_plot(data_mean, L_prior, L_post, grid_distance)
        
        
    def test_compute_periodogram():
        x = np.random.randn(10, 1000)
        result = compute_periodogram(x)
        print(result) 

    def test_make_periodogram():
        data_mean = np.random.randn(100,1)
        x = np.linspace(0,10, 100)[:, None]
        kernel = RBF() + RBF(lengthscales=0.5) + White(variance=0.1)
        K = kernel.K(x)
        L_post = tf.linalg.cholesky(K).numpy()
        kernel = RBF() + White(variance=0.1)
        L_prior = tf.linalg.cholesky(kernel.K(x)).numpy()
        make_peridogram(data_mean, L_prior, L_post)
        
    # test_plot_gp()
    # test_sample_plot_gp()
    # test_compute_mean_var()
    # test_component_and_stat()
    # test_cummulative_plot()
    # test_order_mae()
    # test_result_object()
    
    # test_mmd()
    # test_compute_quantile()
    # test_make_qq_plot()
    
    # test_compute_acf()
    test_make_acf()
    
    # test_compute_periodogram()
    # test_make_periodogram()
    
