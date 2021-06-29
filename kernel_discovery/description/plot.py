"""
Python implemetation of https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/matlab/component_stats_and_plots.m
"""
import logging
import numpy as np
import tensorflow as tf
from anytree import Node
from kernel_discovery.description.transform import ast_to_kernel
from kernel_discovery.description.simplify import simplify
from kernel_discovery.kernel import Sum, Kernel


import matplotlib.pyplot as plt

logger = logging.getLogger("plot.py")
num_interp_points = 2000
left_extend = 0.
right_extend = 0.1
env_thresh = 0.99

def plot_gp(x: np.array, y:np.array, x_extrap:np.array, mean, var, data_only=False, has_data=True):
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
    num_samples= 4
    n = x_range.shape[0]
    L = np.linalg.cholesky(covar + jitter * np.eye(n))
    
    samples = [mean + L @ np.random.randn(n) for _ in range(num_samples)]
    
    plt.figure()
    for sample in samples:
        plt.plot(x_range, sample, lw=lw)
    
    

def gaussian_conditional(Kmn, Lmm, Knn, f, full_cov=False):
    
    A = tf.linalg.triangular_solve(Lmm, Kmn, lower=True)
    mean = tf.linalg.matmul(A, f, transpose_a=True)
    covar = Knn -tf.linalg.matmul(A, A, transpose_a=True)
    if not full_cov:
        covar = tf.linalg.diag_part(covar)
    
    return mean.numpy().squeeze(), covar.numpy().squeeze()
    
def compute_mean_var(x: np.array, 
                     x_extrap:np.array, 
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
    

def component_stats(x: np.array, y: np.array, kernel: Node, noise: np.array):
    
    
    x_left = np.min(x) - (np.max(x) - np.min(x)) * left_extend
    x_right = np.max(x) + (np.max(x) - np.min(x)) * right_extend
    x_range = np.linspace(x_left, x_right, num_interp_points)[:, None]
    xrange_no_extrap = np.linspace(np.min(x), np.max(x), num_interp_points)[:, None]
    
    # make sure to simplify kernel which returns sum of product kernels
    kernel = simplify(kernel)
    kernel = ast_to_kernel(kernel)
    
    if isinstance(kernel, Sum):
        components = kernel.kernels
    else:
        components = [kernel]
        
    complete_mean, complete_var = compute_mean_var(x, x_range, y, kernel=kernel, component=kernel, noise=noise)
    
    # plot raw data
    plot_gp(x, y, x_range, complete_mean, complete_var, data_only=True)
    logger.info("Plot raw data")
    
    # plot full posterior
    plot_gp(x, y, x_range, complete_mean, complete_var, data_only=True)
    logger.info("Plot full posterior")
    
    # plot sample from full posterior
    complete_mean, complete_covar = compute_mean_var(x, x_range, y, kernel=kernel, component=kernel, noise=noise, full_cov=True)
    sample_plot_gp(x, x_range, complete_mean, complete_covar)
    logger.info("Plot sample")
    
    if len(components) == 1:
        return 
    
    for i, comp in enumerate(components):
        
        mean, var = compute_mean_var(x, xrange_no_extrap, y, kernel, comp, noise)
        
        # this is for compute some statistic
        d_mean, d_var = compute_mean_var(x, x, y, kernel, comp, noise)
        
        excluded_kernel = Sum([k for k in components if k is not comp])
        removed_mean, _ = compute_mean_var(x, x, y, kernel, component=excluded_kernel, noise=noise)
        plot_gp(x, removed_mean, xrange_no_extrap, mean, var)
        logger.info(f"Plot posterior of component {i+1}/{len(components)}")
        
        mean, covar = compute_mean_var(x, x_range, y, kernel, comp, noise, full_cov=True)
        plot_gp(x, removed_mean, x_range, mean, np.diag(covar))
        logger.info(f"Plot posterior of component {i+1}/{len(components)} with extrapolation")
        
        sample_plot_gp(x, x_range, mean, covar)
        logger.info(f"Plot sample for component {i+1}/{len(components)}")
        
        

def cummulative_plots(x:np.array, y:np.array, kernel: Node, noise:np.array):
    
    x_left = np.min(x) - (np.max(x) - np.min(x)) * left_extend
    x_right = np.max(x) + (np.max(x) - np.min(x)) * right_extend
    x_range = np.linspace(x_left, x_right, num_interp_points)[:, None]
    xrange_no_extrap = np.linspace(np.min(x), np.max(x), num_interp_points)[:, None]
    
    kernel = simplify(kernel)
    kernel = ast_to_kernel(kernel)
    
    if isinstance(kernel, Sum):
        components = kernel.kernels
    else:
        components = [kernel]
    
    if len(components) == 1:
        return
    
    accumulate_kernels = []
    for i, comp in enumerate(components):
        
        accumulate_kernels.append(comp)
        current_kernel = Sum(accumulate_kernels)
        
        # plot no extrapolation
        mean, var = compute_mean_var(x, xrange_no_extrap, y, kernel=kernel, component=current_kernel, noise=noise)
        plot_gp(x, y, xrange_no_extrap, mean, var)
        logger.info(f"Plot sum of components up to component {i+1}/{len(components)}")
        
        # plot with extrapolation
        mean, covar = compute_mean_var(x, x_range, y, kernel=kernel, component=current_kernel, noise=noise, full_cov=True)
        plot_gp(x, y, x_range, mean, np.diag(covar))
        logger.info(f"Plot sum of components up to component {i+1}/{len(components)} with extrapolation")
        
        # plot random sample with extrapolation
        sample_plot_gp(x, x_range, mean, covar)
        logger.info(f"Plot sample for sum of components up to component {i+1}/{len(components)} with extrapolation")
        
        
        d_mean, d_var = compute_mean_var(x, x, y, kernel=kernel, component=current_kernel, noise=noise)
        
        residual = y - np.reshape(d_mean, y.shape)
        

        if i < len(components) -1:
            anti_kernels = [k for k in components if k not in accumulate_kernels]
            sum_anti_kernel = Sum(anti_kernels)
            
            mean, var = compute_mean_var(x, xrange_no_extrap, residual, kernel=kernel, component=sum_anti_kernel, noise=noise)
            plot_gp(x, residual, xrange_no_extrap, mean, var)
            logger.info(f"Plot residual after component {i+1}/{len(components)}")    

if __name__ == '__main__':
    
    from kernel_discovery.kernel import RBF
    from kernel_discovery.description.transform import kernel_to_ast
    
    def test_plot_gp():
        
        x = np.linspace(0,5, 100)
        y = np.sin(x) + 0.01 * np.random.randn(100)
        x_extrap = np.linspace(0,5.5, 200)
        mean = np.sin(x_extrap)
        var = np.ones(200)*0.1
        
        plot_gp(x, y, x_extrap, mean, var)
        
        plt.savefig("dummy.png")
        
    def test_sample_plot_gp():
        
        x = np.linspace(0,5, 100)
        y = np.sin(x) + 0.01 * np.random.randn(100)
        x_extrap = np.linspace(0,5.5, 200)
        mean = np.sin(x_extrap)
        var = np.eye(200)*0.1
        
        sample_plot_gp(x, x_extrap, mean, var)
        
        plt.savefig("dummy.png")
                
    def test_compute_mean_var():
        
        x = np.linspace(0, 5, 100)[:, None]
        x_extrap = np.linspace(0, 5.5, 300)[:,None]
        y = np.sin(x)
        
        kernel = RBF() + RBF(lengthscales=0.5)
        component = RBF()
        
        mean, var = compute_mean_var(x, x_extrap, y, kernel, component, noise=0.1)
        
        plot_gp(x.squeeze(), y.squeeze(), x_extrap.squeeze(), mean.squeeze(), var.squeeze())
        
        plt.savefig('dummy_2.png')
        
    def test_component_and_stat():
        
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)
        
        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)
        
        component_stats(x, y, kernel=kernel, noise=np.array(0.1))
        
        plt.savefig('dummy_3.png')
        
    
    def test_cummulative_plot():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)
        
        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)
        
        cummulative_plots(x, y, kernel=kernel, noise=np.array(0.1))
        
    
    # test_plot_gp()
    # test_sample_plot_gp()
    # test_compute_mean_var()
    # test_component_and_stat()
    test_cummulative_plot()
    