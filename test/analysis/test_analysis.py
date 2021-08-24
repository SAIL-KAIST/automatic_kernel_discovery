import pytest
import numpy as np
import tensorflow as tf

from gpflow.kernels import RBF, White
from kernel_discovery.analysis import Result, IndividualAnalysis, CummulativeAnalysis, ModelCheckingAnalysis
from kernel_discovery.analysis.result import (compute_acf, 
                                              compute_periodogram, 
                                              compute_quantile, 
                                              make_acf_plot,
                                              make_qqplot,
                                              make_peridogram)
from kernel_discovery.description import kernel_to_ast

def test_order_by_mae():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)
        
        result = Result(x, y, kernel, noise=np.array(0.1))
        result.order_by_mae_reduction()

def test_compute_quantile():
        samples = np.random.randn(2, 1000)
        result = compute_quantile(samples)
        print(result)
        
def test_compute_acf():
        x = np.random.randn(10, 1000)
        result = compute_acf(x)
        print(result)

def test_compute_periodogram():
        x = np.random.randn(10, 1000)
        result = compute_periodogram(x)
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
        
def test_make_qq_plot():
        data_mean = np.random.randn(100,1)
        x = np.linspace(0,10, 100)[:, None]
        kernel = RBF() + RBF(lengthscales=0.5) + White(variance=0.1)
        K = kernel.K(x)
        L_post = tf.linalg.cholesky(K).numpy()
        kernel = RBF() + White(variance=0.1)
        L_prior = tf.linalg.cholesky(kernel.K(x)).numpy()
        make_qqplot(data_mean, L_prior, L_post)

def test_make_periodogram():
        data_mean = np.random.randn(100,1)
        x = np.linspace(0,10, 100)[:, None]
        kernel = RBF() + RBF(lengthscales=0.5) + White(variance=0.1)
        K = kernel.K(x)
        L_post = tf.linalg.cholesky(K).numpy()
        kernel = RBF() + White(variance=0.1)
        L_prior = tf.linalg.cholesky(kernel.K(x)).numpy()
        make_peridogram(data_mean, L_prior, L_post)
        

def test_all_analysis():
        x = np.linspace(0, 5, 100)[:, None]
        y = np.sin(x)

        kernel = RBF() + RBF(lengthscales=0.5)
        kernel = kernel_to_ast(kernel)
        result = Result(x, y, kernel, noise=np.array(0.1))
        result.order_by_mae_reduction()
        
        individual = IndividualAnalysis(x, y, kernel, noise=np.array(0.1))
        individual.load_components(result.components)
        individual.analyze()
        
        cummulative = CummulativeAnalysis(x, y, kernel, noise=np.array(0.1))
        cummulative.load_components(result.components)
        cummulative.analyze()
        
        checking = ModelCheckingAnalysis(x, y, kernel, noise=np.array(0.1))
        checking.load_components(result.components)
        checking.analyze()



