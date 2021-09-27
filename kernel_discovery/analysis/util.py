import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel

def gaussian_conditional_L(Kmn, Lm, Knn, f, full_cov=False):
    
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)
    covar = Knn - tf.linalg.matmul(A, A, transpose_a=True)
    if not full_cov:
        covar = tf.linalg.diag_part(covar)
        
    A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)
    mean = tf.linalg.matmul(A, f, transpose_a=True)

    return mean.numpy().squeeze(), covar.numpy().squeeze()

def gaussian_conditional(Kmn, Kmm, Knn, f, full_cov=False):
    Lm = tf.linalg.cholesky(Kmm)
    return gaussian_conditional_L(Kmn, Lm, Knn, f, full_cov)


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

    mean, var = gaussian_conditional(Kmn=comp_sigma_star,
                                     Kmm=sigma,
                                     Knn=comp_sigma_star2,
                                     f=y,
                                     full_cov=full_cov)

    return mean, var

def get_monotonic(data_mean: np.array, envelop_diag: np.array, env_thresh = 0.99):
    
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


def get_gradient(x, data_mean: np.array, envelop_diag: np.array, env_thresh = 0.99):

    activated = envelop_diag > env_thresh
    data_mean_thresh = data_mean[activated]
    x_thresh = x[activated]
    if len(data_mean_thresh) == 0:
        return 0.
    else:
        return (data_mean_thresh[-1] - data_mean_thresh[0]) / (x_thresh[-1] - x_thresh[0])


def compute_cholesky(K, max_tries=10):
    
    if not isinstance(K, np.ndarray):
        K = K.numpy()
    try:
        L = np.linalg.cholesky(K)
    except:
        K_max = np.max(K)
        for i in range(max_tries):
            jitter = K_max * 10 ** (i-max_tries)
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
                print(f"Added {jitter} to the diagon of matrix. Beware of imprecise result!!!")
                return L
            except:
                print("Increase jitter")
    
    raise RuntimeError("Reaching max jitter try. Cannot perform cholesky ")