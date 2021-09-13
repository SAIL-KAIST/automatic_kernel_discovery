from logging import log

from gpflow.models.training_mixins import Data
from kernel_discovery.preprocessing import DataShape
from typing import Dict
from numpy.random import rand, normal
import numpy as np

import gpflow

from gpflow.kernels import (
    Constant,
    Kernel,
    RBF,
    Linear as gpflow_Linear,
    Periodic as gpflow_Periodic,
    RationalQuadratic,
    Polynomial,
    Sum,
    Product,
    ChangePoints,
    White
)


class Periodic(gpflow_Periodic):

    def __init__(self, variance=1., lengthscales=1., period=1.):
        base_kernel = RBF(variance=variance, lengthscales=lengthscales)
        super().__init__(base_kernel, period=period)


class Linear(gpflow_Linear):

    def __init__(self, variance=1., location=0., active_dims=None):
        super().__init__(variance=variance, active_dims=active_dims)
        self.location = gpflow.Parameter(location)
        self._validate_ard_active_dims(self.location)

    def K(self, X, X2=None):
        X = X - self.location
        if X2 is not None:
            X2 = X2 - self.location
        return super().K(X, X2=X2)

    def K_diag(self, X):
        X = X - self.location
        return super().K_diag(X)


BASE_KERNELS: Dict[str, Kernel] = {
    'constant': Constant,
    'rbf': RBF,
    'linear': Linear,
    'periodic': Periodic,
    'rationalquadratic': RationalQuadratic,
    'polynomial': Polynomial,
    'white': White
}

COMBINATION_KERNELS: Dict[str, Kernel] = {
    'sum': Sum,
    'product': Product
}

EPSILON=1e-5

def init_rbf(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    # lengthscale
    if rand() < 0.5:
        log_lengthscale = normal(loc=datashape_x.std, scale=sd)
    else:
        log_lengthscale = normal(loc=np.log(2*(datashape_x.max - datashape_x.min)),
                                 scale=sd)

    # variance
    if rand() < 0.5:
        log_variance = normal(loc=np.log(datashape_y.std), scale=sd)
    else:
        log_variance = normal(loc=0, scale=sd)
    try:
        init_params = RBF(variance=np.exp(log_variance) + EPSILON,
                        lengthscales=np.exp(log_lengthscale) + EPSILON).parameters
    except:
        raise RuntimeError([np.exp(log_variance),np.exp(log_lengthscale)])
    return [p.numpy() for p in init_params]


def init_periodic(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    # lengthscales
    log_lengthscale = normal(loc=0, scale=sd)

    # periodicity
    if rand() < 0.5:
        # no mim_period
        log_period = normal(loc=datashape_x.std-2., scale=sd)
        # TODO: min_period
    else:
        log_period = normal(loc=np.log(datashape_x.max - datashape_x.min) - 3.2,
                            scale=sd)
        # TODO: min_period

    # variance
    if rand() < 0.5:
        log_variance = normal(loc=np.log(datashape_y.std), scale=sd)
    else:
        log_variance = normal(loc=0., scale=sd)

    try:
        init_params = Periodic(variance=np.exp(log_variance) + EPSILON,
                            lengthscales=np.exp(log_lengthscale) + EPSILON,
                            period=np.exp(log_period) + EPSILON).parameters
    except:
        raise RuntimeError([np.exp(log_variance),np.exp(log_lengthscale),np.exp(log_period)])
    return [p.numpy() for p in init_params]


def init_linear(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    r = rand()
    # if r < 1. / 3.:
    #     log_variance = normal(loc=datashape_y.std - datashape_x.std, scale=sd)
    if r < 0.5:
        dist_y = datashape_y.max - datashape_y.min
        dist_x = datashape_x.max - datashape_x.min
        loc = np.log(np.abs(dist_y / dist_x))
        log_variance = normal(loc=loc, scale=sd)
    else:
        log_variance = normal(loc=0., scale=sd)
        
    location = np.random.uniform(low=2 * datashape_x.min - datashape_x.max,
                                 high=2 * datashape_x.max - datashape_x.min)

    init_params = Linear(variance=np.exp(log_variance) + EPSILON,
                         location=location).parameters
    return [p.numpy() for p in init_params]


def init_white(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    if rand() < 0.5:
        log_variance = normal(loc=np.log(datashape_y.std) - np.log(10), scale=sd)
    else:
        log_variance = normal(loc=0., scale=sd)

    init_params = White(variance=np.exp(log_variance) + EPSILON).parameters
    return [p.numpy() for p in init_params]


def init_constant(datashape_x: DataShape, datashape_y: DataShape, sd=1.):
    r = rand()
    if r < 0.5:
        log_variance = normal(loc=np.log(np.abs(datashape_y.mean)), scale=sd)
    else:
        log_variance = normal(loc=0., scale=sd)

    init_params = Constant(variance=np.exp(log_variance) + EPSILON).parameters
    return [p.numpy() for p in init_params]


def init_kernel(kernel_type, datashape_x, data_shape_y, sd=1.):

    if kernel_type is RBF:
        return init_rbf(datashape_x, data_shape_y, sd)
    elif kernel_type is Periodic:
        return init_periodic(datashape_x, data_shape_y, sd)
    elif kernel_type is Linear:
        return init_linear(datashape_x, data_shape_y, sd)
    elif kernel_type is White:
        return init_white(datashape_x, data_shape_y, sd)
    elif kernel_type is Constant:
        return init_constant(datashape_x, data_shape_y, sd)
    else:
        raise ValueError("Unrecognized kernel type")


if __name__ == "__main__":

    # unit test

    from kernel_discovery.io import retrieve
    from kernel_discovery.preprocessing import get_datashape, preprocessing
    ticker, start, end = 'TSLA', '2020-09-12', '2021-09-12'
    x, y, ticker = retrieve(ticker, start, end)
    x, y = preprocessing(x, y, normalize_y_mean=True, normalize_y_std=False)
    datashape_x, datashape_y = get_datashape(x), get_datashape(y)

    param = init_linear(datashape_x, datashape_y)
    print(param)
    param = init_periodic(datashape_x, datashape_y)
    print(param)
    param = init_rbf(datashape_x, datashape_y)
    print(param)
    param = init_white(datashape_x, datashape_y)
    print(param)
    param = init_constant(datashape_x, datashape_y)
    print(param)
