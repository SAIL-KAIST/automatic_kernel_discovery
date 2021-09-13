from logging import log

from gpflow.models.training_mixins import Data
from kernel_discovery.preprocessing import DataShape
from typing import Dict
from numpy.random import rand, normal
from scipy import stats
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


def positive_sample(loc, scale, lower=1e-5, upper=10e3):
    return stats.truncnorm.rvs(a=lower, b=upper, loc=loc, scale=scale)

def init_rbf(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    # lengthscale
    if rand() < 0.5:
        lengthscale = positive_sample(loc=datashape_x.std, scale=sd)
    else:
        lengthscale = positive_sample(loc=2*(datashape_x.max - datashape_x.min), scale=sd)

    # variance
    if rand() < 0.5:
        variance = positive_sample(loc=datashape_y.std, scale=sd)
    else:
        variance = positive_sample(loc=0, scale=sd)
        
    init_params = RBF(variance= variance,
                    lengthscales=lengthscale).parameters

    return [p.numpy() for p in init_params]


def init_periodic(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    # lengthscales
    lengthscale = positive_sample(loc=0, scale=sd)

    # periodicity
    if rand() < 0.5:
        # no mim_period
        period = positive_sample(loc=datashape_x.std/100., scale=sd)
        # TODO: min_period
    else:
        period = positive_sample(loc=(datashape_x.max - datashape_x.min)/(10**3.2),
                                 scale=sd)
        # TODO: min_period

    # variance
    if rand() < 0.5:
        variance = positive_sample(loc=datashape_y.std, scale=sd)
    else:
        variance = positive_sample(loc=0., scale=sd)

    init_params = Periodic(variance=variance,
                        lengthscales=lengthscale,
                        period=period).parameters

    return [p.numpy() for p in init_params]


def init_linear(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    r = rand()
    if r < 1. / 3.:
         variance = positive_sample(loc=datashape_y.std - datashape_x.std, scale=sd)
    if r < 2. / 3:
        dist_y = datashape_y.max - datashape_y.min
        dist_x = datashape_x.max - datashape_x.min
        loc = np.log(np.abs(dist_y / dist_x))
        variance = positive_sample(loc=loc, scale=sd)
    else:
        variance = positive_sample(loc=0., scale=sd)
        
    location = np.random.uniform(low=2 * datashape_x.min - datashape_x.max,
                                 high=2 * datashape_x.max - datashape_x.min)

    init_params = Linear(variance=variance,
                         location=location).parameters
    return [p.numpy() for p in init_params]


def init_white(datashape_x: DataShape, datashape_y: DataShape, sd=1.):

    if rand() < 0.5:
        variance = positive_sample(loc=datashape_y.std/10, scale=sd)
    else:
        variance = positive_sample(loc=0., scale=sd)

    init_params = White(variance=variance).parameters
    return [p.numpy() for p in init_params]


def init_constant(datashape_x: DataShape, datashape_y: DataShape, sd=1.):
    r = rand()
    if r < 0.5:
        variance = positive_sample(loc=datashape_y.mean, scale=sd)
    else:
        variance = positive_sample(loc=0., scale=sd)

    init_params = Constant(variance=variance).parameters
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
