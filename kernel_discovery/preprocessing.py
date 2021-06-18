import logging
import numpy as np

logger = logging.getLogger(__package__)

def preprocessing(x, y, rescale_x_to_upper_bound):
    
    x = x.reshape(-1,1).astype(float)
    y = y.reshape(-1,1).astype(float)
    
    if x.shape != y.shape:
        logger.exception(f'Shape of x [{x.shape}] and shape of y [{y.shape}] do not match')
        raise ValueError('Shapes of x and y do not match')

    y -= y.mean()
    
    if not np.isclose(y.std(), 0):
        y /= y.std()
    
    x = rescale(x, rescale_x_to_upper_bound)
    
    return x, y

def rescale(x, rescale_x_to_upper_bound):
    
    if rescale_x_to_upper_bound is None:
        return x
    
    if rescale_x_to_upper_bound <= 0 or x.max() == 0:
        logger.exception(f'Bad upper bound for `x`: `{rescale_x_to_upper_bound}`, or bad maximum of `x` to rescale `{x.max()}`')
        raise ValueError('Bad upper bound to rescale `x` or bad maximum for `x` to perform rescaling')
    
    return rescale_x_to_upper_bound * x / x.max()    
