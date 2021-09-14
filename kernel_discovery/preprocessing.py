import logging
import numpy as np
from collections import namedtuple

logger = logging.getLogger(__package__)


def preprocessing(x, y, normalize_y_mean=False, normalize_y_std=False):

    x = x.reshape(-1, 1).astype(float)
    y = y.reshape(-1, 1).astype(float)

    if x.shape != y.shape:
        logger.exception(
            f'Shape of x [{x.shape}] and shape of y [{y.shape}] do not match')
        raise ValueError('Shapes of x and y do not match')

    if normalize_y_mean:
        y -= y.mean()

    if normalize_y_std and not np.isclose(y.std(), 0):
        y /= y.std()

    return x, y



DataShape = namedtuple('DataShape', ['N', 'mean', 'std', 'min', 'max'])


def get_datashape(x):
    data_shape = DataShape(
        N=x.shape[0],
        mean=np.mean(x, axis=0).squeeze(),
        std=np.std(x, axis=0).squeeze(),
        min=np.min(x, axis=0).squeeze(),
        max=np.max(x, axis=0).squeeze())
    return data_shape


if __name__ == "__main__":

    # unit test
    from kernel_discovery.io import retrieve
    ticker = 'MSFT'
    start = '2021-01-01'
    end = '2021-02-01'

    x, y = retrieve(ticker, start, end)
    x, y = preprocessing(x, y, rescale_x_to_upper_bound=None)

    datashape_x = get_datashape(x)
    datashape_y = get_datashape(y)

    print(datashape_x)
    print(datashape_y)
