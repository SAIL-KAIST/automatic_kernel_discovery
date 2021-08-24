import numpy as np
from kernel_discovery.analysis.mmd import mmd_test, mmd_plot

def test_mmd():
    x = np.random.randn(100,2)
    y = np.random.randn(200,2)
    value = mmd_test(x, y)
    print(value)
    
def test_mmd_plot():
    x = np.random.randn(100,2)
    y = np.random.randn(200,2)
    fig, ax = mmd_plot(x, y, sxy=1.)
