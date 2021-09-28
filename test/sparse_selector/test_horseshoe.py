import numpy as np

from kernel_discovery.sparse_selector.horseshoe import HorseshoeDiscovery


def test_horseshoe_discovery():
    
    discover = HorseshoeDiscovery(10)
    
    x = np.linspace(0, 100, 10000)[:, None]
    y = np.sin(x)
    discover.discover(x, y)