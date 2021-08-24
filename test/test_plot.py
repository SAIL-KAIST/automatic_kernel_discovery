import pytest

import numpy as np
from kernel_discovery.plot import plot_gp, sample_plot_gp


def test_plot_gp():
    
        x = np.linspace(0, 5, 100)
        y = np.sin(x) + 0.01 * np.random.randn(100)
        x_extrap = np.linspace(-0.5, 5.5, 200)
        mean = np.sin(x_extrap)
        var = np.ones(200)*0.1

        fig, ax = plot_gp(x, y, x_extrap, mean, var)

        fig.savefig("dummy.png")

def test_sample_plot_gp():

    x = np.linspace(0, 5, 100)
    y = np.sin(x) + 0.01 * np.random.randn(100)
    x_extrap = np.linspace(0, 5.5, 200)
    mean = np.sin(x_extrap)
    var = np.eye(200)*0.1

    fig, _ = sample_plot_gp(x, x_extrap, mean, var)

    fig.savefig("dummy.png")