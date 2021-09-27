import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from kernel_discovery.analysis.util import compute_cholesky


GP_FIG_SIZE = (6,3)
COLOR_PALETTE = sns.color_palette()


def plot_gp_regression(x: np.array, y: np.array, x_extrap: np.array, mean, var, data_only=False, has_data=True):
    """

    Args:
        x (np.array): input x
        y (np.array): output y
        mean ([type]): GP predictive mean
        var ([type]): GP predictive variance
        data_only (bool, optional): Include data to the plot. Defaults to False.
        has_data (bool, optional): Not include data. Defaults to True.
    """

    light_blue = (220./255, 230./255, 1.)
    alpha = 0.6
    lw = 1.2

    x = x.squeeze()
    y = y.squeeze()
    x_extrap = x_extrap.squeeze()

    fig, ax = plt.subplots(figsize=GP_FIG_SIZE)
    if not data_only:
        ax.fill_between(x_extrap,
                         mean + 2 * np.sqrt(var),
                         mean - 2 * np.sqrt(var),
                         color=light_blue, alpha=alpha)

    if has_data:
        ax.plot(x, y, 'k.')

    if not data_only:
        ax.plot(x_extrap, mean, color=COLOR_PALETTE[0], lw=lw)
        
    # draw line if extrapolation
    if (not np.all(mean==0)) and (not np.max(x) == np.max(x_extrap)):
        ylim = ax.get_ylim()
        ax.plot([np.max(x), np.max(x)], ylim, "--", color=(0.3, 0.3, 0.3))
    if (not np.all(mean==0)) and (not np.min(x) == np.min(x_extrap)):
        ylim = ax.get_ylim()
        ax.plot([np.min(x), np.min(x)], ylim, "--", color=(0.3, 0.3, 0.3))
    
    ax.set_xlim(np.min(x_extrap), np.max(x_extrap))    
        
    return fig, ax

def sample_plot_gp(x, x_range, mean, covar, num_samples=3):
    
    lw = 1.2
    n = x_range.shape[0]
    L = compute_cholesky(covar)

    fig, ax = plt.subplots(figsize=GP_FIG_SIZE)
    for i in range(num_samples):
        sample = mean + L @ np.random.randn(n)
        ax.plot(x_range, sample, lw=lw, color=COLOR_PALETTE[i])
    
    # draw line if extrapolation
    if (not np.all(mean==0)) and (not np.max(x) == np.max(x_range)):
        ylim = ax.get_ylim()
        ax.plot([np.max(x), np.max(x)], ylim, "--", color=(0.3, 0.3, 0.3))
    if (not np.all(mean==0)) and (not np.min(x) == np.min(x_range)):
        ylim = ax.get_ylim()
        ax.plot([np.min(x), np.min(x)], ylim, "--", color=(0.3, 0.3, 0.3))
    
    ax.set_xlim(np.min(x_range), np.max(x_range))
    return fig, ax

def two_band_plot(x, mean_1, high_1, low_1, mean_2, high_2, low_2):
    
    """
        Plot two bands describing uncertainty for two data
    """
    
    light_blue = (227./255, 237./255, 1.)
    
    fig, ax = plt.subplots(figsize=GP_FIG_SIZE)
    # first band
    ax.plot(x, mean_1, color=COLOR_PALETTE[0])
    ax.fill_between(x, high_1, low_1, color=light_blue, alpha=0.6)
    
    # second band
    ax.plot(x, mean_2, color=COLOR_PALETTE[1])
    ax.plot(x, high_2, "--", color=COLOR_PALETTE[1])
    ax.plot(x, low_2, "--", color=COLOR_PALETTE[1])
        
    return fig, ax