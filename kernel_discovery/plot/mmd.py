import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

FIG_SIZE=(6,3)

def mmd_plot(x, y, sxy):
    
    # plot
    t = (fullfact([200, 200]) -0.5) / 200
    x_y = np.vstack([x,y])
    max_min = np.max(x_y, axis=0) - np.min(x_y, axis=0)
    t = t * (1.4 * np.tile(max_min, [t.shape[0], 1]))
    t = t + np.tile(np.min(x_y, axis=0) - 0.2 * max_min, [t.shape[0], 1])
    
    d1 = distmat(np.vstack([x, t]))
    d1 = d1[:x.shape[0], x.shape[0]:]
    K1 = np.exp(-d1 /(sxy**2))
    d2 = distmat(np.vstack([y, t]))
    d2 = d2[:y.shape[0], y.shape[0]:]
    K2 = np.exp(-d2 / (sxy * 2) )
    
    witness = np.sum(K1, axis=0) / x.shape[0] - np.sum(K2, axis=0) / y.shape[0]
    
    witness = np.reshape(witness, (200, 200))
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.imshow(witness, extent=[0, 1, 0, 1])
    
    return fig, ax

def distmat(x):
    return cdist(x, x)

def fullfact(levels):
    
    """See https://pythonhosted.org/pyDOE/factorial.html"""
    
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
        
    return H