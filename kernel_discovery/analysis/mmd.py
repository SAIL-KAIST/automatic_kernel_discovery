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

def sigma_estimation(x, y):
    
    D = distmat(np.vstack([x, y]))
    tri_indices = np.tril_indices(D.shape[0], -1)
    tri = D[tri_indices]
    med = np.median(tri)
    
    if med <= 0:
        med = np.mean(tri)
        
    if med < 1e-12:
        med = 1e-2
    
    return med
    
def permutation_test(matrix, n1, n2, n_shuffle=1000):
    
    a00 = 1./ (n1*(n1-1))
    a11 = 1./ (n2 *(n2-1))
    a01 = -1./ (n1 * n2)
    
    n = n1 + n2
    
    pi = np.zeros(n, dtype=np.int8)
    pi[n1:] = 1
    
    larger = 0.
    for sample_n in range(1 + n_shuffle):
        count = 0.
        for i in range(n):
            for j in range(i, n):
                mij = matrix[i,j] + matrix[j,i]
                if pi[i] == pi[j] == 0:
                    count += a00 * mij
                elif pi[i] == pi[j] == 1:
                    count += a11 * mij
                else:
                    count += a01 * mij
        
        if sample_n == 0:
            statistic = count
        elif statistic <= count:
            larger+= 1
        
        np.random.shuffle(pi)
        
    return larger / n_shuffle

def mmd_test(x, y, sigma=None, n_shuffle=1000):
    
    m = x.shape[0]
    H = np.eye(m) - (1./m) * np.ones((m,m))
    
    Dxx = distmat(x)
    Dyy = distmat(y)
    
    if sigma:
        Kx = np.exp(-Dxx/ (2.*sigma**2))
        Ky = np.exp(-Dyy / (2.*sigma**2))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x, y)
        Kx = np.exp(-Dxx / (2.*sx *sx))
        Ky = np.exp(-Dyy/ (2.*sy * sy))
    
    distance = distmat(np.vstack([x, y]))
    kernel = np.exp(-distance / (sxy * sxy))
    Kxy = kernel[:m, m:]
    
    mmdval = np.mean(Kx) + np.mean(Ky) - 2 * np.mean(Kxy)
    
    p_value = permutation_test(kernel, n1=x.shape[0], n2=y.shape[0], n_shuffle=n_shuffle)
    
    fig, ax = mmd_plot(x, y, sxy=sxy)
    
    return fig, ax, mmdval, p_value