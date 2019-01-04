####################################################################
# Big Data Analysis
# Library  2.0 (Paper One)
# Header File Definitions

import numpy as np
########################################################################
from scipy.spatial.distance  import pdist, squareform, euclidean

# If one, get correlation or get covariance
def distcorr(X, Y, corr = 1):
    """ Compute the distance correlation function
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]

    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')

    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    if corr ==1:
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        return(np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))) 
    else:
        return( (A * B).sum()/float(n * n) )


## Dependence Matrix calculation
def dependence_calculation(X, corr = 1):
    m   = X.shape[1];
    C   = np.zeros((m,m))
    rng = np.random.RandomState(0)
    P   = X[rng.randint(X.shape[0], size= 100),:]
    for i in xrange(m):
        for j in xrange(i+1): 
            if i == j and corr ==1:
                C[i][j] = 1.0
            else:
                C[i][j] = distcorr(P[:,i], P[:,j], corr);
            C[i][j] = distcorr(P[:,i], P[:,j], corr);
            C[j][i] = C[i][j];
    return C