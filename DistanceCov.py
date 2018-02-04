####################################################################
# Big Data Analysis
# Library  1.0 (Paper One)
# Header File Definitions
from math import *
import numpy as np
########################################################################
from scipy.spatial.distance  import pdist, squareform, euclidean
## Distance Correlation calculation
def distance_covariance(x,y):
        x = np.reshape(x, [-1,1])
        y = np.reshape(y, [-1,1])
        N = x.shape[0]

        # First Distance Matrix
        A = squareform(pdist(x, 'euclidean'))
        # Centering the symmetric NxN kernel matrix.
        one_n = np.ones((N,N)) / N
        temp = one_n.dot(A)
        A = A - temp - A.dot(one_n) + temp.dot(one_n)

        # Second Distance Matrix
        B= squareform(pdist(y, 'euclidean'))
        temp = one_n.dot(B)
        # Centering the symmetric NxN kernel matrix.
        B = B - temp - B.dot(one_n) + temp.dot(one_n)

        nu_xy = (1/float(N))*np.sqrt(np.sum(np.multiply(A, B)))
        nu_xx = (1/float(N))*np.sqrt(np.sum(A**2))
        nu_yy = (1/float(N))*np.sqrt(np.sum((B**2)))
        if nu_xx*nu_xy < 1e-3:
            return 1e-3
        else:
            return nu_xy/np.sqrt(nu_xx*nu_yy)

## Dependence Matrix calculation
def dependence_calculation(X):
    print("Dependencies")
    m   = X.shape[1];
    C   = np.zeros((m,m))
    rng = np.random.RandomState(0)
    P   = X[ rng.randint(X.shape[0], size= 1000 ), :]
    for i in xrange(m):
        for j in xrange(i+1):
            if i == j:
                C[i][j] = 1.0
            C[i][j] = distance_covariance(P[:,i], P[:,j]);
            C[j][i] = C[i][j]
    return C
