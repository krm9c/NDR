
# First of all the imports 
# lets start with some imports
import os, sys
import numpy as np
import random
from Helper_func import import_pickled_data, extract_samples, subsample
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math

def gen_data(dim, N_size):
    from random import choice
    from sklearn.datasets import make_spd_matrix
    # Define the number of samples 
    num_samples = N_size
    C = make_spd_matrix(dim, random_state=123) + 0.000001*np.identity(dim)
    mean = [0 for i in xrange(dim)] 
    X = np.random.multivariate_normal(mean, C,num_samples)    
    return X, C

Data, Cov = gen_data(2,1000)
print("Data sample", Data.shape, Cov.shape)

from DistanceCov import *
Cov_dist = dependence_calculation(Data, 1)

print("Distance covariance, Covariance", Cov_dist, Cov)