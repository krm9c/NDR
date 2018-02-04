# import public libraries
# Lets do this, get the rolling element data in and run the logistic regression for it
from tqdm import tqdm
import os, sys
import tensorflow as tf
import gzip, cPickle
import numpy as np
from sklearn import preprocessing
import math
from SparsePCA import *
import gc
from random import random


## Suppress warnings
import warnings
warnings.filterwarnings("ignore")

## Import our own libraries
from Helper_func import import_pickled_data
dataset = "madelon"
No, y_train, T, y_test = import_pickled_data(dataset)


# Transform the train data-set
scaler = preprocessing.StandardScaler(with_mean = True,\
 with_std = True).fit(No)
X_train = scaler.transform(No)
X_test = scaler.transform(T)
N = 1
print(dataset, "Train", X_train.shape, "Test", X_test.shape)
from Helper_func import reduced_dimension_data
sco = np.zeros((1,9))
acc = np.zeros((1,9))
n_comp = 2
g_size = 2
rng = np.random.RandomState(0)
from Helper_func import comparison
for i in tqdm(xrange(N)):
    start = time.time()
    Train, Test = reduced_dimension_data(X_train, X_test, [10, 50, 0.50]) 
    print("Finished transform", time.time()-start)
    print("Train", Train.shape, "Test", Test.shape)
    names, sco[i,:], acc[i,:] = comparison(Train, y_train, Test, y_test)
print("Name")
print(names)
print("p-values are")
print(sco.mean(axis = 0))
print("accuracies are")
print(acc.mean(axis = 0))