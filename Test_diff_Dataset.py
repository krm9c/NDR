
###  2-D Plots 
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


from itertools import product
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# from utils import *
# from config import get_config


# sys.path.append('../Research/data')
# ## Import our own libraries
# from Helper_func import import_pickled_data
# dataset = "../Research/data/mnist"
# No, y_train, T, y_test = import_pickled_data(dataset)


def PCA_visual(Train, y, filename):
    mytargets = list(range(0,10))
    XX_train, yy_train = Train, y
    num_classes = len(np.unique(yy_train))

    num_samples_to_plot = 50000
    import random
    idx = [random.randrange(XX_train.shape[0]) for i in xrange(num_samples_to_plot)]
    X_train, y = XX_train[idx,:], yy_train[idx]  # lets subsample a bit for a first impression
    
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_train)

    # Plotting
    fig    = plt.figure()
    ax     = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1,10))

    xx = X_transformed [:, 0]
    yy = X_transformed [:, 1]

    print("xx", xx.shape, "yy", yy.shape)
    print("y_train", max(y) )
    # plot the 3D data points
    for i in range(num_classes):
        ax.scatter(xx[y==i], yy[y==i], color=colors[i], label=str(i), s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(filename+'.pdf', format='pdf', dpi=600)
    # plt.show()

def NDR_visual(Train, y, filename):
    mytargets = list(range(0,10))
    XX_train, yy_train = Train, y
    num_samples_to_plot = 50000
    num_classes = len(np.unique(yy_train))

    import random
    idx = [random.randrange(XX_train.shape[0]) for i in xrange(num_samples_to_plot)]
    X_train, y = XX_train[idx,:], yy_train[idx]  # lets subsample a bit for a first impression
    
    # Plotting
    fig    = plt.figure()
    ax     = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1,10))

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_train)

    xx = X_transformed [:, 0]
    yy = X_transformed [:, 1]

    print("xx", xx.shape, "yy", yy.shape)
    print("y_train", max(y) )
    # plot the 3D data points
    for i in range(num_classes):
        ax.scatter(xx[y==i], yy[y==i], color=colors[i], label=str(i), s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(filename+'.pdf', format='pdf', dpi=600)



def Tsne_visual(Train, y, filename):
    mytargets = list(range(0,10))
    XX_train, yy_train = Train, y
    num_classes = len(np.unique(yy_train))
    labels = np.arange(num_classes)
    num_samples_to_plot = 50000
    import random
    idx = [random.randrange(XX_train.shape[0]) for i in xrange(num_samples_to_plot)]
    X_train, y = XX_train[idx,:], yy_train[idx]  # lets subsample a bit for a first impression
    
    transformer = TSNE(n_components = 2, perplexity=40, verbose=2)

    X_transformed = transformer.fit_transform(X_train)

    # Plotting
    fig    = plt.figure()
    ax     = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1,10))

    xx = X_transformed [:, 0]
    yy = X_transformed [:, 1]

    print("xx", xx.shape, "yy", yy.shape)
    print("y_train", max(y) )
    # plot the 3D data points
    for i in range(num_classes):
        ax.scatter(xx[y==i], yy[y==i], color=colors[i], label=str(i), s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(filename+'.pdf', format='pdf', dpi=600)
    plt.show()



path_here = '/usr/local/home/krm9c/Documents/Research/Parallel/shwetaNew/data/'
datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'SVHN']
def load_data(datasetName):
    print datasetName
    f = gzip.open(path_here+datasetName+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    ''' try:
        if X_train.shape[2]:
            print X_train.shape, y_train.shape, X_test.shape, y_test.shape
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape '''
    return X_train, y_train, X_test, y_test


N = 1
for i, element in enumerate(datasets):
    print("elements", element)
    X_train, y_train, X_test, y_test  = load_data(element)


    ### Flatten the data and scale it 
    if element in [ 'cifar10', 'cifar100', 'mnist', 'notmnist', 'SVHN']:
        d1, _, _, _ = X_train.shape
        X_train= X_train.reshape((d1, -1)) 
        d1, _, _, _ = X_test.shape
        X_test= X_test.reshape((d1, -1))
    else:
        d1, _ = X_train.shape
        X_train= X_train.reshape((d1, -1)) 
        d1, _ = X_test.shape
        X_test= X_test.reshape((d1, -1))

    scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    Arr     = np.random.choice(X_train.shape[0],5000 )
    X_train = X_train[Arr, :]
    y_train = y_train[Arr,:]
    Arr     = np.random.choice(X_test.shape[0],5000 )
    X_test  =  X_test[Arr, :]
    y_test  = y_test[Arr,:]
    from Helper_func import reduced_dimension_data
    acc = np.zeros((N,8))
    n_comp = 2
    # g_size = 2
    rng = np.random.RandomState(0)
    from Helper_func import comparison
    print("y Train", y_train.shape, "y Test", y_test.shape)
    for i in tqdm(xrange(N)):
        start = time.time()
        Train, Test = reduced_dimension_data(X_train, X_test, [1000, 3, 0.99])
        print("Finished NDR time ", time.time()-start)
        print("Train", Train.shape, "Test", Test.shape)
        names, acc[i,:] = comparison(Train, y_train, Test, y_test)
        print("########################################################")
        print("classifiers", names)
        print("Accuracies", acc[i,:])
        print("########################################################")