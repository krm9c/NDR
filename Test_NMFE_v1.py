# # import public libraries
# # Lets do this, get the rolling element data in and run the logistic regression for it
# from tqdm import tqdm
# import os, sys
# import tensorflow as tf
# import gzip, cPickle
# import numpy as np
# from sklearn import preprocessing
# import math
# from SparsePCA import *
# import gc
# from random import random


# ## Suppress warnings
# import warnings
# warnings.filterwarnings("ignore")


# from itertools import product
# from sklearn.decomposition import PCA
# from sklearn.datasets import fetch_mldata
# from sklearn.utils import shuffle


# import sys
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

# from sklearn.manifold import TSNE
# from matplotlib.pyplot import cm
# import matplotlib.mlab as mlab
# from matplotlib.ticker import NullFormatter
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# # from utils import *
# # from config import get_config


# # sys.path.append('../Research/data')
# # ## Import our own libraries
# # from Helper_func import import_pickled_data
# # dataset = "../Research/data/mnist"
# # No, y_train, T, y_test = import_pickled_data(dataset)



# def PCA_visual(Train, y, filename):
#     mytargets = list(range(0,10))
#     XX_train, yy_train = Train, y
#     num_classes = len(np.unique(yy_train))
#     labels = np.arange(num_classes)


#     num_samples_to_plot = 5000
#     import random
#     idx = [random.randrange(XX_train.shape[0]) for i in xrange(num_samples_to_plot)]
#     X_train, y = XX_train[idx,:], yy_train[idx]  # lets subsample a bit for a first impression
    
#     pca = PCA(n_components=3)
#     X_transformed = pca.fit_transform(X_train)

#     # Plotting
#     fig    = plt.figure()
#     ax     = fig.add_subplot(111, projection='3d')
#     colors = cm.Spectral(np.linspace(0, 1,10))

#     xx = X_transformed [:, 0]
#     yy = X_transformed [:, 1]
#     zz = X_transformed [:, 2]

#     print("xx", xx.shape, "yy", yy.shape)
#     print("y_train", max(y) )
#     # plot the 3D data points
#     for i in range(num_classes):
#         ax.scatter(xx[y==i], yy[y==i], zz[y==i], color=colors[i], label=str(i), s=10)

#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.zaxis.set_major_formatter(NullFormatter())
#     plt.suptitle("NDR for MNIST digits ")
#     plt.axis('tight')
#     plt.legend(loc='best', scatterpoints=1, fontsize=5)
#     plt.savefig(filename+'.pdf', format='pdf', dpi=600)
#     # plt.show()

# def NDR_visual(Train, y, filename):
#     mytargets = list(range(0,10))
#     XX_train, yy_train = Train, y
#     num_classes = len(np.unique(yy_train))
#     labels = np.arange(num_classes)

#     num_samples_to_plot = 5000
#     import random
#     idx = [random.randrange(XX_train.shape[0]) for i in xrange(num_samples_to_plot)]
#     X_train, y = XX_train[idx,:], yy_train[idx]  # lets subsample a bit for a first impression
    
#     # Plotting
#     fig    = plt.figure()
#     ax     = fig.add_subplot(111, projection='3d')
#     colors = cm.Spectral(np.linspace(0, 1,10))

#     xx = X_train [:, 0]
#     yy = X_train [:, 1]
#     zz = X_train [:, 2]

#     print("xx", xx.shape, "yy", yy.shape)
#     print("y_train", max(y) )
#     # plot the 3D data points
#     for i in range(num_classes):
#         ax.scatter(xx[y==i], yy[y==i], zz[y==i], color=colors[i], label=str(i), s=10)

#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.zaxis.set_major_formatter(NullFormatter())
#     plt.suptitle("NDR for MNIST digits ")
#     plt.axis('tight')
#     plt.legend(loc='best', scatterpoints=1, fontsize=5)
#     plt.savefig(filename+'.pdf', format='pdf', dpi=600)



# def Tsne_visual(Train, y, filename):
#     mytargets = list(range(0,10))
#     XX_train, yy_train = Train, y
#     num_classes = len(np.unique(yy_train))
#     labels = np.arange(num_classes)


#     num_samples_to_plot = 5000
#     import random
#     idx = [random.randrange(XX_train.shape[0]) for i in xrange(num_samples_to_plot)]
#     X_train, y = XX_train[idx,:], yy_train[idx]  # lets subsample a bit for a first impression
    
#     transformer = TSNE(n_components = 3, perplexity=40, verbose=2)

#     X_transformed = transformer.fit_transform(X_train)

#     # Plotting
#     fig    = plt.figure()
#     ax     = fig.add_subplot(111, projection='3d')
#     colors = cm.Spectral(np.linspace(0, 1,10))

#     xx = X_transformed [:, 0]
#     yy = X_transformed [:, 1]
#     zz = X_transformed [:, 2]

#     print("xx", xx.shape, "yy", yy.shape)
#     print("y_train", max(y) )
#     # plot the 3D data points
#     for i in range(num_classes):
#         ax.scatter(xx[y==i], yy[y==i], zz[y==i], color=colors[i], label=str(i), s=10)

#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.zaxis.set_major_formatter(NullFormatter())
#     plt.suptitle("NDR for MNIST digits ")
#     plt.axis('tight')
#     plt.legend(loc='best', scatterpoints=1, fontsize=5)
#     plt.savefig(filename+'.pdf', format='pdf', dpi=600)

#     # plt.show()

# mnist = fetch_mldata("MNIST original")
# X = mnist.data / 255.0
# y = mnist.target

# # Transform the train data-set
# # scaler = preprocessing.StandardScaler(with_mean = True,\
# #  with_std = True).fit(No)
# # X_train = scaler.transform(X)
# # X_test = scaler.transform(T)

# X_train = X
# X_test  = X
# N = 1

# print("Train", X_train.shape, "Test", X_test.shape)
# from Helper_func import reduced_dimension_data
# sco = np.zeros((1,9))
# acc = np.zeros((1,9))
# n_comp = 2
# g_size = 2
# rng = np.random.RandomState(0)
# from Helper_func import comparison

# for i in tqdm(xrange(N)):
#     start = time.time()
#     Train, Test = reduced_dimension_data(X_train, X_test, [20, 5, 0.99])
#     print("Finished transform", time.time()-start)
#     print("Train", Train.shape, "Test", Test.shape)
#     # names, sco[i,:], acc[i,:] = comparison(Train, y_train, Test, y_test)


# print('PCAwithoutNDR')
# PCA_visual(X_train, y,'PCA_withoutNDR')

# print('NDR')
# NDR_visual(Train, y, 'NDR')

# print('TSNE_NDR')
# Tsne_visual(Train,y, 'TSNENDR')

# print('TSNE_without')
# Tsne_visual(X_train,y, 'TSNE')


# # print("Name")
# # print(names)

# # print("p-values are")
# # print(sco.mean(axis = 0))

# # print("accuracies are")
# # print(acc.mean(axis = 0))



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

    # plt.show()

# mnist = fetch_mldata("MNIST original")
# X = mnist.data / 255.0
# y = mnist.target
# Transform the train data-set


# scaler = preprocessing.StandardScaler(with_mean = True,\
# with_std = True).fit(No)
# X_train = scaler.transform(X)
# X_test = scaler.transform(T)




data_set = []


for element in data_set:
X_train = X
X_test  = X
N = 1

print("Train", X_train.shape, "Test", X_test.shape)
from Helper_func import reduced_dimension_data
sco = np.zeros((1,9))
acc = np.zeros((1,9))
n_comp = 2
g_size = 2
rng = np.random.RandomState(0)
from Helper_func import comparison

for i in tqdm(xrange(N)):
    start = time.time()
    Train, Test = reduced_dimension_data(X_train, X_test, [20, 5, 0.99])
    print("Finished NDR time ", time.time()-start)
    names, sco[i,:], acc[i,:] = comparison(Train, y_train, Test, y_test)

import timeit
print('PCAwithoutNDR')
start = time.time()
PCA_visual(X_train, y,'PCA')
print("Finished PCA time ", time.time()-start)

print('NDR')
start = time.time()
NDR_visual(Train, y, 'NDR')
print("Finished PCA time ", time.time()-start)

# print('TSNE_without')
print("TSNE time")
start = time.time()
Tsne_visual(X_train,y, 'TSNE')
print("Finished NDR time ", time.time()-start)
