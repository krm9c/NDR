"""
Testing For Paper-II
"""
# import all the required Libraries
import math
import numpy as np
import time, os, sys
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import SparsePCA

# Append all the path
sys.path.append('..//CommonLibrariesDissertation')
path_store = '../FinalDistSamples/'
## We have to now import our libraries

from scipy import linalg as LA
from scipy.sparse.linalg import eigsh
from DistanceCov import dependence_calculation
from sklearn import preprocessing


class level():
    def __init__(self):
        self.level_shuffling=[]
        self.group_transformation = []
        self.scaler= []
        self.G_LIST=[]
        self.flag = 0

#  The new group reduction methodology
def novel_groups(T, g_size):
    R_label = [i for i in xrange(T.shape[1])]
    T_label = []
    start = 0;
    step = int(len(R_label)/float(g_size));
    for start in xrange(len(R_label)):
        T_label.extend( [ R_label[i] for i in xrange(start,int(len(R_label)),step)] )
        if(len(T_label)>=len(R_label)):
            break
    return np.array(T[:, T_label]), T_label

def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:];
    return N

## Data In
# Generate reduction vectors
def gen_red_vectors(r, Sigma):
    U, s, V = np.linalg.svd(Sigma)
    pc=  V[0:r,:]
    return pc.T

# Calculate parameters from the train data
def dim_reductionNDR(X, i_dim, o_dim, g_size, alpha):
    Level =[];
    i_len = X.shape[0]

    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size))< o_dim:
        Level.append(level())
        Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(with_mean = False, with_std = True).fit(X))
        D_scaled = Level[len(Level)-1].scaler[0].transform(X)
        Sigma = dependence_calculation(D_scaled)
        V = gen_red_vectors(o_dim, Sigma)
        Level[len(Level)-1].group_transformation.append(V)
        T = D_scaled.dot(V)
        return Level, T
    prev = 0
    while i_dim >= o_dim:
        print("input dim", i_dim)
        # Stopping conditions
        if (i_dim/float(g_size)) < o_dim:
            Final = X
            break
        elif prev == i_dim and Level[len(Level)-1].flag == 1:
            Final = X
            break
        # Initilize for the first level
        Level.append(level())
        if prev == i_dim:
            Level[len(Level)-1].flag = 1
        prev = i_dim;
        # Define the initial arrays for our calculation
        Temp_proj =np.zeros([i_len,1])
        # First create all the groups
        for i in xrange(0, i_dim, g_size):
            if (i+g_size) < i_dim and (i+2*g_size) > i_dim:
                F = i_dim;
            else:
                F = i+g_size;
            if F <= i_dim:
                Level[len(Level)-1].G_LIST.append([j for j in xrange(i,F)])
        if len(Level[len(Level)-1].G_LIST) == 0:
            break
        eigen_final = [];
        for element in Level[len(Level)-1].G_LIST:
            temp = np.array(X[:, np.array(element)]).astype(float)
            Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(\
            with_mean = False, with_std = True).fit(temp))
            D_scaled = Level[len(Level)-1].scaler[len(Level[len(Level)-1].scaler)\
            -1].transform(temp)
            
            # Get the dependency matrix for the group
            Sigma = dependence_calculation(D_scaled)
            # Sigma = np.corrcoef(D_scaled.T)
            # Next achieve the parameters for transformation
            e_vals = LA.eigvals(Sigma)
            # Sort both the eigen value and eigenvector in descending order
            arg_sort  = e_vals.argsort()[::-1][:]
            s_eigvals = e_vals[arg_sort]
            s_eigvals = np.divide(s_eigvals, float(np.sum(s_eigvals)));
            tempsum = np.cumsum(s_eigvals)
            mask = tempsum >= alpha
            temp_number = ((len(tempsum)-len(tempsum[mask]))+1)

            V = gen_red_vectors(temp_number, Sigma)

            Temp_proj = np.column_stack([Temp_proj, D_scaled.dot(V)])
            # Finally get the eigen values and eigenvectors we are carrying
            # forward from this group
            Level[len(Level)-1].group_transformation.append(V)
            eigen_final.extend(e_vals[0:temp_number].astype(np.float).tolist())

        # Next prepare for the level transformaiton
        T = Temp_proj[:,1:Temp_proj.shape[1]]
        pre_shuffle = np.divide(eigen_final, np.sum(eigen_final)).argsort()[::-1][:]
        T = T[:,pre_shuffle]
        # Get the next set of groupings and store the shuffling inside an array
        X, t_shuffling= novel_groups(T, g_size)
        Level[len(Level)-1].level_shuffling.append(pre_shuffle)
        Level[len(Level)-1].level_shuffling.append(t_shuffling)
        # I can start the next level
        i_dim = X.shape[1]
    return Level, Final

## Transform the test samples
def dim_reductionNDR_test(X, Level, i_dim, o_dim, g_size):
    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size) < o_dim):
        D_scaled = Level[len(Level)-1].scaler[0].transform(X)
        T = D_scaled.dot(Level[len(Level)-1].group_transformation[0])
        return T
    p = 0;

    while p <len(Level):
        Temp_proj =np.zeros([X.shape[0],1])
        for j, element in enumerate(Level[p].G_LIST):
            temp = np.array(X[:, np.array(element)]).astype(float)
            # temp_scalar = preprocessing.StandardScaler(with_std =True, with_mean = False).fit(temp)
            D_scaled = Level[p].scaler[j].transform(temp)
            Temp_proj = np.column_stack([Temp_proj, D_scaled.dot(Level[p].group_transformation[j])])
        T = Temp_proj[:,1:Temp_proj.shape[1]]
        X = T[:,Level[p].level_shuffling[0]]
        X = X[:,Level[p].level_shuffling[1]]
        i_dim = X.shape[1]
        p = p+1;

    if Level[len(Level)-1].flag is not 1:
        return X
    elif Level[len(Level)-1].flag is 1:
        return X
