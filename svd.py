import ConfigParser
import numpy
import codecs
import sys
import time
import random
import math
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
import scipy.sparse as ss
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import gensim
import pickle
from scipy.sparse.linalg import svds, eigs
from scipy.linalg import svd
from sklearn.decomposition import PCA

#X = numpy.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = numpy.random.rand(20,20)
pca = PCA(n_components=5)
Xnew = pca.fit_transform(X)

u,s,v = svd(X)
s = numpy.diag(s)
composition = numpy.dot(numpy.dot(u,s),v)
print('max diff for composition is: %f' % (numpy.max(numpy.abs(composition - X))))


#svdRes = numpy.dot(u,s)#*v

#numpy.max(numpy.abs(svdRes - Xnew))

print('X Dim: [%d x %d]' % (X.shape[0], X.shape[1]))
print('Xnew reducedDim: [%d x %d]' % (Xnew.shape[0], Xnew.shape[1]))

matSym = ss.load_npz('./sym_pats_python_mat.npz')

A = csc_matrix([[1, 0, 0,3], [5, 0, 2,7], [0, -1, 0,5],[1,4,7,2]], dtype=float)
#vals, vecs = ss.linalg.eigs(matSym, k=1000)

vals = svds(matSym, k=2, return_singular_vectors=False)

with open('eeVlasForMatSym', 'wb') as fp:
    pickle.dump(vals, fp)


with open('eeVlasForMatSym', 'rb') as fp:
    vals = pickle.load(fp)

vals=numpy.flip(numpy.sort(vals),0)

cumSumRes=numpy.cumsum(numpy.square(vals)/sum(numpy.square(vals)))

plt.plot(cumSumRes)
plt.title("eigenvalues analysis of matSym")
plt.xlabel("eigenvalue index")
plt.ylabel("comulative sum")


x=1
