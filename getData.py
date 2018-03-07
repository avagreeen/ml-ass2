import numpy as np
import numpy.random as rnd
import math as m
from scipy.stats import multivariate_normal, bernoulli
from copy import copy
from sklearn import preprocessing as pp

import csv
import numpy as np
from sklearn import datasets
import random

def load_file(filename):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    data = np.array(list(reader))
    X = data[:,:len(data[0]) - 1]
    X = np.array(X).astype('float')
    Y = data[:, -1]
    return (X,Y)

def load_magic():
    X, Y = load_file('magic04.data')
    for i in range(len(Y)):
        if Y[i] == 'g':
            Y[i] = 0
        elif Y[i] == 'h':
            Y[i] = 1
    X=pp.scale(X)
    return (X, Y)

def split_data(X,y,N_unlabeled):
    #get all the training data
    items = random.sample(range(len(y)),25+N_unlabeled)
    X_train = X[items,]
    y_train_true=y[items,]
    u_items = random.sample(range(len(y_train_true)),N_unlabeled)
    
    y_train = copy(y_train_true)
    train_mask_unl = np.zeros(y_train_true.shape, dtype=bool)
    train_mask_unl[u_items] = True
    y_train[train_mask_unl] = -1
    
    #get test data
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[items]=True
    X_test,y_test=X[~train_mask,:],y[~train_mask]
    
    return X_train, y_train, y_train_true, X_test, y_test