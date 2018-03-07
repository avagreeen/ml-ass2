from SSLDA_Classifier import SSLDA_Classifier
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rnd
import math as m
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal, bernoulli
from copy import copy
from getData import *

def errors(X,y,y_true,classifier,n=100):
    mask = np.ones(len(y), dtype=bool)  #mask is len of labels
    mask[np.where(y==-1)[0]]=False    #mask 1  1 1 false 1 1 is where the labeled data
    train_error = 1 - classifier.score(X[mask,:],y[mask])  # X[mask] is the labeled data train with labeled dta
    test_error = 1 - classifier.score(X[~mask,:],y_true[~mask])  #
    return train_error, test_error

def getLikelihood(X,y, method, Nunl, max_iter=100):
    X_train, y_train, y_train_true, X_test, y_test = split_data(X,y,N_unlabeled=Nunl)
    sslda=SSLDA_Classifier(max_iter)
    sslda.fit(X_train, y_train, method=method)
    C1 = np.where(y_train=='0')[0] #indexs of label=0
    C2 = np.where(y_train=='1')[0]
    
    log_proba = sslda.predict_log_proba(X_train)
    loglikelihood = sum(log_proba[C1,0])+ sum(log_proba[C2,0])
    return loglikelihood

def getError(X,y,method,Nunl,max_iter=100):
    X_train, y_train, y_train_true, X_test, y_test = split_data(X,y,N_unlabeled=Nunl)
    labelled = np.where(y_train!=-1)[0]
    sslda = SSLDA_Classifier(max_iter)
    sslda.fit(X_train,y_train, method=method)
    train_err = 1-sslda.score(X_train[labelled,:], y_train_true[labelled])
    
    test_err = 1-sslda.score(X_test, y_test)
    #print(train_err)
    return train_err, test_err


def getErrors(X,y,method,Nunl,repeat,max_iter=100):
    #X_train, y_train, y_train_true, X_test, y_test = split_data(X,y,N_unlabeled=Nunl)
    errors = [getError(X,y,method,Nunl,max_iter) for i in range(0,repeat)]
    train_errors = np.array([error[0] for error in errors])
    test_errors = np.array([error[1] for error in errors])
    return train_errors, test_errors

def plotErrors(X,y, N_unlabelled,methods, repeat, max_iter=100):
#    methods = ['supervised', 'self-training', 'label-propagation']
    errors = {'supervised' : [], 'self-training' : [], 'label-propagation' : []}
    likelihoods = {'supervised' : [], 'self-training' : [], 'label-propagation' : []}
    for method in methods:
        print(method)
        for Nunl in N_unlabelled:
            train_likelihoods = getLikelihoods(X,y,method,Nunl,repeat,max_iter=max_iter)
            '''
            train_errors, test_errors= getErrors (X,y,method, Nunl, repeat, max_iter=max_iter)
            train_error = train_errors.mean()
            test_error = test_errors.mean()
            likelihood = train_likelihoods.mean()
            errors[method].append({'train': train_error, 'test': test_error})
            likelihoods[method].append(likelihood)
            '''
            train_likelihoods = getLikelihoods(X,y,method,Nunl,repeat,max_iter=max_iter)
            train_errors, test_errors = getErrors(X,y,method, Nunl, repeat, max_iter=max_iter)
            train_error = {'mean' : train_errors.mean(), 'std' : train_errors.std()}
            test_error = {'mean' : test_errors.mean(), 'std' : test_errors.std()}
            likelihood = {'mean' : train_likelihoods.mean(), 'std' : train_likelihoods.std()}
            errors[method].append({'train': train_error, 'test': test_error})
            likelihoods[method].append(likelihood)

        print(train_error, test_error)
       # print(likelihood)
        #train_means = 
            
def getLikelihoods(X,y,method, Nunl, repeat, max_iter):
    likelihoods = [getLikelihood(X,y,method, Nunl, max_iter=100) for i in range(0,repeat)]
    return np.array(likelihoods)


        
        
N_unlabelled = [0, 10, 20, 40, 80, 160, 320, 640]
methods = ['supervised', 'self-training', 'label-propagation']
repeat = 10
max_iter=100


X,y=load_magic()
y=y.astype(np.int64)
#X_train, y_train, y_train_true, X_test, y_test = split_data(X,y,N_unlabeled=20)


plotErrors(X,y, N_unlabelled,methods, repeat, max_iter)
