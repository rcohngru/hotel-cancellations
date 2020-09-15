import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score

def balance_train_data(X, y, method=None):
    '''
    Balances the data passed in according to the specified method.
    Returns balanced numpy arrays.
    '''
    if method == None:
        return X, y
    
    elif method == 'undersampling':
        rus = RandomUnderSampler()
        X_train, y_train = rus.fit_resample(X, y)
        return X_train, y_train
    
    elif method == 'oversampling':    
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X, y)
        return X_train, y_train
    
    elif method == 'smote':
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X, y)
        return X_train, y_train
    
    elif method == 'both':
        smote = SMOTE(sampling_strategy=0.75)
        under = RandomUnderSampler(sampling_strategy=1)
        X_train, y_train = smote.fit_resample(X, y)
        X_train, y_train = under.fit_resample(X_train, y_train)
        return X_train, y_train

    else:
        print('Incorrect balance method')
        return

def plot_cross_val(models, X, y, ax, sampling_method, names, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)

    
    precisions = [] 
    recalls = []
    for i in range(len(models)):
        precisions.append([])
        recalls.append([])
    
    for train, test in kf.split(X):
        X_test, y_test = X[test], y[test]
        X_train, y_train = X[train], y[train]
        
        X_train, y_train = balance_train_data(X_train, y_train, method=sampling_method)
         
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            precisions[i].append(precision_score(y_test, y_pred))
            recalls[i].append(recall_score(y_test, y_pred))
    
    x = range(0, n_splits)
    colormap = {0 : 'r',
                1 : 'b',
                2 : 'g', 
                3 : 'c', 
                4 : 'm'}
    
    
    for i in range(len(models)):
        ax.plot(x, precisions[i], c=colormap[i], 
                linewidth=1, linestyle='-',
                label='%s Precision' % names[i])
        ax.plot(x, recalls[i], c=colormap[i], 
                linewidth=1, linestyle='--',
                label='%s Recall' % names[i])