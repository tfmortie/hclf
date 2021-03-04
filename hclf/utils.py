"""
Some important functions that are used throughout the module.
Author: Thomas Mortier
"""
import torch
import numpy as np

def _random_minority_oversampler(X, y, min_size=10):
    # convert y to numpy array
    y = np.array(y)
    classes, classes_counts = np.unique(y, return_counts=True)
    new_X, new_y = [],[]
    for i, cls in enumerate(classes):
        if classes_counts[i] < min_size:
            ind = np.where(y==cls)[0]
            sel_ind = np.random.choice(ind, size=min_size-classes_counts[i])
            new_X.append(X[sel_ind,:])
            new_y.append(y[sel_ind])
    # now merge new data with original
    if len(new_X) != 0:
        X, y = np.concatenate([X,np.concatenate(new_X)]), np.concatenate([np.array(y), np.concatenate(new_y)])
    return X, y
