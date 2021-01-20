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

def _get_dataloader(X, y, batchsize, y_tensor=True, random_state=None):
    # Shuffle data
    idx = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    # Construct tensor for X
    X = torch.Tensor(X).float()
    if y_tensor:
        y = torch.Tensor(y).long()
    l = len(X)
    for ndx in range(0, l-(l%batchsize), batchsize):
        yield X[ndx:min(ndx+batchsize,l)], y[ndx:min(ndx+batchsize,l)]
    
def _get_accuracy(outputs, labels):
    acc_t = torch.sum(labels==torch.argmax(outputs,dim=1))/(labels.size(0)*1.0)
    return acc_t.item()

def _hse_loss(input, target):
    print("To be implemented!")

