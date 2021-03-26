"""
Some important functions and classes that are used throughout the module.
Author: Thomas Mortier
"""
import random
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, check_random_state

class HLabelEncoder(TransformerMixin, BaseEstimator):
    """ Hierarchical label encoder which tranforms flat labels to hierarchical labels
    for some random generated hierarchy, represented by a k-ary tree.

    Parameters
    ----------
    k : tuple of int, default=(2,2)
        Min and max number of children a node can have in the random generated tree. Is ignored when
        sep is set to None.
    sep : str, default=';'
        string used for path encodings.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
 
    Examples
    --------
    """
    def __init__(self, k=(2,2), sep=';', random_state=None):
        self.k = k
        self.sep = sep
        self.random_state = random_state

    def fit(self, y):
        """Fit hierarchical label encoder.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.random_state_ = check_random_state(self.random_state)
        y = column_or_1d(y, warn=True)
        if type(self.sep) != str:
            raise TypeError("Parameter sep must be of type str.")
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # label->path in random hierarchy dict
        self.lbl_to_path = {c:[] for c in self.classes_}
        # now process each unique label and get path in random hierarchy
        lbls_to_process = [[c] for c in self.classes_]
        while len(lbls_to_process) > 1:
            self.random_state_.shuffle(lbls_to_process)
            ch_list = []
            for i in range(min(self.random_state_.randint(self.k[0], self.k[1]+1),len(lbls_to_process))):
                ch = lbls_to_process.pop(0)
                for c in ch:
                    self.lbl_to_path[c].append(str(i))
                ch_list.extend(ch)
            lbls_to_process.append(ch_list)
        self.lbl_to_path = {k: '.'.join((v+['root'])[::-1]) for k,v in self.lbl_to_path.items()}
        # also store decoding dict
        self.path_to_lbl = {v:k for k,v in self.lbl_to_path.items()}
        return self

    def fit_transform(self, y):
        """Fit hierarchical label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        self.fit(y)
        return self.transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y_transformed = []
        for yi in y:
            path = self.lbl_to_path[yi].split('.')
            y_transformed.append(self.sep.join(['.'.join(path[:i]) for i in range(1,len(path)+1)]))
        return y_transformed

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y_transformed = []
        for yi in y:
            path = yi.split(self.sep)[-1]
            y_transformed.append(self.path_to_lbl[path])
        return y_transformed
