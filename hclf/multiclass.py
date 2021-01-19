"""
Code for hierarchical multi-class classifiers.
Author: Thomas Mortier
"""
import time

import numpy as np

from utils import random_minority_oversampler

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed, parallel_backend

class HSoftmax(BaseEstimator, ClassifierMixin):
    """Hierarchical softmax model.
    Parameters
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representation
        for the hierarchical softmax model.
    hidden_size : int
        Size of the hidden representation.
    lr : float
        Learning rate.
    momentum : float, default=0
        Momentum factor. 
    weight_decay : float, default=0
        Weight decay (L2 penalty).
    dampening : float, default=0
        Dampening for momentum.
    nesterov : bool, default=False
        Enables Nesterov momentum.
    milestones : list, default=[3, 8]
        List of epoch indices. Must be increasing.
    gamma : float, default=0.1
        Multiplicative factor of learning rate decay.
    dropout : float, default=0
        Dropout probability.
    gpu : int, default=1
        Index indicating the GPU device to use for network training.
    batch_size : int, default=32
        Batch size for network training.
    epochs : int, default=10
        Number of training epochs.
    patience : int, default=5
        Patience counter for early stopping.
    n_jobs : int, default=None
        The number of jobs to run in parallel.   
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages
    Examples
    --------
    >>> from hclf.multiclass import HSoftmax
    >>> import numpy as np
    >>> TODO
    """
    def __init__(self, phi, hidden_size, lr, momentum=0.0, weight_decay=0.0, 
            dampening=0.0, nesterov=False, milestones=[3,8], gamma=0.1, dropout=0.0, 
            gpu=1, batch_size=32, epochs=10, patience=5, n_jobs=None, random_state=None, verbose=0):
        self.phi = phi
        self.hidden_size = hidden_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.milestones = milestones
        self.gamma = gamma
        self.dropout = dropout
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Implementation of the fitting function for the hierarchical softmax classifier.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) 
            The class labels, encoded as paths in a tree structure. Each node label is 
            separated by a semicolon.
        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=False)
        # Check if n_jobs is integer
        if not self.n_jobs is None:
            if not isinstance(self.n_jobs, int):
                raise TypeError("Parameter n_jobs must be of type int.")
        # Store the number of outputs, classes for each output and complete data seen during fit
        self.n_outputs_ = 1
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Fit hierarchical model
        start_time = time.time()
        self.model_ = ...
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("HSoftmax", "fitting", stop_time-start_time))

        return self

    def predict(self, X):
        """Return class predictions.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        Returns
        -------
        preds : ndarray
            Returns an array of predicted class labels.
        """
        # Check input
        X = check_array(X)
        start_time = time.time()
        try:
            preds = p.predict(self, X)
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "predicting", stop_time-start_time))
        if avg:
            return np.apply_along_axis(u.get_most_common_el, 1, preds)

        return preds

    def predict_proba(self, X):
        """Return probability estimates.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        Returns
        -------
        probs : ndarray
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_.
        """
        # Check input
        X = check_array(X)
        start_time = time.time()
        try:
            print("TODO")
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' with \
                    appropriate arguments before using this method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("HSoftmax", "predicting probabilities", stop_time-start_time))
        
        return "Not implemented yet!"

    def score(self, X, y, normalize=True, sample_weight=None):
        """Return mean accuracy score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) 
            True labels for X.
        normalize : bool, optional (default=True)
            If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
       
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)
        start_time = time.time()
        try:
            print("TODO")
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("HSoftmax", "calculating score", stop_time-start_time))

        return "Not implemented yet!"

class LCPN(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, oversample=False, min_size=10, sep=";", n_jobs=None, random_state=None, verbose=0):
        self.estimator = estimator
        self.oversample = oversample
        self.min_size = min_size
        self.sep = sep
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tree = {"root": {
                "lbl": "root",
                "estimator": None,
                "children": [],
                "parent": None}}

    def _fit_node(self, node):
        # check if node has estimator
        if node["estimator"] is not None:
            # transform data for node
            y_transform = []
            sel_ind = []
            for i,y in enumerate(self.y_):
                if node["lbl"] in y.split(self.sep):
                    # need to include current label and sample (as long as it's "complete")
                    y_split = y.split(self.sep)
                    if y_split.index(node["lbl"]) < len(y_split)-1:
                        y_transform.append(y_split[y_split.index(node["lbl"])+1])
                        sel_ind.append(i)
            X_transform = self.X_[sel_ind,:]
            # check if we need to apply oversample
            if self.oversample:
                X_train, y_train = randomMinorityOversampler(X_transform, 
                        y_transform, 
                        min_size=self.min_size)
                node["estimator"].fit(X_train, y_train)
            else:
                node["estimator"].fit(X_transform, y_transform)
            if self.verbose >= 2:
                print("Model {0} fitted!".format(node["lbl"]))
            # now make sure that the order of labels correspond to the order of children
            node["children"] = node["estimator"].classes_
        return {node["lbl"]: node}

    def _predict_proba(self, estimator, x, scores=False):
        if not scores:
            return estimator.predict_proba(x)
        else:
            # get scores
            scores = estimator.decision_function(x)
            scores = np.exp(scores)
            # check if we only have one score (ie, when K=2)
            if len(scores.shape) == 2:
                # softmax evaluation
                scores = scores/np.sum(scores)
            else:
                # sigmoid evaluation
                scores = 1/(1+np.exp(-scores[0]))
                scores = np.array([[1-scores, scores]])
            return scores
        
    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        # need to make sure that X and y have the correct shape
        X, y = check_X_y(X, y, multi_output=False) # multi-output not supported (yet)
        # check if n_jobs is integer
        if not self.n_jobs is None:
            if not isinstance(self.n_jobs, int):
                raise TypeError("Parameter n_jobs must be of type int.")
        # store number of outputs and complete data seen during fit
        self.n_outputs_ = 1
        self.X_ = X
        self.y_ = y
        # init and fit the hierarchical model
        start_time = time.time()
        # first init the tree 
        try:
            for lbl in self.y_:
                self.addPath(lbl.split(self.sep))
            # now proceed to fitting
            with parallel_backend("loky", inner_max_num_threads=1):
                fitted_tree = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_node)(self.tree[node]) for node in self.tree)
            self.tree = {k: v for d in fitted_tree for k, v in d.items()}
        except NotFittedError as e:
            raise NotFittedError("Tree fitting failed! Make sure that the provided data is in the correct format.")
        # now store classes (leaf nodes) seen during fit
        cls = []
        nodes_to_visit = [self.tree["root"]]
        while len(nodes_to_visit) > 0:
            curr_node = nodes_to_visit.pop()
            for c in curr_node["children"]:
                # check if child is leaf node 
                if c not in self.tree:
                    cls.append(c)
                else:
                    # add child to nodes_to_visit
                    nodes_to_visit.append(self.tree[c])
        self.classes_ = cls
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "fitting", stop_time-start_time))
        return self
 
    def predict(self, X):
        # check input
        X = check_array(X)
        preds = []
        start_time = time.time()
        try:
            # run over all samples
            for x in X:
                x = x.reshape(1,-1)
                pred = "root"
                pred_path = [pred]
                while pred in self.tree:
                    curr_node = self.tree[pred]
                    # check if we have a node with single path
                    if curr_node["estimator"] is not None:
                        pred = curr_node["estimator"].predict(x)[0]
                    else: 
                        pred = curr_node["children"][0]
                    pred_path.append(pred)
                preds.append(self.sep.join(pred_path))
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPNClassifier", "predicting", stop_time-start_time))
        return np.array(preds)

    def predict_proba(self, X):
        # check input
        X = check_array(X)
        scores = False
        probs = []
        start_time = time.time()
        # check whether the base estimator supports probabilities
        if not hasattr(self.estimator, 'predict_proba'):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, 'decision_function'):
                raise NotFittedError("{0} does not support \
                         probabilistic predictions nor scores.".format(self.estimator))
            else:
                scores = True
        try:
            # run over all samples
            for x in X:
                x = x.reshape(1,-1)
                nodes_to_visit = [(self.tree["root"], 1.)]
                prob = []
                while len(nodes_to_visit) > 0:
                    curr_node, parent_prob = nodes_to_visit.pop()
                    # check if we have a node with single path
                    if curr_node["estimator"] is not None:
                        # get probabilities 
                        curr_node_probs = self._predict_proba(curr_node["estimator"], x, scores)
                        for i,c in enumerate(curr_node["children"]):
                            # apply chain rule of probability
                            prob_child = curr_node_probs[0,i]*parent_prob
                            # check if child is leaf node 
                            if c not in self.tree:
                                prob.append(prob_child)
                            else:
                                # add child to nodes_to_visit
                                nodes_to_visit.append((self.tree[c],prob_child))
                    else:
                        prob_child = 1*parent_prob
                        c = curr_node["children"][0]
                        # check if child is leaf node 
                        if c not in self.tree:
                            prob.append(prob_child)
                        else:
                            # add child to nodes_to_visit
                            nodes_to_visit.append((self.tree[c],prob_child))
                probs.append(prob)
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.") 
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "predicting probabilities", stop_time-start_time))
        return np.array(probs)

    def score(self, X, y):
        # check input and outputs
        X, y  = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        try:
            preds = self.predict(X)
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "calculating score", stop_time-start_time))
        score = accuracy_score(y, preds) 
        return score

    def score_nodes(self, X, y):
        # check input and outputs
        X, y  = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        # initialize score dict
        score_dict = {}
        for node in self.tree:
            node = self.tree[node]
            if node["estimator"] is not None:
                # transform data for node
                y_transform = []
                sel_ind = []
                for i,yi in enumerate(y):
                    if node["lbl"] in yi.split(self.sep):
                        # need to include current label and sample (as long as it's "complete")
                        y_split = yi.split(self.sep)
                        if y_split.index(node["lbl"]) < len(y_split)-1:
                            y_transform.append(y_split[y_split.index(node["lbl"])+1])
                            sel_ind.append(i)
                X_transform = X[sel_ind,:]
                if len(sel_ind) != 0:
                    # obtain predictions
                    try:
                        node_preds = node["estimator"].predict(X_transform)
                        acc = accuracy_score(y_transform, node_preds)
                        curr_node = node
                        level = 0
                        while (curr_node["parent"] != None):
                            curr_node = self.tree[curr_node["parent"]]
                            level += 1
                        if level not in score_dict:
                            score_dict[level] = []
                        score_dict[level].append((node["lbl"],acc))
                    except NotFittedError as e:
                        print("this model is not fitted yet. cal 'fit' \
                                with appropriate arguments before using this \
                                method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPNClassifier", "calculating tree score", stop_time-start_time))
        return score_dict

    def addPath(self, path):
        current_node = path[0]
        add_node = path[1]
        # check if add_node is already registred
        if add_node not in self.tree:
            # check if add_node is terminal
            if len(path) > 2:
                # register add_node to the tree
                self.tree[add_node] = {
                    "lbl": add_node,
                    "estimator": None,
                    "children": [],
                    "parent": current_node} 
            # add add_node to current_node's children (if not yet in list of children)
            if add_node not in self.tree[current_node]["children"]:
                self.tree[current_node]["children"].append(add_node)
            # set estimator when num. of children for current_node is higher than 1
            if len(self.tree[current_node]["children"]) > 1:
                self.tree[current_node]["estimator"] = type(self.estimator)(**self.estimator.get_params())
        # process next couple of nodes in path
        if len(path) > 2:
            path = path[1:]
            self.addPath(path)
 
    def __str__(self):
        # calculate number of leaves 
        num_leaves = 0
        for n in self.tree:
            for c in self.tree[n]["children"]:
                if c not in self.tree:
                    num_leaves += 1
        tree_str = "---------------------------------------\n"
        tree_str += "Number of internal nodes = {0}".format(len(self.tree))
        tree_str += '\n'
        tree_str += "Number of leaves = {0}".format(num_leaves)
        tree_str += '\n'
        tree_str += "---------------------------------------\n"
        return tree_str
