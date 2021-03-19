"""
Code for hierarchical multi-class classifiers.
Author: Thomas Mortier
Date: Feb. 2021

TODO: 
    * Fix issue with root -> CH -> ... situations
    * Introduce range for k
"""
import time

import numpy as np

from hclf.utils import HLabelEncoder

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed, parallel_backend
from collections import ChainMap

class LCPN(BaseEstimator, ClassifierMixin):
    """Local classifier per parent node (LCPN) classifier.

    Parameters
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the classification task in each node.
    sep : str, default=';'
        Path separator used for processing the hierarchical labels. If set to None,
        a random hierarchy is created and provided flat labels are converted,
        accordingly.
    k : int, default=2 
        Max number of children a node can have in the random generated tree. Is ignored when
        sep is set to None.
    n_jobs : int, default=None
        The number of jobs to run in parallel. Currently this applies to fit, 
        and predict.  
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages

    Examples
    --------
    >>> from hclf.multiclass import LCPN
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> clf = LCPN(LogisticRegression(random_state=0),
    >>>         sep=";",
    >>>         n_jobs=4,
    >>>         random_state=0,
    >>>         verbose=1)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    """
    def __init__(self, estimator, sep=';', k=2, n_jobs=None, random_state=None, verbose=0):
        self.estimator = estimator
        self.sep = sep
        self.k = k
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tree = {"root": {
                "lbl": "root",
                "estimator": None,
                "children": [],
                "parent": None}}
    
    def _add_path(self, path):
        current_node = "root"
        for i in range(len(path)):
            add_node = path[i]
            # check if add_node is already registred
            if add_node not in self.tree:
                # check if add_node is terminal
                if i < len(path)-1:
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
            current_node = add_node

    def _fit_node(self, node):
        # check if node has estimator
        if node["estimator"] is not None:
            # transform data for node
            y_transform = []
            sel_ind = []
            if node["lbl"] == "root":
                y_transform = [yi.split(self.sep)[0] for yi in self.y_]
                sel_ind = list(range(len(self.y_)))
            else:
                for i,y in enumerate(self.y_):
                    if node["lbl"] in y.split(self.sep):
                        # need to include current label and sample (as long as it's "complete")
                        y_split = y.split(self.sep)
                        if y_split.index(node["lbl"]) < len(y_split)-1:
                            y_transform.append(y_split[y_split.index(node["lbl"])+1])
                            sel_ind.append(i)
            X_transform = self.X_[sel_ind,:]
            node["estimator"].fit(X_transform, y_transform)
            if self.verbose >= 2:
                print("Model {0} fitted!".format(node["lbl"]))
            # now make sure that the order of labels correspond to the order of children
            node["children"] = node["estimator"].classes_
        return {node["lbl"]: node}
        
    def fit(self, X, y):
        """Implementation of the fitting function for the LCPN classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The class labels

        Returns
        -------
        self : object
            Returns self.
        """
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
        # check if sep is None or str
        if type(self.sep) != str and self.sep is not None:
            raise TypeError("Parameter sep must be of type str or None.")
        # init and fit the hierarchical model
        start_time = time.time()
        # first init the tree 
        try:
            if self.sep is None:
                # transform labels to labels in some random hierarchy
                self.sep = ';'
                self.label_encoder_ = HLabelEncoder(k=self.k,random_state=self.random_state_)
                self.y_ = self.label_encoder_.fit_transform(self.y_) 
            else:
                self.label_encoder_ = None
            for lbl in self.y_:
                self._add_path(lbl.split(self.sep))
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
        # make sure that classes_ are in same format of original labels
        if self.label_encoder_ is not None:
            self.classes_ = self.label_encoder_.inverse_transform(self.classes_)
        else:
            # construct dict with leaf node lbls -> path mappings
            lbl_to_path = {yi.split(self.sep)[-1]: yi for yi in self.y_}
            self.classes_ = [lbl_to_path[cls] for cls in self.classes_]
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "fitting", stop_time-start_time))
        return self
 
    def _predict(self, i, X):
        preds = []
        # run over all samples
        for x in X:
            x = x.reshape(1,-1)
            pred = "root"
            pred_path = []
            while pred in self.tree:
                curr_node = self.tree[pred]
                # check if we have a node with single path
                if curr_node["estimator"] is not None:
                    pred = curr_node["estimator"].predict(x)[0]
                else: 
                    pred = curr_node["children"][0]
                pred_path.append(pred)
            preds.append(self.sep.join(pred_path))
        return {i: preds}
    
    def predict(self, X):
        """Return class predictions.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        avg : boolean, default=True
            Return model average when true, and array of predictions otherwise.

        Returns
        -------
        preds : ndarray
            Returns an array of predicted class labels.
        """
        # check input
        X = check_array(X)
        preds = []
        start_time = time.time()
        try:
            # now proceed to predicting
            with parallel_backend("loky", inner_max_num_threads=1):
                d_preds = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(i,X[ind]) for i,ind in enumerate(np.array_split(range(X.shape[0]), self.n_jobs)))
            # collect predictions
            preds_dict = dict(ChainMap(*d_preds))
            for k in np.sort(list(preds_dict.keys())):
                preds.extend(preds_dict[k])
            # in case of no predefined hierarchy, backtransform to original labels
            if self.label_encoder_ is not None:
                preds = self.label_encoder_.inverse_transform([p.split(self.sep)[-1] for p in preds])
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "predicting", stop_time-start_time))
        return preds
     
    def _predict_proba(self, estimator, X, scores=False):
        if not scores:
            return estimator.predict_proba(X)
        else:
            # get scores
            scores = estimator.decision_function(X)
            scores = np.exp(scores)
            # check if we only have one score (ie, when K=2)
            if len(scores.shape) == 2:
                # softmax evaluation
                scores = scores/np.sum(scores,axis=1).reshape(scores.shape[0],1)
            else:
                # sigmoid evaluation
                scores = 1/(1+np.exp(-scores))
                scores = scores.reshape(-1,1)
                scores = np.hstack([1-scores,scores])
            return scores
    
    def predict_proba(self, X):
        """Return probability estimates.

        Important: the returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        avg : boolean, default=True
            Return model average when true, and array of probability estimates otherwise.

        Returns
        -------
        probs : ndarray
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_.
        """
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
            nodes_to_visit = [(self.tree["root"], np.ones((X.shape[0],1)))]
            while len(nodes_to_visit) > 0:
                curr_node, parent_prob = nodes_to_visit.pop()
                # check if we have a node with single path
                if curr_node["estimator"] is not None:
                    # get probabilities 
                    curr_node_probs = self._predict_proba(curr_node["estimator"], X, scores)
                    # apply chain rule of probability
                    curr_node_probs = curr_node_probs*parent_prob
                    for i,c in enumerate(curr_node["children"]):
                        # check if child is leaf node 
                        prob_child = curr_node_probs[:,i].reshape(-1,1)
                        if c not in self.tree:
                            probs.append(prob_child)
                        else:
                            # add child to nodes_to_visit
                            nodes_to_visit.append((self.tree[c],prob_child))
                else:
                    c = curr_node["children"][0]
                    # check if child is leaf node 
                    if c not in self.tree:
                        probs.append(parent_prob)
                    else:
                        # add child to nodes_to_visit
                        nodes_to_visit.append((self.tree[c],parent_prob))
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.") 
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "predicting probabilities", stop_time-start_time))
        return np.hstack(probs)

    def score(self, X, y):
        """Return mean accuracy score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
       
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # check input and outputs
        X, y  = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        try:
            preds = self.predict(X)
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "calculating score", stop_time-start_time))
        score = accuracy_score(y, preds) 
        return score

    def score_nodes(self, X, y):
        """Return mean accuracy score for each node in the hierarchy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
       
        Returns
        -------
        score_dict : dict
            Mean accuracy of self.predict(X) wrt. y for each node in the hierarchy.
        """
        # check input and outputs
        X, y  = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        score_dict = {}
        try: 
            # transform the flat labels, in case of no predefined hierarchy
            if self.label_encoder_ is not None:
                y = self.label_encoder_.transform(y)
            for node in self.tree:
                node = self.tree[node]
                # check if node has estimator
                if node["estimator"] is not None:
                    # transform data for node
                    y_transform = []
                    sel_ind = []
                    if node["lbl"] == "root":
                        y_transform = [yi.split(self.sep)[0] for yi in y]
                        sel_ind = list(range(len(y)))
                    else:
                        for i, yi in enumerate(y):
                            if node["lbl"] in yi.split(self.sep):
                                # need to include current label and sample (as long as it's "complete")
                                y_split = yi.split(self.sep)
                                if y_split.index(node["lbl"]) < len(y_split)-1:
                                    y_transform.append(y_split[y_split.index(node["lbl"])+1])
                                    sel_ind.append(i)
                    X_transform = X[sel_ind,:]
                    if len(sel_ind) != 0:
                        # obtain predictions
                        node_preds = node["estimator"].predict(X_transform)
                        acc = accuracy_score(y_transform, node_preds)
                        score_dict[node["lbl"]] = acc
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "calculating node scores", stop_time-start_time))
        return score_dict
