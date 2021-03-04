"""
Code for hierarchical multi-class classifiers.
Author: Thomas Mortier
Date: Feb. 2021

TODO:
    * Delete requirement of root
    * Improve Doc
"""
import time

import numpy as np

from hclf.utils import _random_minority_oversampler

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed, parallel_backend
from collections import ChainMap

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
            # check if we need to apply oversample
            if self.oversample:
                X_train, y_train = _random_minority_oversampler(X_transform, 
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
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCPN", "predicting", stop_time-start_time))
        return preds

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
            print(_message_with_time("LCPN", "calculating tree score", stop_time-start_time))
        return score_dict

    def addPath(self, path):
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
