"""
Code for hierarchical multi-label classifiers.
Author: Thomas Mortier
Date: Feb. 2021

TODO:
    * Make multi-label (i.e., remove normalisation at the end of the tree, etc.)
"""
import os

import sys
import argparse
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

class LCN(BaseEstimator, ClassifierMixin):
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
        if node["lbl"] != "root":
            # transform data for node
            y_transform = []
            sel_ind = []
            for i,y in enumerate(self.y_):
                # check if label of node is in path
                if node["lbl"] in y.split(self.sep):
                    y_transform.append(1)
                else:
                    y_transform.append(0)
            # check if we need to apply oversample
            if self.oversample:
                X_train, y_train = randomMinorityOversampler(self.X_, 
                        y_transform, 
                        min_size=self.min_size)
                node["estimator"].fit(X_train, y_train)
            else:
                node["estimator"].fit(self.X_, y_transform)
            if self.verbose >= 2:
                print("Model {0} fitted!".format(node["lbl"]))
        return {node["lbl"]: node}

    def _predict_proba(self, estimator, x, scores=False):
        # check if estimator is probabilistic
        if not scores:
            return estimator.predict_proba(x)
        else:
            # not probabilistic, hence, get scores
            scores = estimator.decision_function(x)
            # transform to probability distribution by means of S-transform
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
                if len(self.tree[c]["children"]) == 0:
                    cls.append(c)
                else:
                    # add child to nodes_to_visit
                    nodes_to_visit.append(self.tree[c])
        self.classes_ = cls
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCNClassifier", "fitting", stop_time-start_time))
        return self
 
    def predict(self, X):
        # check input
        X = check_array(X)
        scores = False
        # check whether the base estimator supports probabilities
        if not hasattr(self.estimator, 'predict_proba'):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, 'decision_function'):
                raise NotFittedError("{0} does not support \
                         probabilistic predictions nor scores.".format(self.estimator))
            else:
                scores = True
        preds = []
        start_time = time.time()
        try:
            # run over all samples
            for x in X:
                x = x.reshape(1,-1)
                # start at root
                pred = "root"
                pred_path = [pred]
                while len(self.tree[pred]["children"]) >= 1:
                    c_probs = []
                    # run over children and obtain probabilities
                    for c in self.tree[pred]["children"]:
                        c_probs.append(self._predict_proba(self.tree[c]["estimator"], x, scores)[0,1])
                    # pick the child with highest probability and add to the prediction
                    pred = self.tree[pred]["children"][np.argmax(np.array(c_probs))]
                    pred_path.append(pred)
                preds.append(self.sep.join(pred_path))
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCNClassifier", "predicting", stop_time-start_time))
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
                    for i,c in enumerate(curr_node["children"]):
                        # apply chain rule of probability
                        prob_child = self._predict_proba(self.tree[c]["estimator"], x, scores)[0,1]*parent_prob
                        # stop if we are at a leaf
                        if len(self.tree[c]["children"]) == 0:
                            prob.append(prob_child)
                        else:
                            # add child to nodes_to_visit
                            nodes_to_visit.append((self.tree[c], prob_child))
                # normalize probs
                prob = np.array(prob)
                probs.append(prob)
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.") 
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("LCNClassifier", "predicting probabilities", stop_time-start_time))
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
            print(_message_with_time("LCNClassifier", "calculating score", stop_time-start_time))
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
                        y_transform.append(1)
                    else:
                        y_transform.append(0)
                # obtain predictions
                try:
                    node_preds = node["estimator"].predict(X)
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
            print(_message_with_time("LCNClassifier", "calculating tree score", stop_time-start_time))
        return score_dict
        
    def addPath(self, path):
        current_node = path[0]
        add_node = path[1]
        # check if add_node is already registred
        if add_node not in self.tree:
            # register add_node to the tree
            self.tree[add_node] = {
                "lbl": add_node,
                "estimator": type(self.estimator)(**self.estimator.get_params()),
                "children": [],
                "parent": current_node} 
            # add add_node to current_node's children (if not yet in list of children)
            if add_node not in self.tree[current_node]["children"]:
                self.tree[current_node]["children"].append(add_node)
        # process next couple of nodes in path
        if len(path) > 2:
            path = path[1:]
            self.addPath(path)
 
    def __str__(self):
        # calculate number of leaves 
        num_leaves = 0
        for n in self.tree:
            # check if we are at a leaf node
            if len(self.tree[n]["children"]) == 0:
                num_leaves += 1
        tree_str = "---------------------------------------\n"
        tree_str += "Number of internal nodes = {0}".format(len(self.tree)-num_leaves)
        tree_str += '\n'
        tree_str += "Number of leaves = {0}".format(num_leaves)
        tree_str += '\n'
        tree_str += "---------------------------------------\n"
        return tree_str
