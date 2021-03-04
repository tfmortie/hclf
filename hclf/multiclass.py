"""
Code for hierarchical multi-class classifiers.
Author: Thomas Mortier
Date: Feb. 2021

TODO:
    * Delete requirement of root
    * Improve runtime h-softmax
    * Code cleanup (i.e., consistency between HSoftmax and LCPN)
    * Doc
"""
import time
import torch

import numpy as np

from hclf.utils import _random_minority_oversampler, _get_dataloader, _get_accuracy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from joblib import Parallel, delayed, parallel_backend
from collections import ChainMap

class _HSModule(torch.nn.Module):
    def __init__(self, tree_dict, module_list, device, sep):
        super(_HSModule, self).__init__()
        self.tree_dict = tree_dict
        self.module_list = module_list
        self.device = device
        self.sep = sep
        self.ce_loss = torch.nn.CrossEntropyLoss()
 
    def forward(self, x, y=None):
        outputs = []
        loss = None
        if y is not None:
            loss = torch.zeros(len(x)).to(self.device)
            for idx, yi in enumerate(y):
                output = []
                for idy, node in enumerate(yi[:-1]):
                    node_dict = self.tree_dict[node]
                    est = self.module_list[node_dict["estimator"]]
                    if est is not None: 
                        z = est(x)[idx]
                        output.append(z)
                        lbl = node_dict["children"].index(yi[idy+1])
                        loss[idx] += self.ce_loss(z.view(1,-1),torch.tensor([lbl]).to(self.device))
                    else:
                        output.append([1.])
                outputs.append(output)
            loss = loss.mean()
        else:
            for idx in range(len(x)):
                pred = "root"
                pred_path = [pred]
                while pred in self.tree_dict:
                    curr_node = self.tree_dict[pred]
                    # Check if we have a node with single path
                    if self.module_list[curr_node["estimator"]] is not None:
                        pred = curr_node["children"][torch.argmax(self.module_list[curr_node["estimator"]](x)[idx])]
                    else: 
                        pred = curr_node["children"][0]
                    pred_path.append(pred)
                outputs.append(self.sep.join(pred_path))        
        return outputs, loss
 
class HSoftmax(BaseEstimator, ClassifierMixin):
    """Hierarchical softmax model.

    Parameters
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representation
        for the hierarchical softmax model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation.
    lr : float
        Learning rate.
    test_size : float, default=0.2
        Size of internal validation set for training the hidden representations and hierarhical
        softmax model.
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
    sep : str, default=";"
        Character which separates levels for each path in y.
    n_jobs : int, default=None
        The number of jobs to run in parallel.   
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages
    """
    def __init__(self, phi, hidden_size, lr, test_size=0.2, momentum=0.0, weight_decay=0.0, 
            dampening=0.0, nesterov=False, milestones=[3,8], gamma=0.1, dropout=0.0, 
            gpu=1, batch_size=32, epochs=10, patience=5, sep=";", n_jobs=None, random_state=None, verbose=0):
        self.phi = phi
        self.hidden_size = hidden_size
        self.lr = lr
        self.test_size = test_size
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
        self.sep = sep
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _check_estimator(self):
        # Check if phi is nn.torch
        if not isinstance(self.phi, torch.nn.Module):
            raise TypeError("Parameter phi must be of type toch.nn.Module.")
        # Check int params
        int_params = {"hidden_size": self.hidden_size,
                "gpu": self.gpu,
                "n_jobs": self.n_jobs}
        for p in int_params.items():
            if not p[1] is None:
                if not isinstance(p[1], int):
                    raise TypeError("Parameter {0} must be of type int.".format(p[0]))
        # TODO: implement other checks

    def _get_accuracy(self, outputs, labels):
        acc_t = torch.sum(labels==torch.argmax(outputs,dim=1))/(labels.size(0)*1.0)
        return acc_t.item()

    def _fit_phi(self):
        # Create classifier
        clf = torch.nn.Sequential(
                self.phi,
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(self.hidden_size, len(self.classes_)))
        # Split data in training and validation set 
        X_train, X_val, y_train, y_val = train_test_split(self.X_, self.y_, 
                test_size=self.test_size, stratify=self.y_, random_state=self.random_state)
        # Encode labels 
        lbl_to_int_enc = LabelEncoder()
        y_train = lbl_to_int_enc.fit_transform(y_train)
        y_val = lbl_to_int_enc.transform(y_val)
        # Define criterion
        criterion = torch.nn.CrossEntropyLoss()
        # Define optimizer
        optimizer = torch.optim.SGD(clf.parameters(), 
                lr=self.lr, 
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov)
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma) 
        clf.to(self.device_)
        best_loss, patience_cntr = None, 0
        for epoch in range(self.epochs):
            loss_tr = 0.0
            # Create dataloader instance for training and validation
            dataloader_phi_train = _get_dataloader(X_train, y_train, self.batch_size, True, self.random_state)
            dataloader_phi_val = _get_dataloader(X_val, y_val, self.batch_size, True, self.random_state)
            # Run over training set
            for batch_index, (X,y) in enumerate(dataloader_phi_train):
                # Turn on training mode
                clf.train()
                # Zero the parameter gradients
                optimizer.zero_grad()
                X, y = X.to(self.device_), y.to(self.device_)
                # Forward + backward + optimize
                loss = criterion(clf(X), y)
                loss.backward()
                optimizer.step()
                # Obtain average (over batch) loss
                loss_tr += (loss.item()/X.shape[0])
            # Run over validation set
            clf.eval()
            loss_vl, acc_vl = 0.0, 0.0
            for batch_index, (X,y) in enumerate(dataloader_phi_val):
                X, y = X.to(self.device_), y.to(self.device_)
                # Forward pass and calculate statistics
                outputs = clf(X)
                loss = criterion(outputs, y)
                loss_vl += (loss.item()/X.shape[0])
                acc_vl += _get_accuracy(outputs, y)
            loss_tr, loss_vl, acc_vl = loss_tr/(batch_index+1), loss_vl/(batch_index+1), acc_vl/(batch_index+1)
            if self.verbose >= 2:
                print("[info] Epoch {0}, training loss = {1}, " 
                        "validation loss = {2}, validation accuracy = {3}".format(epoch+1, loss_tr, loss_vl, acc_vl))
            # Check if improvement
            if best_loss is None:
                best_loss = loss_vl
            elif loss_vl < best_loss:
                best_loss = loss_vl
                patience_cntr = 0
            else:
                patience_cntr += 1
            # Check if patience counter has exceeded
            if patience_cntr >= self.patience:
                if self.verbose >= 2: 
                    print("[info] patience counter exceeded!")
                break
            scheduler.step()
    
    def _fit_tree(self):
        # Split data in training and validation set 
        X_train, X_val, y_train, y_val = train_test_split(self.X_, self.y_, 
                test_size=self.test_size, stratify=self.y_, random_state=self.random_state)
        # Define optimizer
        optimizer = torch.optim.SGD(self.tree_.parameters(), 
                lr=self.lr, 
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov)
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma) 
        self.tree_.to(self.device_) # TODO: check whether necessary
        best_loss, patience_cntr = None, 0
        for epoch in range(self.epochs):
            loss_tr = 0.0
            # Create dataloader instance for training and validation
            dataloader_phi_train = _get_dataloader(X_train, y_train, 
                    self.batch_size, False, 
                    self.random_state)
            dataloader_phi_val = _get_dataloader(X_val, y_val, 
                    self.batch_size, False, 
                    self.random_state)
            # Run over training set
            for batch_index, (X,y) in enumerate(dataloader_phi_train):
                start_time = time.time()
                # Turn on training mode
                self.tree_.train()
                # Zero the parameter gradients
                optimizer.zero_grad()
                X = X.to(self.device_)
                y = [yi.split(self.sep) for yi in y]
                # Forward + backward + optimize
                _, loss = self.tree_(X, y)
                loss.backward()
                optimizer.step()
                # Obtain average (over batch) loss
                loss_tr += (loss.item()/X.shape[0])
            # Run over validation set
            self.tree_.eval()
            loss_vl, acc_vl = 0.0, 0.0
            for batch_index, (X,y) in enumerate(dataloader_phi_val):
                X = X.to(self.device_)
                # Forward pass and calculate statistics
                y_split = [yi.split(self.sep) for yi in y]
                _, loss = self.tree_(X, y_split)
                preds, _ = self.tree_(X)
                loss_vl += (loss.item()/X.shape[0])
                acc_vl += np.mean(np.array(y==preds))
            loss_tr, loss_vl, acc_vl = loss_tr/(batch_index+1), loss_vl/(batch_index+1), acc_vl/(batch_index+1)
            if self.verbose >= 2:
                print("[info] Epoch {0}, training loss = {1}, " 
                        "validation loss = {2}, validation accuracy = {3}".format(epoch+1, loss_tr, loss_vl, acc_vl))
            # Check if improvement
            if best_loss is None:
                best_loss = loss_vl
            elif loss_vl < best_loss:
                best_loss = loss_vl
                patience_cntr = 0
            else:
                patience_cntr += 1
            # Check if patience counter has exceeded
            if patience_cntr >= self.patience:
                if self.verbose >= 2: 
                    print("[info] patience counter exceeded!")
                break
            scheduler.step()
    
    def _add_path(self, path):
        current_node = path[0]
        add_node = path[1]
        # Check if add_node is already registred
        if add_node not in self.tree_.tree_dict:
            # Check if add_node is terminal
            if len(path) > 2:
                # Register add_node to the tree
                self.tree_.module_list.append(None)
                self.tree_.tree_dict[add_node] = {
                    "lbl": add_node,
                    "estimator": len(self.tree_.module_list)-1,
                    "children": [],
                    "parent": current_node} 
            # Add add_node to current_node's children (if not yet in list of children)
            if add_node not in self.tree_.tree_dict[current_node]["children"]:
                self.tree_.tree_dict[current_node]["children"].append(add_node)
            # Set estimator when number of children for current_node is higher than 1
            if len(self.tree_.tree_dict[current_node]["children"]) > 1:
                # Create classifier
                estimator = torch.nn.Sequential(
                        self.phi,
                        torch.nn.Dropout(p=self.dropout),
                        torch.nn.Linear(self.hidden_size, len(self.tree_.tree_dict[current_node]["children"])))
                self.tree_.module_list[self.tree_.tree_dict[current_node]["estimator"]] = estimator
        # Process next couple of nodes in path
        if len(path) > 2:
            path = path[1:]
            self._add_path(path)

    def fit(self, X, y):
        """Implementation of the fitting function for the hierarchical softmax classifier.
        
        Parameters
        ----------
        X : ndarray
            The training input samples.
        y : array-like, shape (n_samples,) 
            The class labels, encoded as paths in a tree structure. Each level is 
            separated by `sep`.

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        # Check that X and y have correct shape
        if X.shape[0]!=y.shape[0]:
            raise TypeError("X and y should have equal length!")
        # Check params
        self._check_estimator()
        # Store the number of outputs, classes for each output and complete data seen during fit
        self.n_outputs_ = 1
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Get device type
        self.device_ = torch.device('cuda:'+str(self.gpu) if torch.cuda.is_available() else 'cpu')
        # Construct tree
        self.tree_ = _HSModule({"root": {
                "lbl": "root",
                "estimator": 0,
                "children": [],
                "parent": None}},torch.nn.ModuleList([None]),self.device_,self.sep)
        # Fit phi network
        start_time = time.time()
        self._fit_phi()
        # Make sure that we freeze phi afterwards
        for param in self.phi.parameters():
            param.requires_grad = False
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("HSoftmax", "training hidden representations", stop_time-start_time))
        # Fit hierarchical model
        start_time = time.time()
        # First init the tree 
        try:
            for lbl in self.y_:
                self._add_path(lbl.split(self.sep))
        except NotFittedError as e:
            raise NotFittedError("Tree fitting failed! Make sure that the provided data is in the correct format.")
        # And finally train the tree
        self._fit_tree()
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("HSoftmax", "fitting hierarchical softmax", stop_time-start_time))
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
            # TODO
            print("TODO")
        except NotFittedError as e:
            print("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("HSoftmax", "predicting", stop_time-start_time))

        return "Not implemented yet!"

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
            # TODO
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
            # TODO
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
        for i,x in enumerate(X):
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
