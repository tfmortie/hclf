# Hierarchical Classification

**hclf** is a Python module for hierarchical classification, built on top of scikit-learn. Currently, there is support for the following models: local classifier per parent node (LCPN) for multi-class classification and local classifier per node (LCN) for multi-label classification.

## TODO

* Finalize LCN for multi-label classification 

### Dependencies 

Following modules are required:

* NumPy 
* Scikit-learn
* Joblib
* SciPy (< 1.6.0)

### User installation

## Basic usage

```python
from hclf.multiclass import LCPN
from sklearn.linear_model import LogisticRegression

X, y = ... # See below for more information wrt format labels
# Construct a LCPN classifier
clf = LCPN(LogisticRegression(random_state=0),
        sep=";",
        n_jobs=4,
        random_state=0,
        verbose=1)
clf.fit(X, y)
clf.score(X, y)
```

Example file containing hierarchical labels (i.e., full paths in some hierarchy that can be represented by a tree):
```
root;Family1;Genus1;Species1
root;Family1;Genus1;Species2
root;Family1;Genus1;Species3
root;Family1;Genus2;Species4
root;Family1;Genus2;Species5
root;Family2;Genus3;Species6
...
```

The module also supports flat labels, for which ``sep`` must be set to ``None``. In that case, a random hierarchy is generated where the argument ``k`` represents a tuple with the minimum and maximum number of children an internal node can have.

## References

- A survey of hierarchical classiﬁcation across different application domains, Silla et al.
