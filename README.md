# Hierarchical Classification

**hclf** is a Python module for hierarchical classification. Currently, the module supports the following models, built on top of sckikit-learn with support for all probabilistic base classifiers: local classifier per parent node (LCPN), local classifier per node (LCN). There is also a scikit-learn compatible hierarchical softmax implemenation, which makes use of PyTorch. **Important: work in progress**

## Installation 

### Dependencies 

Following modules are required:

* NumPy 
* Scikit-learn
* PyTorch
* Joblib
* SciPy (< 1.6.0)

### User installation

## Basic usage

```python
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from hclf.multiclass import HSoftmax, LCPN
from sklearn.linear_model import LogisticRegression

# Example feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 10, stride=4, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 10, stride=3, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self,x):
        h = self.features(x)
        h = h.flatten(1)
        return h

X, y = ... # See below for more information wrt format labels
# Construct a LCPN classifier
clf = LCPN(LogisticRegression(random_state=0),
        oversample=True,
        min_size=10,
        sep=";",
        n_jobs=4,
        random_state=0,
        verbose=1)
clf.fit(X, y)

# Construct a h-softmax classifier
phi = FeatureExtractor() # feature extractor
clf = HSoftmax(phi,hidden_size=1376,lr=0.01,epochs=2,verbose=2)
clf.fit(X, y)
```

Example file containing labels (i.e., paths in some hierarchy):
```
root;Family1;Genus1;Species1
root;Family1;Genus1;Species2
root;Family1;Genus1;Species3
root;Family1;Genus2;Species4
root;Family1;Genus2;Species5
root;Family2;Genus3;Species6
...
```

## References

- A survey of hierarchical classiﬁcation across different application domains, Silla et al.
- A Scalable Hierarchical Distributed Language Model, Mnih et al.
