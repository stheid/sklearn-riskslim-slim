# Scikit-learn Wrappers for Slim and Risk-Slim

This package provides a [sklearn-api](https://scikit-learn.org/stable/glossary.html#glossary-estimator-types) compatible wrapper to
- [slim](https://github.com/ustunb/slim-python)
- [risk-slim](https://github.com/ustunb/risk-slim)

# Install
```bash
pip install git+https://github.com/stheid/scikit-slim](https://github.com/stheid/sklearn-riskslim-slim)
```

# Usage

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skslim import Slim

X, y = make_classification(n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

slim = Slim(max_score=3, random_state=42, timeout=30)
slim.fit(X_train, y_train)
print(f"Accuracy: {slim.score(X_test, y_test)}")
```
