"""Feature extraction code."""
from sklearn.base import BaseEstimator, TransformerMixin


class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X