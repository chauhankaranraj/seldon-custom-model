"""Here goes the prediction code."""
import os
import joblib
from typing import Iterable, Dict, List, Union

import numpy as np


class RFPredictor(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")

    def load(self):
        print("Loading model", os.getpid())
        with open('../notebooks/model.joblib', 'rb') as f:
            self.model = joblib.load(f)
        print("Loaded model")

    def class_names(self) -> Iterable[str]:
        return [f"my_class_{i}" for i in range(10)]

    def transform_input(self, X: np.ndarray, names: Iterable[str], meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        return X


    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print("Predict called - will run identity function")
        return X