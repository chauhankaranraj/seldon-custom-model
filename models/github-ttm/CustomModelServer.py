"""Here goes the prediction code."""
import os
import joblib
import boto3
from io import BytesIO
from typing import Iterable, Dict, List, Union

import numpy as np


class CustomModelServer(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        self.s3_resource = boto3.resource(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        )
        print("Loading model from init", os.getpid())
        buffer = BytesIO()
        s3_object = self.s3_resource.Object(
            os.getenv("S3_BUCKET"),
            f"{os.getenv('S3_MODEL_KEY')}/model.joblib",
        )
        s3_object.download_fileobj(buffer)
        self.model = joblib.load(buffer)
        print("Loaded model from init")

    def load(self):
        print("Loading model from load", os.getpid())
        buffer = BytesIO()
        s3_object = self.s3_resource.Object(
            os.getenv("S3_BUCKET"),
            f"{os.getenv('S3_MODEL_KEY')}/model.joblib",
        )
        s3_object.download_fileobj(buffer)
        self.model = joblib.load(buffer)
        print("Loaded model from load")

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
        return self.model.predict_proba(X)