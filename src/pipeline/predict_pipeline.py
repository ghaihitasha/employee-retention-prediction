import os
import sys
import pandas as pd
import joblib

class PredictPipeline:
    def __init__(self):
        """
        Loads the trained model from artifacts
        """
        model_path = os.path.join("artifacts", "model.pkl")
        self.model = joblib.load(model_path)

    def predict(self, features: pd.DataFrame):
        """
        Takes a dataframe and returns predictions
        """
        predictions = self.model.predict(features)
        return predictions
