from lightwood.api.types import ModelAnalysis
import pickle
import pandas as pd


# Interface that must be respected by predictor objects generated from JSON ML and/or compatible with Mindsdb
class PredictorInterface():
    model_analysis: ModelAnalysis = None

    def __init__(self): pass

    def learn(self, data: pd.DataFrame) -> None: pass

    def adjust(self, data: pd.DataFrame) -> None: pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame: pass

    def save(self, file_path: str):
        with open(file_path, 'w') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'r') as fp:
            return pickle.load(fp)