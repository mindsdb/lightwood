import tempfile
from lightwood.api.types import ModelAnalysis
import dill
import pandas as pd


# Interface that must be respected by predictor objects generated from JSON ML and/or compatible with Mindsdb
class PredictorInterface():
    model_analysis: ModelAnalysis = None
    _code: str = None  # Don't touch this pretty please :), not to be confused with __code__

    def __init__(self): pass

    def learn(self, data: pd.DataFrame) -> None: pass

    def adjust(self, data: pd.DataFrame) -> None: pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame: pass

    def save(self, file_path: str):
        with tempfile.NamedTemporaryFile(suffix='.py') as temp:
            temp.write(self._code.encode('utf-8'))
            import importlib.util
            spec = importlib.util.spec_from_file_location('a_temp_module', temp.name)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            with open(file_path, 'wb') as fp:
                dill.dump(self, fp)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'rb') as fp:
            return dill.load(fp)