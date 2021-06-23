from lightwood import Predictor
from lightwood.constants.lightwood import ColumnDataTypes
from collections import Counter
import numpy as np
import pickle
import os


class LightwoodEnsemble:
    def __init__(self, predictors=None, load_from_path=None):
        self.path_list = None
        if load_from_path is not None:
            with open(os.path.join(load_from_path, 'lightwood_data'), 'rb') as pickle_in:
                obj = pickle.load(pickle_in)
                self.path = load_from_path
                self.path_list = obj.path_list
                self.ensemble = [Predictor(load_from_path=path) for path in self.path_list]
        elif isinstance(predictors, Predictor):
            self.ensemble = [predictors]
        elif isinstance(predictors, list):
            self.ensemble = predictors

    def append(self, predictor):
        if isinstance(self.ensemble, list):
            self.ensemble.append(predictor)
        else:
            self.ensemble = [predictor]

    def __iter__(self):
        yield self.ensemble

    def predict(self, when_data):
        predictions = [p.predict(when_data=when_data) for p in self.ensemble]
        formatted_predictions = {}
        for target in self.ensemble[0].config['output_features']:
            target_name = target['name']
            formatted_predictions[target_name] = {}
            pred_arr = np.array([p[target_name]['predictions'] for p in predictions])
            if target['type'] == ColumnDataTypes.NUMERIC:
                final_preds = np.mean(pred_arr, axis=0).tolist()
            elif target['type'] == ColumnDataTypes.CATEGORICAL:
                final_preds = []
                for idx in range(pred_arr.shape[1]):
                    final_preds.append(max(Counter(pred_arr[:, idx])))
            else:
                raise Exception('Only numeric and categorical datatypes are supported for ensembles')

            formatted_predictions[target_name]['predictions'] = final_preds

        return formatted_predictions

    def save(self, path_to):
        # TODO: potentially save predictors inside ensemble pickle, though there's the issue of nonpersistent stuff with torch.save()
        path_list = []
        for i, model in enumerate(self.ensemble):
            path = os.path.join(path_to, f'lightwood_predictor_{i}')
            path_list.append(path)
            model.save(path_to=path)

        self.path_list = path_list

        # TODO: in the future, save preds inside this data struct
        self.ensemble = None  # we deref predictors for now
        with open(os.path.join(path_to, 'lightwood_data'), 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
