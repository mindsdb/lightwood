import numpy as np
import lightgbm
import torch
import logging
import time

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices
from lightwood.mixers import BaseMixer
from sklearn.preprocessing import OrdinalEncoder


class LightGBMMixer(BaseMixer):
    def __init__(self, stop_training_after_seconds=None):
        super().__init__()
        self.models = {}
        self.ord_encs = {}
        self.stop_training_after_seconds = stop_training_after_seconds


        # GPU Only available via custom compiled version: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-gpu-version
        self.device, _ = get_devices()
        #self.device_str = 'cpu' if str(self.device) == 'cpu' else 'gpu'

        self.device = torch.device('cpu')
        self.device_str = 'cpu'

        self.max_bin = 255 # Default value
        if self.device_str == 'gpu':
            self.max_bin = 63 # As recommended by https://lightgbm.readthedocs.io/en/latest/Parameters.html#device_type

    def _fit(self, train_ds, test_ds=None):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """

        data = {
            'train': {'ds': train_ds, 'data': None, 'label_data': {}},
            'test': {'ds': test_ds, 'data': None, 'label_data': {}}
        }

        for subset_name in data:
            cols = data[subset_name]['ds'].input_feature_names
            out_cols = data[subset_name]['ds'].output_feature_names
            for col_name in cols:
                if data[subset_name]['data'] is None:
                    data[subset_name]['data'] = data[subset_name]['ds'].get_encoded_column_data(col_name)
                else:
                    data[subset_name]['data'] = torch.cat((data[subset_name]['data'].to(self.device),
                                                           data[subset_name]['ds'].get_encoded_column_data(
                                                               col_name).to(self.device)), 1)
            data[subset_name]['data'] = data[subset_name]['data'].tolist()
            for col_name in out_cols:
                label_data = data[subset_name]['ds'].get_column_original_data(col_name)
                if next(item for item in train_ds.output_features if item["name"] == col_name)['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                    self.ord_encs[col_name] = OrdinalEncoder()
                    self.ord_encs[col_name].fit(np.array(list(set(label_data))).reshape(-1, 1))
                    label_data = self.ord_encs[col_name].transform(np.array(label_data).reshape(-1, 1)).flatten()

                data[subset_name]['label_data'][col_name] = label_data

        out_cols = train_ds.output_feature_names
        for col_name in out_cols:
            train_data = lightgbm.Dataset(data['train']['data'], label=data['train']['label_data'][col_name])
            validate_data = lightgbm.Dataset(data['test']['data'], label=data['test']['label_data'][col_name])
            dtype = next(item for item in train_ds.output_features if item["name"] == col_name)['type']
            if dtype not in [COLUMN_DATA_TYPES.NUMERIC, COLUMN_DATA_TYPES.CATEGORICAL]:
                logging.info('cannot support {dtype} in lightgbm'.format(dtype=dtype))
                continue
            else:
                objective = 'regression' if dtype == COLUMN_DATA_TYPES.NUMERIC else 'multiclass'

            params = {'objective': objective,
                      'boosting': 'goss',
                      'verbosity': -1,
                      'lambda_l1': 0.1,
                      'lambda_l2': 0.1,
                      'device_type': self.device_str,
                      'max_bin': self.max_bin
                      }
            if objective == 'multiclass':
                all_classes = self.ord_encs[col_name].categories_[0]
                params['num_class'] = all_classes.size

            num_iterations = 100
            if self.stop_training_after_seconds is not None:
                start = time.time()
                params['num_iterations'] = 1
                bst = lightgbm.train(params, train_data, valid_sets=validate_data)
                end = time.time()
                seconds_for_one_iteration = end - start
                logging.info(f'A single GBM itteration takes {seconds_for_one_iteration} seconds')
                num_iterations = min(num_iterations,int(self.stop_training_after_seconds/seconds_for_one_iteration))

            logging.info(f'Training GBM with {num_iterations} iterations')
            params['num_iterations'] = num_iterations
            bst = lightgbm.train(params, train_data, valid_sets=validate_data)

            self.models[col_name] = bst

    def _predict(self, when_data_source, include_extra_data=False):
        """
        :param when_data_source: DataSource
        :param include_extra_data: bool
        """
        data = None
        for col_name in when_data_source.input_feature_names:
            if data is None:
                data = when_data_source.get_encoded_column_data(col_name)
            else:
                data = torch.cat((torch.Tensor(data).to(self.device),
                                  when_data_source.get_encoded_column_data(col_name).to(self.device)), 1)
        data = data.tolist()

        ypred = {}
        for col_name in when_data_source.output_feature_names:
            col_preds = self.models[col_name].predict(data)
            if col_name in self.ord_encs:
                col_preds = self.ord_encs[col_name].inverse_transform(np.argmax(col_preds, axis=1).reshape(-1, 1)).flatten()
            ypred[col_name] = {'predictions':  list(col_preds)}

        return ypred
