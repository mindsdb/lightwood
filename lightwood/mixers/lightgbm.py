import numpy as np
import optuna.integration.lightgbm as lgb
import lightgbm
import optuna
import torch
import logging
import time

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices
from lightwood.mixers import BaseMixer
from sklearn.preprocessing import OrdinalEncoder


optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class LightGBMMixer(BaseMixer):
    def __init__(self, stop_training_after_seconds=None, grid_search=False):
        super().__init__()
        self.models = {}
        self.ord_encs = {}
        self.stop_training_after_seconds = stop_training_after_seconds
        self.grid_search = grid_search  # using Optuna

        # GPU Only available via --install-option=--gpu with opencl-dev and libboost dev (a bunch of them) installed, so let's turn this off for now and we can put it behind some flag later
        self.device, _ = get_devices()
        # self.device_str = 'cpu' if str(self.device) == 'cpu' else 'gpu'

        self.device = torch.device('cpu')
        self.device_str = 'cpu'

        self.max_bin = 255  # Default value
        if self.device_str == 'gpu':
            self.max_bin = 63  # As recommended by https://lightgbm.readthedocs.io/en/latest/Parameters.html#device_type

    def _fit(self, train_ds, test_ds=None):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """

        data = {
            'train': {'ds': train_ds, 'data': None, 'label_data': {}},
            'test': {'ds': test_ds, 'data': None, 'label_data': {}}
        }

        # Order is important here
        for subset_name in ['train','test']:
            cols = data[subset_name]['ds'].input_feature_names
            out_cols = data[subset_name]['ds'].output_feature_names
            for col_name in cols:
                if data[subset_name]['data'] is None:
                    data[subset_name]['data'] = data[subset_name]['ds'].get_encoded_column_data(col_name)
                else:
                    enc_col = data[subset_name]['ds'].get_encoded_column_data(col_name)
                    data[subset_name]['data'] = torch.cat((data[subset_name]['data'].to(self.device),
                                                           enc_col.to(self.device)), 1)
            data[subset_name]['data'] = data[subset_name]['data'].tolist()
            for col_name in out_cols:
                label_data = data[subset_name]['ds'].get_column_original_data(col_name)
                if next(item for item in train_ds.output_features if item["name"] == col_name)['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                    if subset_name == 'train':
                        self.ord_encs[col_name] = OrdinalEncoder()
                        self.ord_encs[col_name].fit(np.array(list(set(label_data))).reshape(-1, 1))
                    label_data = self.ord_encs[col_name].transform(np.array(label_data).reshape(-1, 1)).flatten()

                data[subset_name]['label_data'][col_name] = label_data

        for col_name in train_ds.output_feature_names:
            dtype = next(item for item in train_ds.output_features if item["name"] == col_name)['type']
            if dtype not in [COLUMN_DATA_TYPES.NUMERIC, COLUMN_DATA_TYPES.CATEGORICAL]:
                logging.info('cannot support {dtype} in lightgbm'.format(dtype=dtype))
                continue
            else:
                objective = 'regression' if dtype == COLUMN_DATA_TYPES.NUMERIC else 'multiclass'
                metric = 'l2' if dtype == COLUMN_DATA_TYPES.NUMERIC else 'multi_logloss'

            params = {'objective': objective,
                      'metric': metric,
                      'verbose': -1,
                      'lambda_l1': 0.1,
                      'lambda_l2': 0.1,
                      'force_row_wise': True,
                      'device_type': self.device_str
                      }
            if objective == 'multiclass':
                self.all_classes = self.ord_encs[col_name].categories_[0]
                params['num_class'] = self.all_classes.size

            num_iterations = 100

            if self.stop_training_after_seconds is not None:
                train_data = lightgbm.Dataset(data['train']['data'], label=data['train']['label_data'][col_name])
                validate_data = lightgbm.Dataset(data['test']['data'], label=data['test']['label_data'][col_name])
                start = time.time()
                params['num_iterations'] = 1
                bst = lightgbm.train(params, train_data, valid_sets=validate_data, verbose_eval=False)
                end = time.time()
                seconds_for_one_iteration = end - start
                logging.info(f'A single GBM itteration takes {seconds_for_one_iteration} seconds')
                max_itt = int(self.stop_training_after_seconds/seconds_for_one_iteration)
                num_iterations = max(1, min(num_iterations, max_itt))
                # Turn on grid search if training doesn't take too long using it
                if max_itt > 10*num_iterations and seconds_for_one_iteration < 10:
                    self.grid_search = True

            train_data = lightgbm.Dataset(data['train']['data'], label=data['train']['label_data'][col_name])
            validate_data = lightgbm.Dataset(data['test']['data'], label=data['test']['label_data'][col_name])
            model = lgb if self.grid_search else lightgbm
            logging.info(f'Training GBM ({model}) with {num_iterations} iterations')
            params['num_iterations'] = num_iterations
            bst = model.train(params, train_data, valid_sets=validate_data, verbose_eval=False)

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
            ypred[col_name] = {}
            if col_name in self.ord_encs:
                ypred[col_name]['class_distribution'] = list(col_preds)
                ypred[col_name]['class_labels'] = {i: cls for i, cls in enumerate(self.all_classes)}
                col_preds = self.ord_encs[col_name].inverse_transform(np.argmax(col_preds, axis=1).reshape(-1, 1)).flatten()
            ypred[col_name]['predictions'] = list(col_preds)

        return ypred
