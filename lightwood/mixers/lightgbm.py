import numpy as np
import lightgbm
import torch
import logging

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.mixers import BaseMixer


class LightGBMMixer(BaseMixer):
    def __init__(self):
        super().__init__()
        self.models = {}

    def _fit(self, train_ds, test_ds=None):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """

        data = {
            'train': {'ds': train_ds, 'data': None, 'label_data': {}},
            'test': {'ds': test_ds, 'data': None, 'label_data': {}}
        }


        for dtype in data:
            cols = data[dtype]['ds'].input_feature_names
            out_cols = data[dtype]['ds'].output_feature_names
            for col_name in cols:
                if data[dtype]['data'] is None:
                    data[dtype]['data'] = data[dtype]['ds'].get_encoded_column_data(col_name)
                else:
                    data[dtype]['data'] = torch.cat((data[dtype]['data'], data[dtype]['ds'].get_encoded_column_data(col_name)), 1)
            data[dtype]['data'] = data[dtype]['data'].tolist()
            for col_name in out_cols:
                data[dtype]['label_data'][col_name] = data[dtype]['ds'].get_column_original_data(col_name)

        out_cols = train_ds.output_feature_names
        for col_name in out_cols:
            train_data = lightgbm.Dataset(data['train']['data'], label=data['train']['label_data'][col_name])
            validate_data = lightgbm.Dataset(data['test']['data'], label=data['test']['label_data'][col_name])
            dtype = next(item for item in train_ds.output_features if item["name"] == col_name)['type']
            if dtype not in [COLUMN_DATA_TYPES.NUMERIC, COLUMN_DATA_TYPES.CATEGORICAL]:
                logging.info('cannot support {dtype} in lightgbm'.format(dtype=dtype))
                continue
            else:
                objective = 'regression' if dtype == 'numeric' else 'multiclass
            param = {'objective': objective}
            if objective == 'multiclass':
                param['num_class'] = len(set(data['train']['label_data'][col_name]))

            num_round = 10
            bst = lightgbm.train(param, train_data, num_round, valid_sets=validate_data)
            self.models[col_name] = bst


    def _predict(self, when_data_source, include_extra_data=False):
        """
        :param when_data_source: DataSource
        :param include_extra_data: bool
        """


        cols = when_data_source.input_feature_names
        out_cols = when_data_source.output_feature_names
        data = None
        for col_name in cols:
            if data is None:
                data = when_data_source.get_encoded_column_data(col_name)
            else:
                data = torch.cat((data, when_data_source.get_encoded_column_data(col_name)), 1)
        data = data.tolist()


        data = lightgbm.Dataset(data)

        ypred = {col_name: self.models[col_name].predict(data) for col_name in out_cols}

        return ypred
