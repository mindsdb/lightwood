import pandas as pd
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.api import dtype
from typing import Dict, List, Set
import numpy as np
import optuna.integration.lightgbm as optuna_lightgbm
import lightgbm
import optuna
import torch
import time
from lightwood.helpers.log import log
from lightwood.helpers.device import get_devices
from sklearn.preprocessing import OrdinalEncoder
from lightwood.model.base import BaseModel

optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def check_gpu_support():
    try:
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        train_data = lightgbm.Dataset(data, label=label)
        params = {'num_iterations': 1, 'device': 'gpu'}
        lightgbm.train(params, train_set=train_data)
        return True
    except Exception:
        return False


class LightGBM(BaseModel):
    model: lightgbm.LGBMModel
    ordinal_encoder: OrdinalEncoder
    label_set: Set[str]
    max_bin: int
    device: torch.device
    device_str: str
    num_iterations: int
    use_optuna: bool

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str], fit_on_dev: bool, use_optuna: bool = True):
        super().__init__(stop_after)
        self.model = None
        self.ordinal_encoder = None
        self.label_set = set()
        self.target = target
        self.dtype_dict = dtype_dict
        self.input_cols = input_cols
        self.use_optuna = use_optuna
        self.params = {}
        self.fit_on_dev = fit_on_dev

        # GPU Only available via --install-option=--gpu with opencl-dev and libboost dev (a bunch of them) installed, so let's turn this off for now and we can put it behind some flag later
        gpu_works = check_gpu_support()
        if not gpu_works:
            self.device = torch.device('cpu')
            self.device_str = 'cpu'
            log.warning('LightGBM running on CPU, this somewhat slower than the GPU version, consider using a GPU instead')
        else:
            self.device = torch.device('cuda')
            self.device_str = 'gpu'

        self.max_bin = 255
    
    def _to_dataset(self, data, output_dtype):
        for subset_name in data.keys():
            for input_col in self.input_cols:
                if data[subset_name]['data'] is None:
                    data[subset_name]['data'] = data[subset_name]['ds'].get_encoded_column_data(input_col).to(self.device)
                else:
                    enc_col = data[subset_name]['ds'].get_encoded_column_data(input_col)
                    data[subset_name]['data'] = torch.cat((data[subset_name]['data'], enc_col.to(self.device)), 1)
            data[subset_name]['data'] = data[subset_name]['data'].tolist()

            label_data = data[subset_name]['ds'].get_column_original_data(self.target)

            if output_dtype in (dtype.categorical, dtype.binary):
                if subset_name == 'train':
                    self.ordinal_encoder = OrdinalEncoder()
                    self.label_set = set(label_data)
                    self.label_set.add('__mdb_unknown_cat')
                    self.ordinal_encoder.fit(np.array(list(self.label_set)).reshape(-1, 1))

                label_data = [x if x in self.label_set else '__mdb_unknown_cat' for x in label_data]
                label_data = self.ordinal_encoder.transform(np.array(label_data).reshape(-1, 1)).flatten()
            elif output_dtype == dtype.integer:
                label_data = label_data.astype(int)
            elif output_dtype == dtype.float:
                label_data = label_data.astype(float)

            data[subset_name]['label_data'] = label_data

        return data

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting LGBM model')
        train_ds_arr = ds_arr[0:int(len(ds_arr) * 0.9)]
        dev_ds_arr = ds_arr[int(len(ds_arr) * 0.9):]
        data = {
            'train': {'ds': ConcatedEncodedDs(train_ds_arr), 'data': None, 'label_data': {}},
            'dev': {'ds': ConcatedEncodedDs(dev_ds_arr), 'data': None, 'label_data': {}}
        }
        self.fit_data_len = len(data['train']['ds'])

        output_dtype = self.dtype_dict[self.target]

        data = self._to_dataset(data, output_dtype)

        if output_dtype not in (dtype.categorical, dtype.integer, dtype.float, dtype.binary):
            log.error(f'Lightgbm mixer not supported for type: {output_dtype}')
            raise Exception(f'Lightgbm mixer not supported for type: {output_dtype}')
        else:
            objective = 'regression' if output_dtype in (dtype.integer, dtype.float) else 'multiclass'
            metric = 'l2' if output_dtype in (dtype.integer, dtype.float) else 'multi_logloss'

        self.params = {
            'objective': objective,
            'metric': metric,
            'verbose': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'force_row_wise': True,
            'device_type': self.device_str,
        }

        if objective == 'multiclass':
            self.all_classes = self.ordinal_encoder.categories_[0]
            self.params['num_class'] = self.all_classes.size
        if self.device_str == 'gpu':
            self.params['gpu_use_dp'] = True

        # Determine time per iterations
        start = time.time()
        self.params['num_iterations'] = 1
        self.model = lightgbm.train(self.params, lightgbm.Dataset(data['train']['data'], label=data['train']['label_data']), verbose_eval=False)
        end = time.time()
        seconds_for_one_iteration = max(0.1, end - start)

        # Determine nr of iterations
        log.info(f'A single GBM iteration takes {seconds_for_one_iteration} seconds')
        self.num_iterations = int(self.stop_after * 0.8 / seconds_for_one_iteration)


        # Turn on grid search if training doesn't take too long using it
        kwargs = {}
        if self.use_optuna and self.num_iterations >= 200:
            model_generator = optuna_lightgbm
            kwargs['time_budget'] = self.stop_after * 0.4
            self.num_iterations = int(self.num_iterations / 2)
            kwargs['optuna_seed'] = 0
        else:
            model_generator = lightgbm

        # Prepare the data
        train_dataset = lightgbm.Dataset(data['train']['data'], label=data['train']['label_data'])
        dev_dataset = lightgbm.Dataset(data['dev']['data'], label=data['dev']['label_data'])

        # Train the models
        log.info(f'Training GBM ({model_generator}) with {self.num_iterations} iterations given {self.stop_after} seconds constraint')
        self.params['num_iterations'] = int(self.num_iterations)

        self.params['early_stopping_rounds'] = 5

        self.model = model_generator.train(self.params, train_dataset, valid_sets=[dev_dataset, train_dataset], valid_names=['dev', 'train'], verbose_eval=False, **kwargs)
        self.num_iterations = self.model.best_iteration
        log.info(f'Lightgbm model contains {self.model.num_trees()} weak estimators')

        if self.fit_on_dev:
            self.partial_fit(dev_ds_arr, train_ds_arr)

    def partial_fit(self, train_data: List[EncodedDs], dev_data: List[EncodedDs]) -> None:
        ds = ConcatedEncodedDs(train_data)
        pct_of_original = len(ds) / self.fit_data_len
        iterations = max(1, int(self.num_iterations * pct_of_original)/2)

        data = {'retrain': {'ds': ds, 'data': None, 'label_data': {}}, 'dev': {'ds': ConcatedEncodedDs(dev_data), 'data': None, 'label_data': {}}}

        output_dtype = self.dtype_dict[self.target]
        data = self._to_dataset(data, output_dtype)
        
        train_dataset = lightgbm.Dataset(data['retrain']['data'], label=data['retrain']['label_data'])
        dev_dataset = lightgbm.Dataset(data['dev']['data'], label=data['dev']['label_data'])

        log.info(f'Updating lightgbm model with {iterations} iterations')
        self.params['num_iterations'] = int(iterations)
        self.model = lightgbm.train(self.params, train_dataset, valid_sets=[dev_dataset, train_dataset], valid_names=['dev', 'retrain'], verbose_eval=False, init_model=self.model)
        log.info(f'Model now has a total of {self.model.num_trees()} weak estimators')
        
        pass

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        data = None
        for input_col in self.input_cols:
            if data is None:
                data = ds.get_encoded_column_data(input_col).to(self.device)
            else:
                data = torch.cat((data, ds.get_encoded_column_data(input_col).to(self.device)), 1)

        data = data.tolist()
        raw_predictions = self.model.predict(data)

        if self.ordinal_encoder is not None:
            decoded_predictions = self.ordinal_encoder.inverse_transform(np.argmax(raw_predictions, axis=1).reshape(-1, 1)).flatten()
        else:
            decoded_predictions = raw_predictions

        ydf = pd.DataFrame({'prediction': decoded_predictions})
        return ydf
