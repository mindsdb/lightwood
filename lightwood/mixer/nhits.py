from typing import Dict, Union

import numpy as np
import pandas as pd
from hyperopt import hp
import neuralforecast as nf

from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class NHitsMixer(BaseMixer):
    horizon: int
    target: str
    supports_proba: bool
    model_path: str
    hyperparam_search: bool

    default_config: dict = {
        'model': 'n-hits',
        'mode': 'simple',
        'activation': 'SELU',

        'stack_types': ['identity', 'identity', 'identity'],
        'constant_n_blocks': 1,
        'constant_n_layers': 2,
        'constant_n_mlp_units': 256,
        'n_pool_kernel_size': [4, 2, 1],
        'n_freq_downsample': [24, 12, 1],
        'pooling_mode': 'max',
        'interpolation_mode': 'linear',
        'shared_weights': False,

        # Optimization and regularization parameters
        'initialization': 'lecun_normal',
        'learning_rate': 0.001,
        'batch_size': 1,
        'n_windows': 32,
        'lr_decay': 0.5,
        'lr_decay_step_size': 2,
        'max_epochs': 1,
        'max_steps': None,
        'early_stop_patience': 20,
        'eval_freq': 500,
        'batch_normalization': False,
        'dropout_prob_theta': 0.0,
        'dropout_prob_exogenous': 0.0,
        'weight_decay': 0,
        'loss_train': 'MAE',
        'loss_hypar': 0.5,
        'loss_valid': 'MAE',
        'random_seed': 1,

        # Data Parameters
        'idx_to_sample_freq': 1,
        'val_idx_to_sample_freq': 1,
        'n_val_weeks': 52,
        'normalizer_y': None,
        'normalizer_x': 'median',
        'complete_windows': False,
        'frequency': 'H',
    }

    def __init__(
            self,
            stop_after: float,
            target: str,
            horizon: int,
            ts_analysis: Dict,
    ):
        """
        Mixer description here.
        
        :param stop_after: time budget in seconds.
        :param target: column to forecast.
        :param horizon: length of forecasted horizon.
        :param ts_analysis: dictionary with miscellaneous time series info, as generated by 'lightwood.data.timeseries_analyzer'.
        """  # noqa
        super().__init__(stop_after)
        self.stable = True
        self.prepared = False
        self.supports_proba = False
        self.target = target
        self.config = NHitsMixer.default_config.copy()

        self.ts_analysis = ts_analysis
        self.horizon = horizon
        self.grouped_by = ['__default'] if not ts_analysis['tss'].group_by else ts_analysis['tss'].group_by
        self.model = None

        self.config['n_time_in'] = self.ts_analysis['tss'].window
        self.config['n_time_out'] = self.horizon
        self.config['n_x_hidden'] = 0  # 8
        self.config['n_s_hidden'] = 0

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the N-HITS model.
        """  # noqa
        log.info('Started fitting N-HITS forecasting model')

        cat_ds = ConcatedEncodedDs([train_data, dev_data])
        oby_col = self.ts_analysis["tss"].order_by[0]
        df = cat_ds.data_frame.sort_values(by=f'__mdb_original_{oby_col}')

        # 2. adapt data into the expected DFs
        Y_df = self._make_initial_df(df)

        # set val-test cutoff
        n_time = len(df[f'__mdb_original_{oby_col}'].unique())
        n_ts_val = int(.1 * n_time)
        n_ts_test = int(.1 * n_time)

        # 3. TODO: merge user-defined config into default

        # train the model
        n_time_out = self.horizon
        self.model = nf.auto.NHITS(horizon=n_time_out)
        self.model.space['max_steps'] = hp.choice('max_steps', [1000])
        self.model.fit(Y_df=Y_df,
                       X_df=None,  # Exogenous variables
                       S_df=None,  # Static variables
                       hyperopt_steps=5,
                       n_ts_val=n_ts_val,
                       n_ts_test=n_ts_test,
                       results_dir='./results/autonhits',  # TODO: change this
                       save_trials=True,
                       loss_function_val=nf.losses.numpy.mae,
                       loss_functions_test={'mae': nf.losses.numpy.mae,
                                            'mse': nf.losses.numpy.mse},
                       return_test_forecast=True,  # False
                       verbose=False)

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Due to how lightwood implements the `update` procedure, expected inputs for this method are:
        
        :param dev_data: original `test` split (used to validate and select model if ensemble is `BestOf`).
        :param train_data: concatenated original `train` and `dev` splits.
        """  # noqa
        self.hyperparam_search = False
        self.fit(dev_data, train_data)
        self.prepared = True

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Calls the mixer to emit forecasts.
        """  # noqa
        if args.predict_proba:
            log.warning('This mixer does not output probability estimates')

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=['prediction'],
                           dtype=object)

        input_df = self._make_initial_df(ds.data_frame)  # TODO make it so that it's horizon worth of data in each row
        for i in range(input_df.shape[0]):
            ydf.iloc[i]['prediction'] = self.model.forecast(input_df.iloc[i:i + 1])['y'].tolist()

        return ydf[['prediction']]

    def _make_initial_df(self, df):
        oby_col = self.ts_analysis["tss"].order_by[0]
        Y_df = pd.DataFrame()
        Y_df['y'] = df[self.target]
        Y_df['ds'] = pd.to_datetime(df[f'__mdb_original_{oby_col}'], unit='s')
        Y_df['unique_id'] = df[self.grouped_by].apply(lambda x: ','.join([elt for elt in x]), axis=1)
        return Y_df