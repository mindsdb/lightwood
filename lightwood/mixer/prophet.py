from typing import Dict, Union

import pandas as pd
from prophet import Prophet
from datetime import datetime

from lightwood.mixer.base import BaseMixer
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class ProphetMixer(BaseMixer):
    def __init__(self,
                 stop_after: float,
                 target: str,
                 dtype_dict: Dict[str, str],
                 horizon: int,
                 ts_analysis: Dict,
                 ):
        super().__init__(stop_after)
        self.stable = False
        self.target = target
        self.dtype_dict = dtype_dict
        self.horizon = horizon
        self.ts_analysis = ts_analysis
        self.model = Prophet()

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        print("Will try to fit now")
        # TODO: add groups support
        data = ConcatedEncodedDs([train_data, dev_data])
        df = self._preprocess_df(data.data_frame[[f'__mdb_original_{self.ts_analysis["tss"].order_by[0]}',
                                                  data.target]])
        self.model.fit(df)
        self.prepared = True

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs], args: PredictionArguments = PredictionArguments()) \
            -> pd.DataFrame:
        print("Will try to predict now")
        # df = self._preprocess_df(ds.data_frame[[f'__mdb_original_{self.ts_analysis["tss"].order_by[0]}', ds.target]])
        df = self.model.make_future_dataframe(periods=self.horizon, include_history=False)
        fcst = self.model.predict(df)
        out = pd.DataFrame(columns=['prediction'])
        out['prediction'] = fcst[['yhat']].values.reshape(1, -1).tolist()
        return out

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = ['ds', 'y']
        df['ds'] = df['ds'].apply(datetime.utcfromtimestamp)
        return df
