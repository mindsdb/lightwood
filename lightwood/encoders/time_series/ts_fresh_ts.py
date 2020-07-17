import pandas as pd
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
import torch

from lightwood.encoders.numeric.numeric import NumericEncoder
from lightwood.encoders.encoder_base import BaseEncoder


class TsFreshTsEncoder(BaseEncoder):

    def __init__(self, is_target=False):
        super().__init__(is_target)
        self.numerical_encoder = NumericEncoder()
        self.max_series_len = 0
        self.n_jobs = 6

    def prepare_encoder(self, priming_data):
        all_numbers = []

        for i, values in enumerate(priming_data):
            if values is None:
                values = [0]
            elif type(values) == type([]):
                values = list(map(float,values))
            else:
                values = list(map(lambda x: float(x), values.split(' ')))

            self.max_series_len = max(self.max_series_len,len(values))
            all_numbers.extend(values)

        self.numerical_encoder.prepare_encoder(all_numbers)

    def encode(self, column_data):
        """
        Encode a column data into time series

        :param column_data: a list of timeseries data eg: ['91.0 92.0 93.0 94.0', '92.0 93.0 94.0 95.0' ...]
        :return: a torch.floatTensor
        """

        ret = []
        default_fc_parameters=MinimalFCParameters()
        all_values = []


        for i, values in enumerate(column_data):
            if values is None:
                values = [0] * self.max_series_len
            elif type(values) == type([]):
                values = list(map(float,values))
            else:
                values = list(map(lambda x: float(x), values.split(' ')))

            all_values.append(values)
            df = pd.DataFrame({'main_feature': values, 'id': [1] * len(values)})

            try:
                features = extract_features(df, column_id='id',disable_progressbar=True, default_fc_parameters=default_fc_parameters,n_jobs=self.n_jobs)
            except:
                self.n_jobs = 1
                features = extract_features(df, column_id='id',disable_progressbar=True, default_fc_parameters=default_fc_parameters,n_jobs=self.n_jobs)

            features.fillna(value=0, inplace=True)

            features = list(features.iloc[0])
            ret.append(features)

        for i, values in  enumerate(all_values):
            while len(values) < self.max_series_len:
                values.append(0)

            encoded_values = self.numerical_encoder.encode(values)

            encoded_numbers_list = []
            for pair in encoded_values.tolist():
                encoded_numbers_list.extend(pair)

            ret[i].extend(encoded_numbers_list)

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')


# only run the test if this file is called from debugger, it takes <1 min
if __name__ == "__main__":
    import math

    data = [" ".join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

    enc = TsFreshTsEncoder()
    enc.prepare_encoder(data)
    ret = enc.encode(data)

    print(ret)
    print(f'Got above vecotr of lenght: {len(ret)} and feature lenght: {len(ret[0])} for that of length {len(data)} and member length {len(data[0])}')
    assert(len(ret) == len(data))
    assert(len(ret) < 60)
