import pandas as pd
from tsfresh import extract_relevant_features, extract_features
from tsfresh.examples import load_robot_execution_failures
import torch


class TsFreshTsEncoder:

    def __init__(self, is_target=False):
        self._pytorch_wrapper = torch.FloatTensor

    def prepare_encoder(self, priming_data):
        pass

    def encode(self, column_data):
        """
        Encode a column data into time series

        :param column_data: a list of timeseries data eg: ['91.0 92.0 93.0 94.0', '92.0 93.0 94.0 95.0' ...]
        :return: a torch.floatTensor
        """

        ret = []
        for i, values in enumerate(column_data):
            if type(values) == type([]):
                values = list(map(float,values))
            else:
                values = list(map(lambda x: float(x), values.split()))

            df = pd.DataFrame({'main_feature': values, 'id': [1] * len(values)})

            features = extract_features(df, column_id='id',disable_progressbar=True)
            features.fillna(value=0, inplace=True)

            features = list(features.iloc[0])

            ret.append(features)

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')


# only run the test if this file is called from debugger, it takes <1 min
if __name__ == "__main__":
    import math

    data = [" ".join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

    ret = TsFreshTsEncoder().encode(data)

    print(ret)
    print(f'Got above vecotr of lenght: {len(ret)} and feature lenght: {len(ret[0])} for that of length {len(data)} and member length {len(data[0])}')
    assert(len(ret) == len(data))
    assert(len(ret) < 800)
