from tsfresh import extract_relevant_features
import torch


class TsFreshTsEncoder:

    def __init__(self):
        self._pytorch_wrapper = torch.FloatTensor

    def encode(self, values_data, target_name, order_by, group_by):
        """
        Encode a column data into time series

        :param values_data: data frame with all columns including target
        :param target_name: target column name
        :param group_by: group_by column name
        :param order_by: order_by column name
        :return: a torch.floatTensor
        """
        y = values_data.groupby(group_by).first()[target_name]
        values_data.drop(columns=[target_name], inplace=True)

        features = extract_relevant_features(values_data, y=y,
                                             column_id=group_by, column_sort=order_by)

        features.fillna(value=0, inplace=True)

        ret = []
        for row in features.iterrows():
            ret.append(list(row[1]))

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')


# only run the test if this file is called from debugger, it takes <1 min
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("test_data/train.csv")

    ret = TsFreshTsEncoder().encode(df, "target", "time", "id")

    print(ret)
