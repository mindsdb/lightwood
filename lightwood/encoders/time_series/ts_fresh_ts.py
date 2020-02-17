from tsfresh import extract_relevant_features
import torch


class TsFreshTsEncoder:

    def __init__(self, is_target=False):
        self._pytorch_wrapper = torch.FloatTensor

    def prepare_encoder(self, priming_data):
        pass

    def encode(self, column_data):
        """
        Encode a column data into time series

        :param values_data: data frame with all columns including target
        :param target_name: target column name
        :param group_by: group_by column name
        :param order_by: order_by column name
        :return: a torch.floatTensor
        """

        ret = []
        for i, values in enumerate(column_data):
            if type(values) == type([]):
                values = list(map(float,values))
            else:
                values = list(map(lambda x: float(x), values.split()))

            features = extract_relevant_features(values)
            features.fillna(value=0, inplace=True)
            print(features)
            ret.append(features)

        #y = values_data.groupby(group_by).first()[target_name]
        #values_data.drop(columns=[target_name], inplace=True)

        #features = extract_relevant_features(values_data, y=y,  column_id=group_by, column_sort=order_by)



        for row in features.iterrows():
            ret.append(list(row[1]))

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')


# only run the test if this file is called from debugger, it takes <1 min
if __name__ == "__main__":
    import math

    data = [" ".join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

    ret = TsFreshTsEncoder().encode(data)

    print(ret)
