import torch


class Transformer:

    def __init__(self, input_features, output_features):

        self.input_features = input_features
        self.output_features = output_features

        self.feature_len_map = {}
        self.out_indexes = []

    def transform(self, sample):

        input_vector = []
        output_vector = []

        for input_feature in self.input_features:
            sub_vector = sample['input_features'][input_feature].tolist()
            input_vector += sub_vector
            if input_feature not in self.feature_len_map:
                self.feature_len_map[input_feature] = len(sub_vector)

        for output_feature in self.output_features:
            sub_vector = sample['output_features'][output_feature].tolist()
            output_vector += sub_vector
            if output_feature not in self.feature_len_map:
                self.feature_len_map[output_feature] = len(sub_vector)

            if len(self.out_indexes) < len(sample['output_features']):
                if len(self.out_indexes) == 0:
                    self.out_indexes.append([0,len(sub_vector)])
                else:
                    self.out_indexes.append([self.out_indexes[-1][1], self.out_indexes[-1][1] + len(sub_vector)])

        return torch.FloatTensor(input_vector), torch.FloatTensor(output_vector)

    def revert(self, vector, feature_set='output_features'):
        start = 0
        ret = {}
        list_vector = vector.tolist()
        for feature_name in getattr(self, feature_set):
            top = start + self.feature_len_map[feature_name]
            ret[feature_name] = list_vector[start:top]
            start = top
        return ret
