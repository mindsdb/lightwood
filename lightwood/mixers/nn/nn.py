
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import math
import copy
import logging
import numpy as np

from lightwood.mixers.nn.helpers.default_net import DefaultNet
from lightwood.mixers.nn.helpers.transformer import Transformer


class NnMixer:

    def __init__(self, input_column_names=None, output_column_names=None):
        self.net = None
        self.optimizer = None
        self.input_column_names = None
        self.output_column_names = None
        self.data_loader = None
        self.transformer = None
        self.encoders = None


        self.criterion = nn.MSELoss()#MSELoss()#CrossEntropyLoss()
        self.epochs = 120000
        self.eval_every = 0.03
        self.optimizer_class = optim.Adadelta
        self.optimizer_args = {'lr': 0.01}
        self.nn_class = DefaultNet
        self.batch_size = 100


        pass

    def fit(self, ds= None, callback=None):

        ret = 0
        for i in self.iter_fit(ds):
            ret = i
        self.encoders = ds.encoders
        return ret

    def predict(self, when_data_source):
        """

        :param when_data_source:
        :return:
        """
        when_data_source.transformer = self.transformer
        when_data_source.encoders = self.encoders
        data_loader = DataLoader(when_data_source, batch_size=len(when_data_source), shuffle=False, num_workers=0)

        self.net.eval()
        data = next(iter(data_loader))
        inputs, labels = data
        outputs = self.net(inputs)
        output_encoded_vectors = {}

        for output_vector in outputs:
            output_vectors = when_data_source.transformer.revert(output_vector,feature_set = 'output_features')
            for feature in output_vectors:
                if feature not in output_encoded_vectors:
                    output_encoded_vectors[feature] = []
                output_encoded_vectors[feature] += [output_vectors[feature]]



        predictions = dict()

        for output_column in output_encoded_vectors:

            decoded_predictions = when_data_source.get_decoded_column_data(output_column, when_data_source.encoders[output_column]._pytorch_wrapper(output_encoded_vectors[output_column]),  cache=False)
            predictions[output_column] = {'Encoded Predictions': output_encoded_vectors[output_column],
                                          'Actual Predictions': decoded_predictions}

        logging.info('Model predictions and decoding completed')

        return predictions

    def error(self, ds):
        """

        :param ds:
        :return:
        """
        ds.transformer = self.transformer
        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        running_loss = 0.0
        error = 0

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # print statistics
            running_loss += loss.item()
            error = running_loss / (i + 1)

        return error

    def iter_fit(self, ds, test_ds = None):
        """

        :param ds:
        :return:
        """

        self.input_column_names = self.input_column_names if self.input_column_names is not None else ds.get_feature_names(
            'input_features')
        self.output_column_names = self.output_column_names if self.output_column_names is not None else ds.get_feature_names(
            'output_features')
        self.transformer = Transformer(self.input_column_names, self.output_column_names)
        self.encoders = ds.encoders

        if test_ds is None:

            test_ds = ds.extractRandomSubset(0.1)




        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)



        ds.transformer = self.transformer
        self.net = self.nn_class(ds)
        self.net.train()
        self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)


        total_epochs = self.epochs
        epoch_eval_jump = int(total_epochs*self.eval_every)
        eval_next_on_epoch = epoch_eval_jump

        error_delta_buffer = [] # this is a buffer of the delta of test and train error
        delta_mean = 0
        last_test_error = None
        lowest_error = None
        last_good_model = None

        for epoch in range(total_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            error = 0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                error = running_loss / (i + 1)

            yield error

            if epoch >= eval_next_on_epoch and test_ds:

                tmp_next = eval_next_on_epoch + epoch_eval_jump
                eval_next_on_epoch = tmp_next if tmp_next < total_epochs else total_epochs-1

                test_error = self.error(test_ds)
                if lowest_error is None:
                    lowest_error = test_error
                    is_lowest_error = True

                else:
                    if test_error < lowest_error:
                        lowest_error = test_error
                        is_lowest_error = True
                    else:
                        is_lowest_error = False


                if last_test_error is None:
                    last_test_error = test_error

                if is_lowest_error:
                    last_good_model = copy.deepcopy(self.net)


                delta_error = last_test_error - test_error
                last_test_error = test_error

                error_delta_buffer += [delta_error]
                error_delta_buffer = error_delta_buffer[-10:]
                delta_mean = np.mean(error_delta_buffer)
                logging.debug('Delta of test error {delta}'.format(delta=delta_mean))

                if delta_mean < 0:
                    break

        if last_good_model is not None:
            print('restoring last good model')
            self.net = last_good_model






if __name__ == "__main__":
    import random
    import pandas
    from lightwood.api.data_source import DataSource

    ###############
    # GENERATE DATA
    ###############

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            },
            {
                'name': 'y',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'categorical',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    ##For Classification
    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = ['low' if i < 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)

    # print(data_frame)

    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = NnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(predict_input_ds)
    print(predictions)

    ##For Regression

    # GENERATE DATA
    ###############

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            },
            {
                'name': 'y',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = [i + 0.5 for i in range(10)]

    data_frame = pandas.DataFrame(data)
    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = NnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    for i in  mixer.iter_fit(ds):
        print(i)

    predictions = mixer.predict(predict_input_ds)
    print(predictions)