import copy
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from lightwood.mixers.nn.helpers.default_net import DefaultNet
from lightwood.mixers.nn.helpers.transformer import Transformer


class NnMixer:

    def __init__(self, dynamic_parameters):
        self.net = None
        self.optimizer = None
        self.input_column_names = None
        self.output_column_names = None
        self.data_loader = None
        self.transformer = None
        self.encoders = None
        self.optimizer_class = None
        self.optimizer_args = None
        self.criterion = None

        self.batch_size = 200
        self.epochs = 120000

        self.nn_class = DefaultNet
        self.dynamic_parameters = dynamic_parameters

    def fit(self, ds=None, callback=None):

        ret = 0
        for i in self.iter_fit(ds):
            ret = i
        self.encoders = ds.encoders
        return ret

    def predict(self, when_data_source, include_encoded_predictions = False):
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
        inputs = inputs.to(self.net.device)
        labels = labels.to(self.net.device)

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
            predictions[output_column] = {'predictions': decoded_predictions}
            if include_encoded_predictions:
                predictions[output_column]['encoded_predictions'] = output_encoded_vectors[output_column]

        logging.info('Model predictions and decoding completed')

        return predictions

    def error(self, ds):
        """
        :param ds:
        :return:
        """

        ds.encoders = self.encoders
        ds.transformer = self.transformer

        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        running_loss = 0.0
        error = 0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(self.net.device)
            labels = labels.to(self.net.device)

            # If the criterion is CrossEntropyLoss, this happens when weights are present
            if ds.output_weights is not None and ds.output_weights is not False:
                target = labels.numpy()
                target_indexes = np.where(target>0)[1]
                targets_c = torch.LongTensor(target_indexes)
                labels = targets_c.to(self.net.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            error = running_loss / (i + 1)

        return error

    def get_model_copy(self):
        """
        get the actual mixer model
        :return: self.net
        """
        return copy.deepcopy(self.net)

    def update_model(self, model):
        """
        replace the current model with a model object
        :param model: a model object
        :return: None
        """

        self.net = model

    def fit_data_source(self, ds):
        self.input_column_names = self.input_column_names if self.input_column_names is not None else ds.get_feature_names('input_features')
        self.output_column_names = self.output_column_names if self.output_column_names is not None else ds.get_feature_names('output_features')
        ds.transformer = Transformer(self.input_column_names, self.output_column_names)
        self.encoders = ds.encoders
        self.transformer = ds.transformer

    def iter_fit(self, ds):
        """
        :param ds:
        :return:
        """
        self.fit_data_source(ds)
        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.net = self.nn_class(ds, self.dynamic_parameters)
        self.net.train()


        if self.criterion is None:
            if ds.output_weights is not None and ds.output_weights is not False:
                self.criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(ds.output_weights).to(self.net.device))
            else:
                self.criterion = torch.nn.MSELoss()


        base_lr = self.dynamic_parameters['base_lr']
        max_lr = self.dynamic_parameters['max_lr']
        scheduler_mode = self.dynamic_parameters['scheduler_mode'] #triangular, triangular2, exp_range
        weight_decay = self.dynamic_parameters['weight_decay']

        step_size_up=200

        if self.optimizer_class is None:
            self.optimizer_class = torch.optim.AdamW

        if self.optimizer_args is None:
            self.optimizer_args = {}

        self.optimizer_args['amsgrad'] = False
        self.optimizer_args['lr'] = base_lr
        self.optimizer_args['weight_decay'] = weight_decay

        self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

        cycle_momentum = False # Set to "True" if we get optimizers with momentum
        # Note: we can probably the distance between and the values for `base_momentum` and `max_momentum` based on the poportion between base_lr and max_lr (not sure how yet, but it makes some intuitive sense that this could be done), that way we don't have to use fixed values but we don't have to search for the best values... or at least we could reduce the search space and run only a few ax iterations

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr, max_lr, step_size_up=step_size_up, step_size_down=None, mode=scheduler_mode, gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=cycle_momentum, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

        total_epochs = self.epochs

        for epoch in range(total_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            error = 0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                labels = labels.to(self.net.device)
                inputs = inputs.to(self.net.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                # If the criterion is CrossEntropyLoss, this happens when weights are present
                if ds.output_weights is not None and ds.output_weights is not False:
                    target = labels.numpy()
                    target_indexes = np.where(target>0)[1]
                    targets_c = torch.LongTensor(target_indexes)
                    labels = targets_c.to(self.net.device)

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                error = running_loss / (i + 1)

            yield error



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
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                #'encoder_path': 'lightwood.encoders.numeric.numeric'
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
