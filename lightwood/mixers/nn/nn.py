import copy
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import gc
import operator

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.config.config import CONFIG


class NnMixer:

    def __init__(self, dynamic_parameters, is_categorical_output=False):
        self.is_categorical_output = is_categorical_output
        self.net = None
        self.optimizer = None
        self.input_column_names = None
        self.output_column_names = None
        self.transformer = None
        self.encoders = None
        self.optimizer_class = None
        self.optimizer_args = None
        self.criterion = None

        self.batch_size = 200
        self.epochs = 120000

        self.nn_class = DefaultNet
        self.dynamic_parameters = dynamic_parameters
        self.awareness_criterion = None
        self.loss_combination_operator = operator.add

        self._nonpersistent = {
            'sampler': None
        }

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
        data_loader = DataLoader(when_data_source, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # set model into evaluation mode in order to skip things such as Dropout
        self.net = self.net.eval()

        outputs = []
        awareness_arr = []
        for i, data in enumerate(data_loader, 0):
            inputs, _ = data
            inputs = inputs.to(self.net.device)

            with torch.no_grad():
                if CONFIG.SELFAWARE:
                    output, awareness = self.net(inputs)
                    awareness = awareness.to('cpu')
                    awareness_arr.extend(awareness)
                else:
                    output = self.net(inputs)
                    awareness_arr = None

                output = output.to('cpu')

            outputs.extend(output)

        output_trasnformed_vectors = {}
        confidence_trasnformed_vectors = {}

        for i in range(len(outputs)):
            if awareness_arr is not None:
                confidence_vector = awareness_arr[i]
                transformed_confidence_vectors = when_data_source.transformer.revert(confidence_vector,feature_set = 'output_features')
                for feature in transformed_confidence_vectors:
                    if feature not in confidence_trasnformed_vectors:
                        confidence_trasnformed_vectors[feature] = []
                    # @TODO: Very simple algorithm to get a confidence from the awareness, not necessarily what we want for the final version
                    confidence_trasnformed_vectors[feature] += [1 - sum(np.abs(transformed_confidence_vectors[feature]))/len(transformed_confidence_vectors[feature])]

            output_vector = outputs[i]
            transformed_output_vectors = when_data_source.transformer.revert(output_vector,feature_set = 'output_features')
            for feature in transformed_output_vectors:
                if feature not in output_trasnformed_vectors:
                    output_trasnformed_vectors[feature] = []
                output_trasnformed_vectors[feature] += [transformed_output_vectors[feature]]

        predictions = {}
        for output_column in output_trasnformed_vectors:
            decoded_predictions = when_data_source.get_decoded_column_data(output_column, when_data_source.encoders[output_column]._pytorch_wrapper(output_trasnformed_vectors[output_column]))
            predictions[output_column] = {'predictions': decoded_predictions}
            if awareness_arr is not None:
                predictions[output_column]['confidences'] = confidence_trasnformed_vectors[output_column]

            if include_encoded_predictions:
                predictions[output_column]['encoded_predictions'] = output_trasnformed_vectors[output_column]

        logging.info('Model predictions and decoding completed')

        return predictions

    def overall_certainty(self):
        """
        return an estimate of how certain is the model about the overall predictions,
        in this case its a measurement of how much did the variance of all the weights distributions reduced from its initial distribution
        :return:
        """
        if hasattr(self.net, 'calculate_overall_certainty'):
            return self.net.calculate_overall_certainty()
        else:
            return -1

    def error(self, ds):
        """
        :param ds:
        :return:
        """
        self.net = self.net.eval()

        ds.encoders = self.encoders
        ds.transformer = self.transformer


        if self._nonpersistent['sampler'] is None:
            data_loader = DataLoader(ds, batch_size=self.batch_size, sampler=self._nonpersistent['sampler'], num_workers=0)
        else:
            data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        running_loss = 0.0
        error = 0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(self.net.device)
            labels = labels.to(self.net.device)

            if self.is_categorical_output:
                target = labels.cpu().numpy()
                target_indexes = np.where(target>0)[1]
                targets_c = torch.LongTensor(target_indexes)
                labels = targets_c.to(self.net.device)

            with torch.no_grad():
                if CONFIG.SELFAWARE:
                    outputs, awareness = self.net(inputs)
                else:
                    outputs = self.net(inputs)

            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            error = running_loss / (i + 1)
        self.net = self.net.train()

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

        transformer_already_initialized = False
        try:
            if len(list(ds.transformer.feature_len_map.keys())) > 0:
                transformer_already_initialized = True
        except:
            pass

        if not transformer_already_initialized:
            ds.transformer = Transformer(self.input_column_names, self.output_column_names)

        self.encoders = ds.encoders
        self.transformer = ds.transformer

    def iter_fit(self, ds):
        """
        :param ds:
        :return:
        """
        self.fit_data_source(ds)
        if self.is_categorical_output:
            # The WeightedRandomSampler samples "randomly" but can assign higher weight to certain rows, we assign each rows it's weight based on the target variable value in that row and it's associated weight in the output_weights map (otherwise used to bias the loss function)
            if ds.output_weights is not None and ds.output_weights is not False and CONFIG.OVERSAMPLE:
                weights = []
                for row in ds:
                    _, out = row
                    # @Note: This assumes one-hot encoding for the encoded_value
                    weights.append(ds.output_weights[torch.argmax(out).item()])

                self._nonpersistent['sampler'] = torch.utils.data.WeightedRandomSampler(weights=weights,num_samples=len(weights),replacement=True)

        self.net = self.nn_class(ds, self.dynamic_parameters)
        self.net = self.net.train()

        if self.batch_size < self.net.available_devices:
            self.batch_size = self.net.available_devices

        self.awareness_criterion = torch.nn.MSELoss()

        if self.criterion is None:
            if self.is_categorical_output:
                if ds.output_weights is not None and ds.output_weights is not False and not CONFIG.OVERSAMPLE:
                    output_weights = torch.Tensor(ds.output_weights).to(self.net.device)
                else:
                    output_weights = None
                self.criterion = torch.nn.CrossEntropyLoss(weight=output_weights)
            else:
                self.criterion = torch.nn.MSELoss()

        self.optimizer_class = Ranger
        if self.optimizer_args is None:
            self.optimizer_args = {}

        if 'beta1' in self.dynamic_parameters:
            self.optimizer_args['betas'] = (self.dynamic_parameters['beta1'],0.999)

        for optimizer_arg_name in ['lr','k','N_sma_threshold']:
            if optimizer_arg_name in self.dynamic_parameters:
                self.optimizer_args[optimizer_arg_name] = self.dynamic_parameters[optimizer_arg_name]

        self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)
        total_epochs = self.epochs


        if self._nonpersistent['sampler'] is None:
            data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        else:
            data_loader = DataLoader(ds, batch_size=self.batch_size, num_workers=0, sampler=self._nonpersistent['sampler'])

        total_iterations = 0
        for epoch in range(total_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            error = 0
            for i, data in enumerate(data_loader, 0):
                total_iterations += 1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                labels = labels.to(self.net.device)
                inputs = inputs.to(self.net.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # outputs = self.net(inputs)
                if CONFIG.SELFAWARE:
                    outputs, awareness = self.net(inputs)
                else:
                    outputs = self.net(inputs)

                if self.is_categorical_output:
                    target = labels.cpu().numpy()
                    target_indexes = np.where(target>0)[1]
                    targets_c = torch.LongTensor(target_indexes)
                    cat_labels = targets_c.to(self.net.device)
                    loss = self.criterion(outputs, cat_labels)
                else:
                    loss = self.criterion(outputs, labels)

                if CONFIG.SELFAWARE:
                    real_loss = torch.abs(labels - outputs) # error precentual to the target
                    real_loss = torch.Tensor(real_loss.tolist()) # disconnect from the graph (test if this is necessary)
                    real_loss = real_loss.to(self.net.device)

                    awareness_loss = self.awareness_criterion(awareness, real_loss)

                    #print(awareness_loss.item())
                    #print(loss.item())

                    total_loss = self.loss_combination_operator(awareness_loss, loss)
                    running_loss += total_loss.item()

                    # Make sure the LR doesn't get too low
                    if self.optimizer.lr > 5 * pow(10,-6):
                        if np.isnan(running_loss) or np.isinf(running_loss) or running_loss > pow(10,4):
                            self.optimizer_args['lr'] = self.optimizer.lr/2
                            gc.collect()
                            if 'cuda' in str(self.net.device):
                                torch.cuda.empty_cache()

                            self.loss_combination_operator = operator.add
                            self.net = self.nn_class(ds, self.dynamic_parameters)
                            self.optimizer.zero_grad()
                            self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

                            break
                else:
                    total_loss = loss

                total_loss.backward()
                self.optimizer.step()
                # now that we have run backward in both losses, optimize() (review: we may need to optimize for each step)

                error = running_loss / (i + 1)


                if error < 1:
                    if self.loss_combination_operator == operator.add:
                        self.loss_combination_operator = operator.mul

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
    ds.prepare_encoders()
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    predict_input_ds.prepare_encoders()
    ####################

    mixer = NnMixer({})

    for i in  mixer.iter_fit(ds):
        if i < 0.01:
            break

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
    ds.prepare_encoders()
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    predict_input_ds.prepare_encoders()
    ####################

    mixer = NnMixer({})

    for i in  mixer.iter_fit(ds):
        if i < 0.01:
            break

    predictions = mixer.predict(predict_input_ds)
    print(predictions)
