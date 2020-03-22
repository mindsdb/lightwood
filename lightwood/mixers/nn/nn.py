import copy
import logging
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
import gc
import operator

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.mixers.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES


class NnMixer:
    def __init__(self, dynamic_parameters):
        self.output_types = None
        self.net = None
        self.optimizer = None
        self.input_column_names = None
        self.output_column_names = None
        self.transformer = None
        self.encoders = None
        self.optimizer_class = None
        self.optimizer_args = None
        self.criterion_arr = None
        self.unreduced_criterion_arr = None

        self.batch_size = 200
        self.epochs = 120000

        self.nn_class = DefaultNet
        self.dynamic_parameters = dynamic_parameters
        self.awareness_criterion = None
        self.loss_combination_operator = operator.add
        self.start_selfaware_training = False
        self.stop_selfaware_training = False
        self.is_selfaware = False
        self.last_unaware_net = False

        self.max_confidence_per_output = []

        self.monitor = None
        for k in CONFIG.MONITORING:
            if CONFIG.MONITORING[k]:
                from lightwood.mixers.helpers.debugging import TrainingMonitor
                self.monitor = TrainingMonitor()
                break

        self.total_iterations = 0

        self._nonpersistent = {
            'sampler': None
        }

    def build_confidence_normalization_data(self, ds, subset_id=None):
        """
        :param ds:
        :return:
        """
        self.net = self.net.eval()

        ds.encoders = self.encoders
        ds.transformer = self.transformer

        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        loss_confidence_arr = []

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(self.net.device)
            labels = labels.to(self.net.device)

            with torch.no_grad():
                if self.is_selfaware:
                    outputs, awareness = self.net(inputs)
                else:
                    outputs = self.net(inputs)

            loss = None
            for k, criterion in enumerate(self.criterion_arr):
                if len(loss_confidence_arr) <= k:
                    loss_confidence_arr.append([])
                    self.max_confidence_per_output.append(None)

                try:
                    confidences = criterion.estimate_confidence(outputs[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]])
                    loss_confidence_arr[k].extend(confidences)
                except:
                    pass

        for k, _ in enumerate(self.criterion_arr):
            if len(loss_confidence_arr[k]) > 0:
                loss_confidence_arr[k] = np.array(loss_confidence_arr[k])
                nf_pct = np.percentile(loss_confidence_arr[k], 95)
                self.max_confidence_per_output[k] = max(loss_confidence_arr[k][loss_confidence_arr[k] < nf_pct])

        return True


    def fit(self, ds=None, callback=None):

        ret = 0
        for i in self.iter_fit(ds):
            ret = i
        self.encoders = ds.encoders
        return ret

    def predict(self, when_data_source, include_encoded_predictions=False):
        """
        :param when_data_source:
        :return:
        """
        when_data_source.transformer = self.transformer
        when_data_source.encoders = self.encoders
        _, _ = when_data_source[0]

        data_loader = DataLoader(when_data_source, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # set model into evaluation mode in order to skip things such as Dropout
        self.net = self.net.eval()

        outputs = []
        awareness_arr = []
        loss_confidence_arr = [[]] * len(when_data_source.out_indexes)

        for i, data in enumerate(data_loader, 0):
            inputs, _ = data
            inputs = inputs.to(self.net.device)

            with torch.no_grad():
                if self.is_selfaware:
                    output, awareness = self.net(inputs)
                    awareness = awareness.to('cpu')
                    awareness_arr.extend(awareness.tolist())
                else:
                    output = self.net(inputs)
                    awareness_arr = None

                for k, criterion in enumerate(self.criterion_arr):
                    try:
                        max_conf = 1
                        if len(self.max_confidence_per_output) >= (k - 1) and self.max_confidence_per_output[k] is not None:
                            max_conf = self.max_confidence_per_output[k]

                        confidences = criterion.estimate_confidence(output[:,when_data_source.out_indexes[k][0]:when_data_source.out_indexes[k][1]], max_conf)
                        loss_confidence_arr[k].extend(confidences)
                    except Exception as e:
                        loss_confidence_arr[k] = None

                output = output.to('cpu')

            outputs.extend(output)

        output_trasnformed_vectors = {}
        confidence_trasnformed_vectors = {}

        for i in range(len(outputs)):
            output_vector = outputs[i]
            transformed_output_vectors = when_data_source.transformer.revert(
                output_vector, feature_set='output_features')
            for feature in transformed_output_vectors:
                if feature not in output_trasnformed_vectors:
                    output_trasnformed_vectors[feature] = []
                output_trasnformed_vectors[feature] += [transformed_output_vectors[feature]]

        predictions = {}
        for k, output_column in enumerate(list(output_trasnformed_vectors.keys())):
            decoded_predictions = when_data_source.get_decoded_column_data(
                output_column,
                when_data_source.encoders[output_column]._pytorch_wrapper(output_trasnformed_vectors[output_column])
            )
            predictions[output_column] = {'predictions': decoded_predictions}
            if awareness_arr is not None:
                predictions[output_column]['selfaware_confidences'] = [1/x[k] for x in awareness_arr]

            if loss_confidence_arr[k] is not None:
                predictions[output_column]['loss_confidences'] = loss_confidence_arr[k]

            if include_encoded_predictions:
                predictions[output_column]['encoded_predictions'] = output_trasnformed_vectors[output_column]

        logging.info('Model predictions and decoding completed')

        return predictions

    def overall_certainty(self):
        """
        return an estimate of how certain is the model about the overall predictions,
        in this case its a measurement of how much did the variance of all the weights distributions
        reduced from its initial distribution
        :return:
        """
        if hasattr(self.net, 'calculate_overall_certainty'):
            return self.net.calculate_overall_certainty()
        else:
            return -1

    def error(self, ds, subset_id=None):
        """
        :param ds:
        :return:
        """
        self.net = self.net.eval()

        ds.encoders = self.encoders
        ds.transformer = self.transformer

        if self._nonpersistent['sampler'] is None:
            data_loader = DataLoader(ds, batch_size=self.batch_size,
                                     sampler=self._nonpersistent['sampler'], num_workers=0)
        else:
            data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        running_loss = 0.0
        error = 0

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(self.net.device)
            labels = labels.to(self.net.device)

            with torch.no_grad():
                if self.is_selfaware:
                    outputs, awareness = self.net(inputs)
                else:
                    outputs = self.net(inputs)

            loss = None
            for k, criterion in enumerate(self.criterion_arr):
                target_loss = criterion(outputs[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]], labels[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]])

                if loss is None:
                    loss = target_loss
                else:
                    loss += target_loss

            running_loss += loss.item()
            error = running_loss / (i + 1)

        if CONFIG.MONITORING['epoch_loss']:
            self.monitor.plot_loss(error, self.total_iterations, 'Test Epoch Error')
            self.monitor.plot_loss(error, self.total_iterations, f'Test Epoch Error - Subset {subset_id}')

        self.net = self.net.train()

        return error

    def get_model_copy(self):
        """
        get the actual mixer model
        :return: self.net
        """
        self.optimizer.zero_grad()
        return copy.deepcopy(self.net)

    def update_model(self, model):
        """
        replace the current model with a model object
        :param model: a model object
        :return: None
        """

        self.net = model

    def fit_data_source(self, ds):
        self.input_column_names = self.input_column_names \
            if self.input_column_names is not None else ds.get_feature_names('input_features')
        self.output_column_names = self.output_column_names \
            if self.output_column_names is not None else ds.get_feature_names('output_features')

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

    def iter_fit(self, ds, initialize=True, subset_id=None):
        """
        :param ds:
        :return:
        """
        if initialize:
            self.fit_data_source(ds)

            self.net = self.nn_class(ds, self.dynamic_parameters, selfaware=False)
            self.net = self.net.train()

            if self.batch_size < self.net.available_devices:
                self.batch_size = self.net.available_devices

            self.awareness_criterion = torch.nn.MSELoss()

            if self.criterion_arr is None:
                self.criterion_arr = []
                self.unreduced_criterion_arr = []
                if ds.output_weights is not None and ds.output_weights is not False:
                    output_weights = torch.Tensor(ds.output_weights).to(self.net.device)
                else:
                    output_weights = None
                for output_type in ds.out_types:
                    if output_type in (COLUMN_DATA_TYPES.CATEGORICAL):
                        self.criterion_arr.append(TransformCrossEntropyLoss(weight=output_weights))
                        self.unreduced_criterion_arr.append(TransformCrossEntropyLoss(weight=output_weights,reduce=False))
                    elif output_type in (COLUMN_DATA_TYPES.NUMERIC):
                        self.criterion_arr.append(torch.nn.MSELoss())
                        self.unreduced_criterion_arr.append(torch.nn.MSELoss(reduce=False))
                    else:
                        self.criterion_arr.append(torch.nn.MSELoss())
                        self.unreduced_criterion_arr.append(torch.nn.MSELoss(reduce=False))

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
            data_loader = DataLoader(ds, batch_size=self.batch_size, num_workers=0,
                                     sampler=self._nonpersistent['sampler'])

        for epoch in range(total_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            error = 0
            for i, data in enumerate(data_loader, 0):
                if self.start_selfaware_training and not self.is_selfaware:
                    logging.info('Making network selfaware !')
                    self.is_selfaware = True
                    self.net = self.nn_class(ds, self.dynamic_parameters, selfaware=True, pretrained_net=self.net.net)
                    self.last_unaware_net = copy.deepcopy(self.net.net)

                    # Lower the learning rate once we start training the selfaware network
                    self.optimizer_args['lr'] = self.optimizer.lr/4
                    gc.collect()
                    if 'cuda' in str(self.net.device):
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

                if self.stop_selfaware_training and self.is_selfaware:
                    logging.info('Cannot train selfaware network, training a normal network instead !')
                    self.is_selfaware = False
                    self.net = self.nn_class(ds, self.dynamic_parameters, selfaware=False, pretrained_net=self.last_unaware_net) #, pretrained_net=copy.deepcopy(self.net.net)

                    # Increase the learning rate closer to the previous levels
                    self.optimizer_args['lr'] = self.optimizer.lr * 4
                    gc.collect()
                    if 'cuda' in str(self.net.device):
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

                self.total_iterations += 1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                labels = labels.to(self.net.device)
                inputs = inputs.to(self.net.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # outputs = self.net(inputs)
                if self.is_selfaware:
                    outputs, awareness = self.net(inputs)
                else:
                    outputs = self.net(inputs)

                loss = None
                for k, criterion in enumerate(self.criterion_arr):
                    target_loss = criterion(outputs[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]], labels[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]])
                    if loss is None:
                        loss = target_loss
                    else:
                        loss += target_loss

                awareness_loss = None
                if self.is_selfaware:
                    unreduced_losses = []
                    for k, criterion in enumerate(self.unreduced_criterion_arr):
                        # redyce = True
                        target_loss = criterion(outputs[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]], labels[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]])

                        target_loss = target_loss.tolist()
                        if type(target_loss[0]) == type([]):
                            target_loss = [np.mean(x) for x in target_loss]
                        for i, value in enumerate(target_loss):
                            if len(unreduced_losses) <= i:
                                unreduced_losses.append([])
                            unreduced_losses[i].append(value)

                    unreduced_losses = torch.Tensor(unreduced_losses).to(self.net.device)

                    awareness_loss = self.awareness_criterion(awareness,unreduced_losses)

                    if CONFIG.MONITORING['batch_loss']:
                        self.monitor.plot_loss(awareness_loss.item(), self.total_iterations, 'Awreness Batch Loss')


                if CONFIG.MONITORING['batch_loss']:
                    self.monitor.plot_loss(loss.item(), self.total_iterations, 'Targets Batch Loss')



                if awareness_loss is not None:
                    awareness_loss.backward(retain_graph=True)

                running_loss += loss.item()
                loss.backward()

                # @NOTE: Decrease 900 if you want to plot gradients more often, I find it's too expensive to do so
                if CONFIG.MONITORING['network_heatmap'] and random.randint(0,1000) > 900:
                    weights = []
                    gradients = []
                    layer_name = []
                    for index, layer in enumerate(self.net.net):
                        if 'Linear' in str(type(layer)):
                            weights.append( list(layer.weight.cpu().detach().numpy().ravel()) )
                            gradients.append( list(layer.weight.grad.cpu().detach().numpy().ravel()) )
                            layer_name.append(f'Layer {index}-{index+1}')
                    self.monitor.weight_map(layer_name, weights, 'Predcitive network weights')
                    self.monitor.weight_map(layer_name, weights, 'Predictive network gradients')

                    if self.is_selfaware:
                        weights = []
                        gradients = []
                        layer_name = []
                        for index, layer in enumerate(self.net.awareness_net):
                            if 'Linear' in str(type(layer)):
                                weights.append( list(layer.weight.cpu().detach().numpy().ravel()) )
                                gradients.append( list(layer.weight.grad.cpu().detach().numpy().ravel()) )
                                layer_name.append(f'Layer {index}-{index+1}')
                        self.monitor.weight_map(layer_name, weights, 'Awareness network weights')
                        self.monitor.weight_map(layer_name, weights, 'Awareness network gradients')

                self.optimizer.step()
                # now that we have run backward in both losses, optimize()
                # (review: we may need to optimize for each step)

                error = running_loss / (i + 1)

                if CONFIG.MONITORING['batch_loss']:
                    #self.monitor.plot_loss(total_loss.item(), self.total_iterations, 'Total Batch Loss')
                    self.monitor.plot_loss(error, self.total_iterations, 'Mean Total Running Loss')

                if error < 1:
                    if self.loss_combination_operator == operator.add:
                        self.loss_combination_operator = operator.mul

            if CONFIG.MONITORING['epoch_loss']:
                self.monitor.plot_loss(error, self.total_iterations, 'Train Epoch Error')
                self.monitor.plot_loss(error, self.total_iterations, f'Train Epoch Error - Subset {subset_id}')
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

    # For Classification
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

    for i in mixer.iter_fit(ds):
        if i < 0.01:
            break

    predictions = mixer.predict(predict_input_ds)
    print(predictions)

    # For Regression

    # GENERATE DATA
    ###############

    config = {
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
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

    for i in mixer.iter_fit(ds):
        if i < 0.01:
            break

    predictions = mixer.predict(predict_input_ds)
    print(predictions)
