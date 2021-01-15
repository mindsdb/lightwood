import copy
import random
import time

import torch
from torch.utils.data import DataLoader
import numpy as np
import operator

from lightwood.helpers.torch import LightwoodAutocast
from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.selfaware import SelfAware
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.mixers.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.mixers import BaseMixer
from lightwood.logger import log


class NnMixer(BaseMixer):

    def __init__(self,
                 selfaware=False,
                 callback_on_iter=None,
                 eval_every_x_epochs=20,
                 dropout_p=0.0,
                 stop_training_after_seconds=None,
                 stop_model_building_after_seconds=None,
                 param_optimizer=None):
        """
        :param selfaware: bool
        :param callback_on_iter: Callable[epoch, training_error, test_error, delta_mean, accuracy]
        :param eval_every_x_epochs: int
        :param stop_training_after_seconds: int
        :param stop_model_building_after_seconds: int
        :param param_optimizer: ?
        """
        super().__init__()

        self.selfaware = selfaware
        self.eval_every_x_epochs = eval_every_x_epochs
        self.stop_training_after_seconds = stop_training_after_seconds
        self.stop_model_building_after_seconds = stop_model_building_after_seconds
        self.param_optimizer = param_optimizer

        self.net = None
        self.selfaware_net = None
        self.optimizer = None
        self.selfaware_optimizer = None
        self.optimizer_class = None
        self.optimizer_args = None
        self.selfaware_optimizer_args = None
        self.criterion_arr = None
        self.unreduced_criterion_arr = None

        self.batch_size = 200

        self.nn_class = DefaultNet
        self.dropout_p = max(0.0, min(1.0, dropout_p))
        self.dynamic_parameters = {}
        self.awareness_criterion = None
        self.awareness_scale_factor = 1/10  # scales self-aware total loss contribution
        self.selfaware_lr_factor = 1/2      # scales self-aware learning rate compared to mixer
        self.start_selfaware_training = False
        self.stop_selfaware_training = False
        self.is_selfaware = False

        self.max_confidence_per_output = []
        self.monitor = None

        for k in CONFIG.MONITORING:
            if CONFIG.MONITORING[k]:
                from lightwood.mixers.helpers.debugging import TrainingMonitor
                self.monitor = TrainingMonitor()
                break

        self.total_iterations = 0
        self._nonpersistent = {'sampler': None, 'callback': callback_on_iter}

    def _default_on_iter(self, epoch, train_error, test_error, delta_mean, accuracy):
        test_error_rounded = round(test_error, 4)
        for col in accuracy:
            value = accuracy[col]['value']
            if accuracy[col]['function'] == 'r2_score':
                value_rounded = round(value, 3)
                log.info(f'We\'ve reached training epoch nr {epoch} with an r2 score of {value_rounded} on the testing dataset')
            else:
                value_pct = round(value * 100, 2)
                log.info(f'We\'ve reached training epoch nr {epoch} with an accuracy of {value_pct}% on the testing dataset')

    def _build_confidence_normalization_data(self, ds, subset_id=None):
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
                    self.selfaware_net.eval()
                    outputs = self.net(inputs)
                    awareness = self.selfaware_net(inputs, outputs)
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
                except Exception:
                    pass

        for k, _ in enumerate(self.criterion_arr):
            if len(loss_confidence_arr[k]) > 0:
                loss_confidence_arr[k] = np.array(loss_confidence_arr[k])
                nf_pct = np.percentile(loss_confidence_arr[k], 95)

                losses_bellow_95th_percentile = loss_confidence_arr[k][loss_confidence_arr[k] < nf_pct]
                if len(losses_bellow_95th_percentile) < 1:
                    losses_bellow_95th_percentile = loss_confidence_arr[k]

                self.max_confidence_per_output[k] = max(losses_bellow_95th_percentile)

        return True

    def _fit(self, train_ds, test_ds, stop_training_after_seconds=None):
        """
        :param stop_training_after_seconds: int
        """
        if stop_training_after_seconds is None:
            stop_training_after_seconds = self.stop_training_after_seconds

        input_sample, output_sample = train_ds[0]

        self.net = self.nn_class(
            self.dynamic_parameters,
            input_size=len(input_sample),
            output_size=len(output_sample),
            nr_outputs=len(train_ds.out_types),
            dropout=self.dropout_p
        )
        self.net = self.net.train()

        self.selfaware_net = SelfAware(
            input_size=len(input_sample),
            output_size=len(output_sample),
            nr_outputs=len(train_ds.out_types)
        )
        self.selfaware_net = self.selfaware_net.train()

        if self.batch_size < self.net.available_devices:
            self.batch_size = self.net.available_devices

        self.awareness_criterion = torch.nn.MSELoss()

        if self.criterion_arr is None:
            self.criterion_arr = []
            self.unreduced_criterion_arr = []
            if train_ds.output_weights is not None and train_ds.output_weights is not False:
                output_weights = torch.Tensor(train_ds.output_weights).to(self.net.device)
            else:
                output_weights = None

            for k, output_type in enumerate(train_ds.out_types):
                if output_type in (COLUMN_DATA_TYPES.CATEGORICAL, COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL):
                    if output_weights is None:
                        weights_slice = None
                    else:
                        # account for numerical features, not included in the output_weights
                        s_idx = train_ds.out_indexes[k][0] - train_ds.output_weights_offset[k]
                        e_idx = train_ds.out_indexes[k][1] - train_ds.output_weights_offset[k]
                        weights_slice = output_weights[s_idx:e_idx]

                    if output_type == COLUMN_DATA_TYPES.CATEGORICAL:
                        self.criterion_arr.append(TransformCrossEntropyLoss(weight=weights_slice))
                        self.unreduced_criterion_arr.append(TransformCrossEntropyLoss(weight=weights_slice, reduction='none'))
                    elif output_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                        self.criterion_arr.append(torch.nn.BCEWithLogitsLoss(weight=weights_slice))
                        self.unreduced_criterion_arr.append(torch.nn.BCEWithLogitsLoss(weight=weights_slice, reduction='none'))
                # Note: MSELoss works great for numeric, for the other types it's more of a placeholder
                else:
                    self.criterion_arr.append(torch.nn.MSELoss())
                    self.unreduced_criterion_arr.append(torch.nn.MSELoss(reduction='none'))

        self.optimizer_class = Ranger
        if self.optimizer_args is None:
            self.optimizer_args = {'lr': 0.0005}

        if 'beta1' in self.dynamic_parameters:
            self.optimizer_args['betas'] = (self.dynamic_parameters['beta1'], 0.999)

        for optimizer_arg_name in ['lr', 'k', 'N_sma_threshold']:
            if optimizer_arg_name in self.dynamic_parameters:
                self.optimizer_args[optimizer_arg_name] = self.dynamic_parameters[optimizer_arg_name]

        self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

        self.selfaware_optimizer_args = copy.deepcopy(self.optimizer_args)
        self.selfaware_optimizer_args['lr'] = self.selfaware_optimizer_args['lr'] * self.selfaware_lr_factor
        self.selfaware_optimizer = self.optimizer_class(self.selfaware_net.parameters(), **self.optimizer_args)

        if stop_training_after_seconds is None:
            stop_training_after_seconds = round(train_ds.data_frame.shape[0] * train_ds.data_frame.shape[1] / 5)

        if self.stop_model_building_after_seconds is None:
            self.stop_model_building_after_seconds = stop_training_after_seconds * 3

        if self.param_optimizer is not None:
            input_size = len(train_ds[0][0])
            training_data_length = len(train_ds)
            while True:
                training_time_per_iteration = stop_model_building_after_seconds / self.param_optimizer.total_trials

                # Some heuristics...
                if training_time_per_iteration > input_size:
                    if training_time_per_iteration > min((training_data_length / (4 * input_size)), 16 * input_size):
                        break

                self.param_optimizer.total_trials = self.param_optimizer.total_trials - 1
                if self.param_optimizer.total_trials < 8:
                    self.param_optimizer.total_trials = 8
                    break

            training_time_per_iteration = stop_model_building_after_seconds / self.param_optimizer.total_trials

            self.dynamic_parameters = self.param_optimizer.evaluate(lambda dynamic_parameters: self.evaluate(from_data_ds, test_data_ds, dynamic_parameters, max_training_time=training_time_per_iteration, max_epochs=None))

            log.info('Using hyperparameter set: ', best_parameters)
        else:
            self.dynamic_parameters = {}

        started = time.time()
        log_reasure = time.time()
        stop_training = False

        for subset_iteration in [1, 2]:
            if stop_training:
                break
            subset_id_arr =  [*train_ds.subsets.keys()]
            for subset_id in subset_id_arr:
                started_subset = time.time()
                if stop_training:
                    break

                subset_train_ds = train_ds.subsets[subset_id]
                subset_test_ds = test_ds.subsets[subset_id]

                lowest_error = None
                last_test_error = None
                last_subset_test_error = None
                test_error_delta_buff = []
                subset_test_error_delta_buff = []
                best_model = None

                #iterate over the iter_fit and see what the epoch and mixer error is
                for epoch, training_error in enumerate(self._iter_fit(subset_train_ds, subset_id=subset_id)):

                    # Log this every now and then so that the user knows it's running
                    if (int(time.time()) - log_reasure) > 30:
                        log_reasure = time.time()
                        log.info(f'Lightwood training, iteration {epoch}, training error {training_error}')


                    # Prime the model on each subset for a bit
                    if subset_iteration == 1:
                        break

                    # Once the training error is getting smaller, enable dropout to teach the network to predict without certain features
                    if subset_iteration > 1 and training_error < 0.4 and not train_ds.enable_dropout:
                        self.eval_every_x_epochs = max(1, int(self.eval_every_x_epochs / 2) )
                        log.info('Enabled dropout !')
                        train_ds.enable_dropout = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    # If the selfaware network isn't able to train, go back to the original network
                    if subset_iteration > 1 and (np.isnan(training_error) or np.isinf(training_error) or training_error > pow(10,5)) and not self.stop_selfaware_training:
                        self.start_selfaware_training = False
                        self.stop_selfaware_training = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    # Once we are past the priming/warmup period, start training the selfaware network
                    if subset_iteration > 1 and not self.is_selfaware and self.selfaware and not self.stop_selfaware_training and training_error < 0.35:
                        log.info('Started selfaware training !')
                        self.start_selfaware_training = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    if epoch % self.eval_every_x_epochs == 0:
                        test_error = self._error(test_ds)
                        subset_test_error = self._error(subset_test_ds, subset_id=subset_id)
                        log.info(f'Subtest test error: {subset_test_error} on subset {subset_id}, overall test error: {test_error}')

                        if lowest_error is None or test_error < lowest_error:
                            lowest_error = test_error
                            best_model = self._get_model_copy()

                        if last_subset_test_error is None:
                            pass
                        else:
                            subset_test_error_delta_buff.append(last_subset_test_error - subset_test_error)

                        last_subset_test_error = subset_test_error

                        if last_test_error is None:
                            pass
                        else:
                            test_error_delta_buff.append(last_test_error - test_error)

                        last_test_error = test_error

                        delta_mean = np.mean(test_error_delta_buff[-5:]) if test_error_delta_buff else 0
                        subset_delta_mean = np.mean(subset_test_error_delta_buff[-5:]) if subset_test_error_delta_buff else 0

                        if self._nonpersistent['callback'] is not None:
                            self._nonpersistent['callback'](epoch, training_error, test_error, delta_mean, self.calculate_accuracy(test_ds))
                        else:
                            self._default_on_iter(epoch, training_error, test_error, delta_mean, self.calculate_accuracy(test_ds))
                        self.net.train()

                        # Stop if we're past the time limit allocated for training
                        if (time.time() - started) > stop_training_after_seconds:
                            stop_training = True

                        # If the trauining subset is overfitting on it's associated testing subset
                        if (subset_delta_mean <= 0 and len(subset_test_error_delta_buff) > 4) or (time.time() - started_subset) > stop_training_after_seconds/len(train_ds.subsets.keys()) or subset_test_error < 0.001:
                            log.info('Finished fitting on {subset_id} of {no_subsets} subset'.format(subset_id=subset_id, no_subsets=len(train_ds.subsets.keys())))

                            self._update_model(best_model)


                            if subset_id == subset_id_arr[-1]:
                                stop_training = True
                            elif not stop_training:
                                break

                        if stop_training:
                            self._update_model(best_model)
                            if self.is_selfaware:
                                self._build_confidence_normalization_data(train_ds)
                                self._adjust(test_ds)

                            self._iter_fit(test_ds, max_epochs=1)
                            self.encoders = train_ds.encoders
                            log.info('Finished training model !')
                            break

    def _predict(self, when_data_source, include_extra_data=False):
        data_loader = DataLoader(
            when_data_source,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        # set model into evaluation mode in order to skip things such as Dropout
        self.net = self.net.eval()

        outputs = []
        awareness_arr = []
        loss_confidence_arr = [[] for _ in when_data_source.out_indexes]

        for i, data in enumerate(data_loader, 0):
            inputs, _ = data
            inputs = inputs.to(self.net.device)

            with torch.no_grad():
                if self.is_selfaware:
                    self.selfaware_net.eval()
                    output = self.net(inputs)
                    awareness = self.selfaware_net(inputs, output)
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
                    except Exception:
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
                output_trasnformed_vectors[feature].append(transformed_output_vectors[feature])

        predictions = {}
        for k, output_column in enumerate(list(output_trasnformed_vectors.keys())):
            decoded_predictions = when_data_source.get_decoded_column_data(
                output_column,
                torch.Tensor(output_trasnformed_vectors[output_column])
            )

            predictions[output_column] = {'predictions': decoded_predictions['predictions']}

            if awareness_arr is not None:
                predictions[output_column]['selfaware_confidences'] = [1/abs(x[k]) if x[k] != 0 else 1/0.000001 for x in awareness_arr]

            if loss_confidence_arr[k] is not None:
                predictions[output_column]['loss_confidences'] = loss_confidence_arr[k]

            if include_extra_data:
                predictions[output_column]['encoded_predictions'] = output_trasnformed_vectors[output_column]

            if 'class_distribution' in decoded_predictions:
                predictions[output_column]['class_distribution'] = decoded_predictions['class_distribution']
                predictions[output_column]['class_labels'] = decoded_predictions['class_labels']

        log.info('Model predictions and decoding completed')

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

    def _error(self, ds, subset_id=None):
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

            with torch.no_grad():
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

    def _get_model_copy(self):
        """
        get the actual mixer model
        :return: self.net
        """
        self.optimizer.zero_grad()
        return copy.deepcopy(self.net)

    def _update_model(self, model):
        """
        replace the current model with a model object
        :param model: a model object
        :return: None
        """

        self.net = model

        if 'cuda' in str(self.net.device):
            torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

    def _iter_fit(self, ds, subset_id=None, max_epochs=120000):
        if self._nonpersistent['sampler'] is None:
            data_loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
        else:
            data_loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=0,
                sampler=self._nonpersistent['sampler']
            )

        for epoch in range(max_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            error = 0
            for i, data in enumerate(data_loader, 0):
                if self.start_selfaware_training and not self.is_selfaware:
                    log.info('Starting to train selfaware network for better confidence determination !')
                    self.is_selfaware = True

                if self.stop_selfaware_training and self.is_selfaware:
                    log.info('Cannot train selfaware network, will fallback to using simpler confidence models !')
                    self.is_selfaware = False

                self.total_iterations += 1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                labels = labels.to(self.net.device)
                inputs = inputs.to(self.net.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.selfaware_optimizer.zero_grad()

                # forward + backward + optimize
                with LightwoodAutocast():
                    outputs = self.net(inputs)
                if self.is_selfaware:
                    with LightwoodAutocast():
                        awareness = self.selfaware_net(inputs.detach(), outputs.detach())

                loss = None
                for k, criterion in enumerate(self.criterion_arr):
                    with LightwoodAutocast():
                        target_loss = criterion(outputs[:, ds.out_indexes[k][0]:ds.out_indexes[k][1]],
                                                labels[:, ds.out_indexes[k][0]:ds.out_indexes[k][1]])

                    if loss is None:
                        loss = target_loss
                    else:
                        loss += target_loss

                awareness_loss = None
                if self.is_selfaware:
                    unreduced_losses = []
                    for k, criterion in enumerate(self.unreduced_criterion_arr):
                        target_loss = criterion(
                            outputs[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]],
                            labels[:,ds.out_indexes[k][0]:ds.out_indexes[k][1]]
                        )

                        target_loss = target_loss.tolist()
                        if type(target_loss[0]) == type([]):
                            target_loss = [np.mean(x) for x in target_loss]
                        for i, value in enumerate(target_loss):
                            if len(unreduced_losses) <= i:
                                unreduced_losses.append([])
                            unreduced_losses[i].append(value)

                    unreduced_losses = torch.Tensor(unreduced_losses).to(self.net.device)

                    awareness_loss = self.awareness_criterion(awareness, unreduced_losses) * self.awareness_scale_factor

                    if CONFIG.MONITORING['batch_loss']:
                        self.monitor.plot_loss(awareness_loss.item(), self.total_iterations, 'Awreness Batch Loss')

                if CONFIG.MONITORING['batch_loss']:
                    self.monitor.plot_loss(loss.item(), self.total_iterations, 'Targets Batch Loss')

                if awareness_loss is not None:
                    awareness_loss.backward(retain_graph=True)

                running_loss += loss.item()
                loss.backward()

                # @NOTE: Decrease 900 if you want to plot gradients more often, I find it's too expensive to do so
                if CONFIG.MONITORING['network_heatmap'] and random.randint(0, 1000) > 900:
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
                        for index, layer in enumerate(self.selfaware_net.net):
                            if 'Linear' in str(type(layer)):
                                weights.append( list(layer.weight.cpu().detach().numpy().ravel()) )
                                gradients.append( list(layer.weight.grad.cpu().detach().numpy().ravel()) )
                                layer_name.append(f'Layer {index}-{index+1}')
                        self.monitor.weight_map(layer_name, weights, 'Awareness network weights')
                        self.monitor.weight_map(layer_name, weights, 'Awareness network gradients')

                # now that we have run backward in both losses, optimize()
                # (review: we may need to optimize for each step)
                self.optimizer.step()

                if self.is_selfaware and self.start_selfaware_training:
                    self.selfaware_optimizer.step()

                error = running_loss / (i + 1)

                if CONFIG.MONITORING['batch_loss']:
                    #self.monitor.plot_loss(total_loss.item(), self.total_iterations, 'Total Batch Loss')
                    self.monitor.plot_loss(error, self.total_iterations, 'Mean Total Running Loss')

            if CONFIG.MONITORING['epoch_loss']:
                self.monitor.plot_loss(error, self.total_iterations, 'Train Epoch Error')
                self.monitor.plot_loss(error, self.total_iterations, f'Train Epoch Error - Subset {subset_id}')
            yield error

    def to(self, device, available_devices):
        self.net.to(device, available_devices)
        self.selfaware_net.to(device, available_devices)
        for enc in self.encoders:
            self.encoders[enc].to(device, available_devices)
        return self

    def calculate_accuracy(self, ds):
        predictions = self.predict(ds, include_extra_data=True)
        accuracies = {}

        for output_column in [feature['name'] for feature in ds.config['output_features']]:

            col_type = ds.get_column_config(output_column)['type']

            if col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                reals = [tuple(x) for x in ds.get_column_original_data(output_column)]
                preds = [tuple(x) for x in predictions[output_column]['predictions']]
            else:
                reals = [str(x) for x in ds.get_column_original_data(output_column)]
                preds = [str(x) for x in predictions[output_column]['predictions']]

            if 'weights' in ds.get_column_config(output_column):
                weight_map = ds.get_column_config(output_column)['weights']
                # omit points for which there's no target info in timeseries forecasting
                if '_timestep_' in ds.get_column_config(output_column)['name']:
                    classes = set(weight_map.keys())
                    observed_classes = set(reals)
                    for unrecognized_class in observed_classes.difference(classes):
                        weight_map[unrecognized_class] = 0
            else:
                weight_map = None

            accuracy = BaseMixer._apply_accuracy_function(
                ds.get_column_config(output_column)['type'],
                reals,
                preds,
                weight_map=weight_map,
                encoder=ds.encoders[output_column]
            )

            if ds.get_column_config(output_column)['type'] == COLUMN_DATA_TYPES.NUMERIC:
                ds.encoders[output_column].decode_log = True
                preds = ds.get_decoded_column_data(
                    output_column,
                    predictions[output_column]['encoded_predictions']
                )['predictions']

                alternative_accuracy = BaseMixer._apply_accuracy_function(
                    ds.get_column_config(output_column)['type'],
                    reals,
                    preds,
                    weight_map=weight_map
                )

                if alternative_accuracy['value'] > accuracy['value']:
                    accuracy = alternative_accuracy
                else:
                    ds.encoders[output_column].decode_log = False

            accuracies[output_column] = accuracy

        return accuracies
