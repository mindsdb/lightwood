import copy
import logging
import random
import time

import torch
from torch.utils.data import DataLoader
import numpy as np
import gc
import operator

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.mixers.helpers.quantile_loss import QuantileLoss
from lightwood.mixers.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES


class NnMixer:
    def __init__(self, dynamic_parameters, config=None):
        self.config = config
        self.out_types = None
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
        self.start_selfaware_training = False
        self.stop_selfaware_training = False
        self.is_selfaware = False
        self.last_unaware_net = False

        self.max_confidence_per_output = []
        self.monitor = None
        self.quantiles = [0.5,  0.2,0.8,  0.1,0.9,  0.05,0.95,  0.02,0.98,  0.005,0.995]
        self.quantiles_pair = [9,10]
        self.map_mean_sc_qi = None

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

                losses_bellow_95th_percentile = loss_confidence_arr[k][loss_confidence_arr[k] < nf_pct]
                if len(losses_bellow_95th_percentile) < 1:
                    losses_bellow_95th_percentile = loss_confidence_arr[k]

                self.max_confidence_per_output[k] = max(losses_bellow_95th_percentile)

        return True

    def fit(self, train_ds, test_ds, callback=None, stop_training_after_seconds=None, eval_every_x_epochs=None):
        started = time.time()
        log_reasure = time.time()
        first_run = True
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
                best_selfaware_model = None

                #iterate over the iter_fit and see what the epoch and mixer error is
                for epoch, training_error in enumerate(self.iter_fit(subset_train_ds, initialize=first_run, subset_id=subset_id)):
                    first_run = False

                    # Log this every now and then so that the user knows it's running
                    if (int(time.time()) - log_reasure) > 30:
                        log_reasure = time.time()
                        logging.info(f'Lightwood training, iteration {epoch}, training error {training_error}')


                    # Prime the model on each subset for a bit
                    if subset_iteration == 1:
                        break

                    # Once the training error is getting smaller, enable dropout to teach the network to predict without certain features
                    if subset_iteration > 1 and training_error < 0.4 and not train_ds.enable_dropout:
                        eval_every_x_epochs = max(1, int(eval_every_x_epochs/2) )
                        logging.info('Enabled dropout !')
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
                    if subset_iteration > 1 and not self.is_selfaware and self.config['mixer']['selfaware'] and not self.stop_selfaware_training and training_error < 0.35:
                        logging.info('Started selfaware training !')
                        self.start_selfaware_training = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    if epoch % eval_every_x_epochs == 0:
                        test_error = self.error(test_ds)
                        subset_test_error = self.error(subset_test_ds, subset_id=subset_id)
                        logging.info(f'Subtest test error: {subset_test_error} on subset {subset_id}, overall test error: {test_error}')

                        if lowest_error is None or test_error < lowest_error:
                            lowest_error = test_error
                            if self.is_selfaware:
                                best_selfaware_model = self.get_model_copy()
                            else:
                                best_model = self.get_model_copy()

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

                        delta_mean = np.mean(test_error_delta_buff[-5:])
                        subset_delta_mean = np.mean(subset_test_error_delta_buff[-5:])

                        if callback is not None:
                            callback(epoch, training_error, test_error, delta_mean)

                        # Stop if we're past the time limit allocated for training
                        if (time.time() - started) > stop_training_after_seconds:
                            stop_training = True

                        # If the trauining subset is overfitting on it's associated testing subset
                        if (subset_delta_mean <= 0 and len(subset_test_error_delta_buff) > 4) or (time.time() - started_subset) > stop_training_after_seconds/len(train_ds.subsets.keys()) or subset_test_error < 0.001:
                            logging.info('Finished fitting on {subset_id} of {no_subsets} subset'.format(subset_id=subset_id, no_subsets=len(train_ds.subsets.keys())))

                            if self.is_selfaware:
                                if best_selfaware_model is not None:
                                    self.update_model(best_selfaware_model)
                            else:
                                self.update_model(best_model)


                            if subset_id == subset_id_arr[-1]:
                                stop_training = True
                            elif not stop_training:
                                break

                        if stop_training:
                            if self.is_selfaware:
                                self.update_model(best_selfaware_model)
                                self.build_confidence_normalization_data(train_ds)
                                self.adjust(test_ds)
                            else:
                                self.update_model(best_model)
                            self.encoders = train_ds.encoders
                            logging.info('Finished training model !')
                            break

    def adjust(self, test_data_source):
        predictions = self.predict(test_data_source, include_extra_data=True)

        narrowest_correct_qi_arr = []
        corr_conf_correct_qi_arr = []
        selfaware_confidence_arr = []

        for col in predictions:
            p = predictions[col]
            if 'every_confidence_range' in p:
                for i in range(len(p['predictions'])):
                    set_qi = None
                    set_qi_conf = None
                    for qi in range(int((len(self.quantiles) - 1)/2)):
                        a = p['every_confidence_range'][i][qi*2]
                        if p['every_confidence_range'][i][qi*2] < p['predictions'][i] < p['every_confidence_range'][i][qi*2 + 1]:
                            set_qi = qi
                            break

                    narrowest_correct_qi_arr.append(set_qi)
                    selfaware_confidence_arr.append(p['selfaware_confidences'][i])

        map_qi_mean_sc = {}
        for i in range(len(narrowest_correct_qi_arr)):
            qi = narrowest_correct_qi_arr[i]
            if qi not in map_qi_mean_sc:
                map_qi_mean_sc[qi] = []
            map_qi_mean_sc[qi].append(selfaware_confidence_arr[i])

        for qi in map_qi_mean_sc:
            map_qi_mean_sc[qi] = np.mean(map_qi_mean_sc[qi])

        self.map_mean_sc_qi = {}
        for qi in map_qi_mean_sc:
            self.map_mean_sc_qi[map_qi_mean_sc[qi]] = qi

    def select_quantile(self, selfaware_confidence):
        if self.map_mean_sc_qi is None:
            return self.quantiles_pair

        for k in sorted(list(self.map_mean_sc_qi.keys())):
            if selfaware_confidence <= k:
                if self.map_mean_sc_qi[k] is not None:
                    return [self.map_mean_sc_qi[k]*2+1,self.map_mean_sc_qi[k]*2+2]

        return self.quantiles_pair

    def predict(self, when_data_source, include_extra_data=False):
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
                output_trasnformed_vectors[feature].append(transformed_output_vectors[feature])

        predictions = {}
        for k, output_column in enumerate(list(output_trasnformed_vectors.keys())):
            decoded_predictions = when_data_source.get_decoded_column_data(
                output_column,
                when_data_source.encoders[output_column]._pytorch_wrapper(output_trasnformed_vectors[output_column])
            )

            if self.out_types[k] in (COLUMN_DATA_TYPES.NUMERIC):
                predictions[output_column] = {'predictions': [x[0] for x in decoded_predictions]}

                if include_extra_data:
                    predictions[output_column]['every_confidence_range'] = [x[1:] for x in decoded_predictions]

            else:
                predictions[output_column] = {'predictions': decoded_predictions}

            if awareness_arr is not None:
                predictions[output_column]['selfaware_confidences'] = [1/abs(x[k]) if x[k] != 0 else 1/0.000001 for x in awareness_arr]

            if self.out_types[k] in (COLUMN_DATA_TYPES.NUMERIC):
                predictions[output_column]['confidence_range'] = []
                predictions[output_column]['quantile_confidences'] = []

                for i, pred in enumerate(decoded_predictions):
                    if 'selfaware_confidences' in predictions[output_column]:
                        sc = predictions[output_column]['selfaware_confidences'][i]
                    else:
                        sc = pow(10,3)

                    qp = self.select_quantile(sc)
                    predictions[output_column]['confidence_range'].append([pred[qp[0]],pred[qp[1]]])
                    predictions[output_column]['quantile_confidences'].append(self.quantiles[qp[1]] - self.quantiles[qp[0]])

            if loss_confidence_arr[k] is not None:
                predictions[output_column]['loss_confidences'] = loss_confidence_arr[k]

            if include_extra_data:
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

        if 'cuda' in str(self.net.device):
            torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.optimizer = self.optimizer_class(self.net.parameters(), **self.optimizer_args)

    def fit_data_source(self, ds):
        self.input_column_names = self.input_column_names \
            if self.input_column_names is not None else ds.get_feature_names('input_features')
        self.output_column_names = self.output_column_names \
            if self.output_column_names is not None else ds.get_feature_names('output_features')

        self.out_types = ds.out_types
        for n, out_type in enumerate(self.out_types):
            if out_type == COLUMN_DATA_TYPES.NUMERIC:
                ds.encoders[self.output_column_names[n]].extra_outputs = len(self.quantiles) - 1

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

            input_sample, output_sample = ds[0]

            self.net = self.nn_class(self.dynamic_parameters,
                                     input_size=len(input_sample),
                                     output_size=len(output_sample),
                                     nr_outputs=len(self.out_types),
                                     selfaware=False,
                                     deterministic=self.config['mixer']['deterministic'])
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

                for k, output_type in enumerate(self.out_types):
                    if output_type == COLUMN_DATA_TYPES.CATEGORICAL:
                        if output_weights is None:
                            weights_slice = None
                        else:
                            weights_slice = output_weights[ds.out_indexes[k][0]:ds.out_indexes[k][1]]

                        self.criterion_arr.append(TransformCrossEntropyLoss(weight=weights_slice))
                        self.unreduced_criterion_arr.append(TransformCrossEntropyLoss(weight=weights_slice,reduce=False))
                    elif output_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                        if output_weights is None:
                            weights_slice = None
                        else:
                            weights_slice = output_weights[ds.out_indexes[k][0]:ds.out_indexes[k][1]]

                        self.criterion_arr.append(torch.nn.BCEWithLogitsLoss(weight=weights_slice))
                        self.unreduced_criterion_arr.append(torch.nn.BCEWithLogitsLoss(weight=weights_slice, reduce=False))
                    elif output_type == COLUMN_DATA_TYPES.NUMERIC:
                        self.criterion_arr.append(QuantileLoss(quantiles=self.quantiles))
                        self.unreduced_criterion_arr.append(QuantileLoss(quantiles=self.quantiles, reduce=False))
                    else:
                        self.criterion_arr.append(torch.nn.MSELoss())
                        self.unreduced_criterion_arr.append(torch.nn.MSELoss(reduce=False))

            self.optimizer_class = Ranger
            if self.optimizer_args is None:
                self.optimizer_args = {'lr': 0.0005}

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
                    self.net = self.nn_class(self.dynamic_parameters, nr_outputs=len(self.out_types) ,selfaware=True, pretrained_net=self.net.net, deterministic=self.config['mixer']['deterministic'])
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
                    self.net = self.nn_class(self.dynamic_parameters, nr_outputs=len(self.out_types) ,selfaware=False, pretrained_net=self.last_unaware_net, deterministic=self.config['mixer']['deterministic'])

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

            if CONFIG.MONITORING['epoch_loss']:
                self.monitor.plot_loss(error, self.total_iterations, 'Train Epoch Error')
                self.monitor.plot_loss(error, self.total_iterations, f'Train Epoch Error - Subset {subset_id}')
            yield error


    def to(self, device, available_devices):
        self.net.to(device, available_devices)
        return self

if __name__ == "__main__":
    import random
    import pandas
    from lightwood.api.data_source import DataSource
    from lightwood.data_schemas.predictor_config import predictor_config_schema

    config = {
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric'
            },
            {
                'name': 'y',
                'type': 'numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'numeric'
            },
            {
                'name': 'z`',
                'type': 'categorical'
            }
        ]
    }
    config = predictor_config_schema.validate(config)

    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = [i + 0.5 for i in range(10)]
    data['z`'] = ['low' if i < 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)
    ds = DataSource(data_frame, config)
    ds.prepare_encoders()

    mixer = NnMixer({}, config)
    mixer.fit(ds,ds, stop_training_after_seconds=50)

    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    predict_input_ds.prepare_encoders()
    predictions = mixer.predict(predict_input_ds)
    print(predictions)
