import math
import itertools as it

import torch
from torch.optim.optimizer import Optimizer



class Ranger(Optimizer):
    def __init__(self, params, lr=0.03, alpha=0.5, k=6, N_sma_threshold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.
        # @TODO Implement the above testing with AX ^

        #prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, N_sma_threshold=N_sma_threshold, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)

        #adjustable threshold
        self.N_sma_threshold = N_sma_threshold

        #look ahead params
        self.initial_lr = lr
        self.alpha = alpha
        self.k = k

        #radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]


    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)


    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data.float()
                    if grad.is_sparse:
                        raise RuntimeError('Ranger optimizer does not support sparse gradients')

                    p_data_fp32 = p.data.float()

                    state = self.state[p]  #get state dict for this param

                    # On the first run initialize the dictionary for each weight group
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_data_fp32)
                        state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                        #look ahead weight storage now in state dict
                        state['slow_buffer'] = torch.empty_like(p.data)
                        state['slow_buffer'].copy_(p.data)
                    # @TODO Couldn't this branch happen after the if above is entered in thus replacing torch.zero_like) ??
                    else:
                        state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                #begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                #compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    # Find the interpolated weight between the slower buffer (the weight `k` steps ago) and the current weight, set that as the state for RAdam
                    slow_p.add_(self.alpha, p.data - slow_p)
                    p.data.copy_(slow_p)

        return loss


import copy
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from lightwood.mixers.nn.helpers.default_net import DefaultNet
from lightwood.mixers.nn.helpers.transformer import Transformer


class NnMixer:

    def __init__(self, dynamic_parameters, is_categorical_output=False):
        self.is_categorical_output = is_categorical_output
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

            if self.is_categorical_output:
                target = labels.cpu().numpy()
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
            if self.is_categorical_output:
                if ds.output_weights is not None and ds.output_weights is not False:
                    output_weights = torch.Tensor(ds.output_weights).to(self.net.device)
                else:
                    output_weights = None
                print('\n\n\n==================')
                print(output_weights)
                print('==================')
                self.criterion = torch.nn.CrossEntropyLoss(weight=output_weights)
            else:
                print('HERE USING MSE LOSS !!!')
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
                outputs = self.net(inputs)

                if self.is_categorical_output:
                    target = labels.cpu().numpy()
                    target_indexes = np.where(target>0)[1]
                    targets_c = torch.LongTensor(target_indexes)
                    labels = targets_c.to(self.net.device)

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                # Maybe make this a scheduler later
                # Start flat and then go into cosine annealing
                if total_iterations > 1200 and epoch > 60:
                    for group in self.optimizer.param_groups:
                        if self.optimizer.initial_lr * 1/100 < group['lr']:
                            group['lr'] = group['lr'] - self.optimizer.initial_lr * 1/400

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
