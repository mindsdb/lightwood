import copy
import logging
from collections import Counter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pyro

from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger


class BayesianNnMixer:

    def __init__(self, dynamic_parameters, is_categorical_output=False):
        self.is_categorical_output = is_categorical_output
        self.net = None
        self.optimizer = None
        self.input_column_names = None
        self.output_column_names = None
        self.data_loader = None
        self.transformer = None
        self.encoders = None
        self.criterion = None

        self.batch_size = 200
        self.epochs = 120000

        self.nn_class = DefaultNet
        self.dynamic_parameters = dynamic_parameters

        #Pyro stuff
        self.softplus = torch.nn.Softplus()

    def fit(self, ds=None, callback=None):

        ret = 0
        for i in self.iter_fit(ds):
            ret = i
        self.encoders = ds.encoders
        return ret

    def pyro_model(self, input_data, output_data):

        inw_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.net.net[0].weight, device=self.net.device), scale=torch.ones_like(self.net.net[0].weight))
        inb_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.net.net[0].bias, device=self.net.device), scale=torch.ones_like(self.net.net[0].bias))

        outw_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.net.net[-1].weight, device=self.net.device), scale=torch.ones_like(self.net.net[-1].weight))
        outb_prior = pyro.distributions.Normal(loc=torch.zeros_like(self.net.net[-1].bias, device=self.net.device), scale=torch.ones_like(self.net.net[-1].bias))

        priors = {'net[0].weight': inw_prior, 'net[0].bias': inb_prior,  'net[-1].weight': outw_prior, 'net[-1].bias': outb_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        lhat = lifted_reg_model(input_data)

        pyro.sample("obs", pyro.distributions.Categorical(logits=lhat), obs=output_data)

    def pyro_guide(self, input_data, output_data):
        # First layer weight distribution priors
        inw_mu = torch.randn_like(self.net.net[0].weight, device=self.net.device)
        inw_sigma = torch.randn_like(self.net.net[0].weight, device=self.net.device)
        inw_mu_param = pyro.param("inw_mu", inw_mu)
        inw_sigma_param = self.softplus(pyro.param("inw_sigma", inw_sigma))
        inw_prior = pyro.distributions.Normal(loc=inw_mu_param, scale=inw_sigma_param)

        # First layer bias distribution priors
        inb_mu = torch.randn_like(self.net.net[0].bias, device=self.net.device)
        inb_sigma = torch.randn_like(self.net.net[0].bias, device=self.net.device)
        inb_mu_param = pyro.param("inb_mu", inb_mu)
        inb_sigma_param = self.softplus(pyro.param("inb_sigma", inb_sigma))
        inb_prior = pyro.distributions.Normal(loc=inb_mu_param, scale=inb_sigma_param)

        # Output layer weight distribution priors
        outw_mu = torch.randn_like(self.net.net[-1].weight, device=self.net.device)
        outw_sigma = torch.randn_like(self.net.net[-1].weight, device=self.net.device)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = self.softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = pyro.distributions.Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)

        # Output layer bias distribution priors
        outb_mu = torch.randn_like(self.net.net[-1].bias, device=self.net.device)
        outb_sigma = torch.randn_like(self.net.net[-1].bias, device=self.net.device)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = self.softplus(pyro.param("outb_sigma", outb_sigma))

        outb_prior = pyro.distributions.Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {'net[0].weight': inw_prior, 'net[0].bias': inb_prior, 'net[-1].weight': outw_prior, 'net[-1].bias': outb_prior}

        lifted_module = pyro.random_module("module", self.net, priors)

        return lifted_module()

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
        inputs, _ = data
        inputs = inputs.to(self.net.device)

        sampled_models = [self.pyro_guide(None, None) for _ in range(CONFIG.NUMBER_OF_PROBABILISTIC_MODELS)]
        # A tensor of tensors representing the whole batch of predictions for each "model" pyro built
        out_hats = [model(inputs).data for model in sampled_models]
        outputs_mean = torch.mean(torch.stack(out_hats), 0)


        '''
                histo_exp = []

            print(output_vectors)
            for i in range(len(outputs_mean[0])):
                for j in range(len(output_vectors)):
                    histo_exp.append(np.exp(output_vectors[i][j]))
                meadian_probability = np.percentile(histo_exp, 50)
                print(meadian_probability)
                print(histo_exp)

        '''
        predictions_arr_dict = {}
        for outputs in out_hats:
            output_trasnformed_vectors = {}

            for output_vector in outputs:
                transformed_output_vectors = when_data_source.transformer.revert(output_vector,feature_set = 'output_features')
                for feature in transformed_output_vectors:
                    if feature not in output_trasnformed_vectors:
                        output_trasnformed_vectors[feature] = []
                    output_trasnformed_vectors[feature] += [transformed_output_vectors[feature]]

            for output_column in output_trasnformed_vectors:

                decoded_predictions = when_data_source.get_decoded_column_data(output_column, when_data_source.encoders[output_column]._pytorch_wrapper(output_trasnformed_vectors[output_column]))

                if output_column not in predictions_arr_dict:
                    predictions_arr_dict[output_column] = []

                predictions_arr_dict[output_column].append(decoded_predictions)

        predictions_dict = {}
        for output_column in predictions_arr_dict:
            final_predictions = []
            final_predictions_confidence = []
            possible_predictions = []
            possible_predictions_confidence = []


            predictions_arr = predictions_arr_dict[output_column]

            predictions_arr_r = [[]] * len(predictions_arr[0])
            for i in range(len(predictions_arr)):
                for j in range(len(predictions_arr[i])):
                    predictions_arr_r[j].append(predictions_arr[i][j])

            for predictions in predictions_arr_r:
                if when_data_source.get_column_config(output_column)['type'] == 'categorical':
                    occurance_dict = dict(Counter(predictions))
                    final_prediction = max(occurance_dict, key=occurance_dict.get)
                    final_predictions.append(final_prediction)
                    final_predictions_confidence.append(occurance_dict[final_prediction]/len(predictions))

                    possible_predictions.append(list(occurance_dict.keys()))
                    possible_predictions_confidence.append([x/len(predictions) for x in list(occurance_dict.values())])
                else:
                    predictions = [x if x is not None else 0 for x in predictions]
                    p_hist = np.histogram(np.array(predictions),10)
                    final_predictions_confidence.append(max(p_hist[0])/sum(p_hist[0]))
                    final_predictions.append(p_hist[1][np.where(p_hist[0] == max(p_hist[0]))][0])
                    possible_predictions_confidence.append([x/sum(p_hist[0]) for x in p_hist[0]])
                    possible_predictions.append(list(p_hist[1]))


            predictions_dict[output_column] = {}
            predictions_dict[output_column]['predictions'] = final_predictions
            predictions_dict[output_column]['confidence'] = final_predictions_confidence
            predictions_dict[output_column]['possible_predictions'] = possible_predictions
            predictions_dict[output_column]['possible_predictions_confidence'] = possible_predictions_confidence

        logging.info('Model predictions and decoding completed')

        return predictions_dict

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

            sampled_models = [self.pyro_guide(None, None) for _ in range(CONFIG.NUMBER_OF_PROBABILISTIC_MODELS)]
            out_hats = [model(inputs).data for model in sampled_models]
            outputs_mean = torch.mean(torch.stack(out_hats), 0)

            loss = self.criterion(outputs_mean, labels)
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
        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.net = self.nn_class(ds, self.dynamic_parameters)

        #self.net.train()


        if self.criterion is None:
            if self.is_categorical_output:
                if ds.output_weights is not None and ds.output_weights is not False:
                    output_weights = torch.Tensor(ds.output_weights).to(self.net.device)
                else:
                    output_weights = None
                self.criterion = torch.nn.CrossEntropyLoss(weight=output_weights)
            else:
                self.criterion = torch.nn.MSELoss()

        #self.optimizer = pyro.optim.Adadelta({"lr": 0.1})
        self.optimizer = pyro.optim.Adam({"lr": 0.005, "betas": (0.95, 0.999)})
        svi = pyro.infer.SVI(self.pyro_model, self.pyro_guide, self.optimizer, loss=pyro.infer.Trace_ELBO())

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

                #mi = inputs.view(8,len(self.net.net[0].bias))
                mi = inputs
                running_loss += svi.step(mi[0], labels)
                error = running_loss / (i + 1)

            yield error
