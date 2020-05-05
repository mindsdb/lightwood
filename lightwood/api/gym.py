import copy
import time
import torch

import numpy as np


class Gym:

    def __init__(self, model, optimizer, scheduler, loss_criterion, device,
                 name=None, input_encoder=None, output_encoder=None):
        """
        Create an environment for training a pytorch machine learning model
        """
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_criterion = loss_criterion
        self.name = name
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder

        self.best_model = None

    def fit(self, train_data_loader, test_data_loader, desired_error, max_time, callback,
            eval_every_x_epochs=1, max_unimproving_models=10, custom_train_func=None, custom_test_func=None):
        started = time.time()
        epoch = 0
        lowest_test_error = None
        last_test_error = None
        test_error_delta_buff = []

        keep_training = True

        while keep_training:
            epoch += 1
            running_loss = 0.0
            error = 0
            self.model = self.model.train()
            for i, data in enumerate(train_data_loader, 0):
                if custom_train_func is None:
                    input, real = data

                    if self.input_encoder is not None:
                        input = self.input_encoder(input)
                    if self.output_encoder is not None:
                        real = self.output_encoder(real)

                    input = input.to(self.device)
                    real = real.to(self.device)

                    predicted = self.model(input)
                    loss = self.loss_criterion(predicted, real)
                    loss.backward()
                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                else:
                    loss = custom_train_func(self.model, data, self)

                running_loss += loss.item()
                error = running_loss / (i + 1)

            if epoch % eval_every_x_epochs == 0:
                if test_data_loader is not None:
                    test_running_loss = 0.0
                    test_error = 0
                    self.model = self.model.eval()
                    real_buff = []
                    predicted_buff = []

                    for i, data in enumerate(test_data_loader, 0):
                        if custom_test_func is None:
                            input, real = data

                            if self.input_encoder is not None:
                                input = self.input_encoder(input)
                            if self.output_encoder is not None:
                                real = self.output_encoder(real)

                            input = input.to(self.device)
                            real = real.to(self.device)

                            with torch.no_grad():
                                predicted = self.model(input)

                            real_buff.append(real.tolist())
                            predicted_buff.append(predicted.tolist())

                            loss = self.loss_criterion(predicted, real)
                        else:
                            with torch.no_grad():
                                loss = custom_test_func(self.model, data, self)

                        test_running_loss += loss.item()
                        test_error = test_running_loss / (i + 1)
                else:
                    test_error = error
                    real_buff = None
                    predicted_buff = None

                if lowest_test_error is None or test_error < lowest_test_error:
                    lowest_test_error = test_error
                    self.best_model = copy.deepcopy(self.model).to('cpu')

                if last_test_error is None:
                    test_error_delta_buff.append(0)
                else:
                    test_error_delta_buff.append(last_test_error - test_error)

                last_test_error = test_error

                if (time.time() - started) > max_time:
                    keep_training = False

                if lowest_test_error < desired_error:
                    keep_training = False

                if len(test_error_delta_buff) >= max_unimproving_models:
                    delta_mean = np.mean(test_error_delta_buff[-max_unimproving_models:])
                    if delta_mean <= 0:
                        keep_training = False

                callback(test_error, real_buff, predicted_buff)

        return self.best_model, lowest_test_error, int(time.time() - started)
