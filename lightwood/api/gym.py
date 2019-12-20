import torch



class Gym():

    def __init__(self, model, optimizer, scheduler, loss, extra={}):
        """
        Create an environment for training a pytroch machine learning model
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

    def iter_fit(train_data, test_data, desired_error, error_func, max_time, callback, extra={}):
        pass
