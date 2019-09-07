import ax

class BasicAxOptimizer:
    def __init__(self):
        self.total_trials = 40

    def evaluate(self, error_yielding_function):


        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.
        # @TODO Implement the above testing with AX ^
        # lr=1e-3

        best_parameters, values, experiment, model = ax.optimize(
            parameters=[
                {'name': 'base_lr', 'type': 'range', 'bounds': [0.0003,0.003]}, # , 'log_scale': True ?
                {'name': 'max_lr', 'type': 'range', 'bounds': [0.005,0.02]},
                #{'name': 'network_depth', 'type': 'choice', 'values': [5,6]},
                #{'name': 'scheduler_mode', 'type': 'choice', 'values': ['triangular', 'triangular2', 'exp_range']},
                {'name': 'weight_decay', 'type': 'range', 'bounds': [0.00005, 0.0003]},
            ],
            evaluation_function=error_yielding_function,
            objective_name='accuracy',
            total_trials = self.total_trials,
        )

        return best_parameters
