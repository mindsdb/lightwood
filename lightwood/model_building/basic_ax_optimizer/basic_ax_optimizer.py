import ax

class BasicAxOptimizer:
    def __init__(self):
        self.total_trials = 40

    def evaluate(self, error_yielding_function):

        best_parameters, values, experiment, model = ax.optimize(
            parameters=[
                {'name': 'base_lr', 'type': 'range', 'bounds': [0.0003,0.003]}, # , 'log_scale': True ?
                {'name': 'max_lr', 'type': 'range', 'bounds': [0.005,0.02]},
                #{'name': 'network_depth', 'type': 'choice', 'values': [5,6]},
                #{'name': 'scheduler_mode', 'type': 'choice', 'values': ['triangular', 'triangular2', 'exp_range']},
                {'name': 'weight_decay', 'type': 'range', 'bounds': [0.001, 0.01]},
            ],
            evaluation_function=error_yielding_function,
            objective_name='accuracy',
            total_trials = self.total_trials,
        )

        return best_parameters
