import ax

class BasicAxOptimizer:
    def __init__(self):
        self.total_trials = 40

    def evaluate(self, error_yielding_function):

        best_parameters, values, experiment, model = ax.optimize(
            parameters=[
                {'name': 'base_lr', 'type': 'range', 'bounds': [3 * 1e-4,3 * 1e-3]}, # , 'log_scale': True ?
                {'name': 'max_lr', 'type': 'range', 'bounds': [5 * 1e-3,5 * 1e-2]},
                {'name': 'network_depth', 'type': 'choice', 'values': [5,6]},
                {'name': 'scheduler_mode', 'type': 'choice', 'values': ['triangular', 'triangular2', 'exp_range']},
                {'name': 'weight_decay', 'type': 'range', 'bounds': [6 * 1e-3, 4 * 1e-2]},
            ],
            evaluation_function=error_yielding_function,
            objective_name='accuracy',
            total_trials = self.total_trials,
        )

        print(best_parameters, values, experiment, model)

        return best_parameters
