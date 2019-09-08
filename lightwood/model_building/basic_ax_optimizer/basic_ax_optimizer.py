import ax

class BasicAxOptimizer:
    def __init__(self):
        self.total_trials = 32

    def evaluate(self, error_yielding_function):
        best_parameters, values, experiment, model = ax.optimize(
            parameters=[
                {'name': 'beta1', 'type': 'range', 'bounds': [0.90,0.95]},
                {'name': 'lr', 'type': 'range', 'bounds': [0.001, 0.01]},
                {'name': 'N_sma_threshold', 'type': 'choice', 'values': [4,5]},
                {'name': 'k', 'type': 'choice', 'values': [6,12]},
            ],
            evaluation_function=error_yielding_function,
            objective_name='accuracy',
            total_trials = self.total_trials,
        )

        return best_parameters
