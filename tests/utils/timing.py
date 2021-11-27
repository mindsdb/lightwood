from lightwood.api.predictor import PredictorInterface
import time
import pandas as pd


def train_and_check_time_aim(predictor: PredictorInterface, train_df: pd.DataFrame):
    """
    Trains the predictor with the desired data *and* checks that the time aim we set for it was respected.
    """
    time_aim_expected = predictor.problem_definition.time_aim
    start = time.time()
    predictor.learn(train_df)
    time_aim_actual = (time.time() - start)
    # Current margin of error for time aim is 2x, but we should aim to lower this
    if time_aim_expected is not None:
        if((time_aim_expected * 2.5) < time_aim_actual):
            error = f'time_aim is set to {time_aim_expected} seconds, however learning took {time_aim_actual}'
            raise ValueError(error)
