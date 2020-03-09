from sklearn.metrics import balanced_accuracy_score, accuracy_score
import lightwood
import pandas as pd


def train_model():
    # Load some training data (default on credit, for predicting whether or not someone will default on their credit)
    df = pd.read_csv('https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/default_of_credit/dataset/train.csv')

    # A configuration describing the contents of the dataframe, what are the targets we want to predict and what are the features we want to use
    # Note: the `weights` for the output column `default.payment.next.month`, since the number of samples is uneven between the two categories, but we care about balanced accuracy rather than overall accuracy
    config = {'input_features': [
        {'name': 'ID', 'type': 'numeric'}, {'name': 'LIMIT_BAL', 'type': 'numeric'},
        {'name': 'SEX', 'type': 'categorical'}, {'name': 'EDUCATION', 'type': 'categorical'},
        {'name': 'MARRIAGE', 'type': 'categorical'}, {'name': 'AGE', 'type': 'numeric'},
        {'name': 'PAY_0', 'type': 'numeric'}, {'name': 'PAY_2', 'type': 'numeric'},
        {'name': 'PAY_3', 'type': 'numeric'}, {'name': 'PAY_4', 'type': 'numeric'},
        {'name': 'PAY_5', 'type': 'numeric'}, {'name': 'PAY_6', 'type': 'numeric'},
        {'name': 'BILL_AMT1', 'type': 'numeric'}, {'name': 'BILL_AMT2', 'type': 'numeric'},
        {'name': 'BILL_AMT3', 'type': 'numeric'}, {'name': 'BILL_AMT4', 'type': 'numeric'},
        {'name': 'BILL_AMT5', 'type': 'numeric'}, {'name': 'BILL_AMT6', 'type': 'numeric'},
        {'name': 'PAY_AMT1', 'type': 'numeric'}, {'name': 'PAY_AMT2', 'type': 'numeric'},
        {'name': 'PAY_AMT3', 'type': 'numeric'}, {'name': 'PAY_AMT4', 'type': 'numeric'},
        {'name': 'PAY_AMT5', 'type': 'numeric'}, {'name': 'PAY_AMT6', 'type': 'numeric'}],
        'output_features': [{'name': 'default.payment.next.month', 'type': 'categorical', 'weights': {'0': 0.3, '1': 1}}],
        'mixer': {'class': lightwood.BUILTIN_MIXERS.NnMixer}}

    # Callback to log various training stats (currently the only hook into the training process)
    def train_callback(epoch, error, test_error, test_error_gradient, test_accuracy):
        print(f'We reached epoch {epoch} with error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}')

    # The actual training process
    predictor = lightwood.Predictor(config)
    # Note: If `stop_training_after_seconds` is not set, training will stop automatically once we determine the model is overfitting (we separate a testing and a training dataset internally from the dataframe given and only train on the training one, using the testing one to determine overfitting, pick the best model and evaluate model accuracy)
    predictor.learn(from_data=df, callback_on_iter=train_callback, eval_every_x_epochs=5, stop_training_after_seconds=100)

    # Save the lightwood model
    predictor.save('lightwood_model.dill')


def test_model():
    # Load some testing data and extract the real values for the target column
    test = pd.read_csv('https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/default_of_credit/dataset/test.csv')
    real = [str(x) for x in test['default.payment.next.month']]

    test = test.drop(columns=['default.payment.next.month'])

    # Load the lightwood model from where we previously saved it and predict using it
    predictor = lightwood.Predictor(load_from_path='lightwood_model.dill')
    predictions = predictor.predict(when_data=test)
    predicted = [str(x) for x in predictions['default.payment.next.month']['predictions']]

    # Get the balanced accuracy score to see how well we did (in this case > 50% means better than random)
    balanced_accuracy_pct = balanced_accuracy_score(real, predicted) * 100
    print(f'Balacned accuracy score of {round(balanced_accuracy_pct,1)}%')


# Run as main
if __name__ == '__main__':
    train_model()
    test_model()
