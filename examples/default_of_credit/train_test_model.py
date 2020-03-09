from sklearn.metrics import balanced_accuracy_score, accuracy_score
import lightwood
import pandas as pd


def train_model():
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
        'output_features': [{'name': 'default.payment.next.month', 'type': 'categorical', }],
        'mixer': {'class': lightwood.BUILTIN_MIXERS.NnMixer}}

    df = pd.read_csv('dataset/train.csv')

    def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
        print(
            'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}'.format(
                iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
                accuracy=predictor.train_accuracy, test_accuracy=test_accuracy))

    predictor = lightwood.Predictor(config)
    predictor.learn(from_data=df, callback_on_iter=iter_function, eval_every_x_epochs=1, stop_training_after_seconds=1)
    predictor.save('test.pkl')


def test_model():
    test = pd.read_csv('dataset/test.csv')
    actual = [str(x) for x in test['default.payment.next.month']]

    predictor = lightwood.Predictor(load_from_path='test.pkl')
    predictions = predictor.predict(when_data=test)
    predicted = [str(x) for x in predictions['default.payment.next.month']['predictions']]

    balanced_accuracy = balanced_accuracy_score(actual, predicted)
    print(f'Balacned accuracy score of {balanced_accuracy}')
    accuracy = accuracy_score(list(actual), list(predicted))
    print(f'accuracy score of {accuracy}')


# Run as main
if __name__ == '__main__':
    train_model()
    test_model()
