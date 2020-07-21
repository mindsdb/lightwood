"""
This example requires the "MPST: Movie Plot Synopses with Tags" dataset which can be obtained from Kaggle:
https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags#mpst_full_data.csv

The goal is to predict movie tags given a plot synopsis.
Tags include "horror", "comedy", "romantic" and others.
Each movie can have multiple tags to it, so this problems falls into the multi-label classification domain.
"""
from sklearn.metrics import f1_score, multilabel_confusion_matrix
import lightwood
import pandas as pd


def train_model(df_train):
    # A configuration describing the contents of the dataframe, what are the targets we want to predict and what are the features we want to use
    config = {
        'input_features': [
            {'name': 'plot_synopsis',
             'type': 'text'},
        ],
        'output_features': [
            {'name': 'tags', 'type': 'multiple_categorical'}
        ],
    }

    # Callback to log various training stats (currently the only hook into the training process)
    def train_callback(epoch, error, test_error, test_error_gradient, test_accuracy):
        print(f'We reached epoch {epoch} with error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}')

    # The actual training process
    predictor = lightwood.Predictor(config)
    print('Starting model training')
    predictor.learn(from_data=df,
                    callback_on_iter=train_callback,
                    eval_every_x_epochs=5)
    # Save the lightwood model
    predictor.save('lightwood_model.dill')


def test_model(df_test):
    print('Testing model')
    # Load some testing data and extract the real values for the target column
    predictor = lightwood.Predictor(load_from_path='lightwood_model.dill')
    predictions = predictor.predict(when_data=df_test)

    test_tags = df_test.tags
    predicted_tags = predictions['tags']['predictions']

    # We will use an internal encoder to convert the tags to binary vectors
    # This allows us to evaluate the F1 score measure
    # It evaluates how good the model is at predicting correct tags and avoiding false positives, while staying robust to class imbalances
    test_tags_encoded = predictor._mixer.encoders['tags'].encode(test_tags)
    pred_tags_encoded = predictor._mixer.encoders['tags'].encode(predicted_tags)
    score = f1_score(test_tags_encoded, pred_tags_encoded, average='weighted')

    # An f1 score of around 0.2 is expected for this dataset
    # Mind that such score is expected if applying manual text preprocessing, which we don't do in this example
    print('Test f1_score', round(score, 4))


# Run as main
if __name__ == '__main__':
    df = pd.read_csv("mpst_full_data.csv")

    # For simplicity we will try to predict the tags only from the synopsis
    # Other columns can be included for a better performance
    df = df[['plot_synopsis', 'tags', 'split']]

    # Split the tags into an array
    df.tags = df.tags.apply(lambda x: x.split(', '))

    # Select training data only
    df_train = df[df.split == 'train'].copy()
    df_train = df_train.drop(['split'], axis=1)

    df_test = df[df.split == 'test'].copy()
    df_test = df_test.drop(['split'], axis=1)

    train_model(df_train)
    test_model(df_test)
