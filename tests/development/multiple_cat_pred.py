from sklearn.metrics import accuracy_score, f1_score

import lightwood
import pandas as pd

from lightwood import Predictor
from lightwood.encoders.text import ShortTextEncoder
from lightwood.encoders.categorical import MultiHotEncoder

lightwood.config.config.CONFIG.USE_CUDA = True

df = pd.read_csv("../../../datasets/mpst/mpst_full_data.csv")

df = df[['title', 'plot_synopsis', 'tags', 'split']]
df.tags = df.tags.apply(lambda x: x.split(', '))

df_train = df[df.split == 'train'].copy()
df_test = df[df.split == 'test'].copy()
df_train = df_train.drop(['split'], axis=1)
df_test = df_test.drop(['split'], axis=1)

config = {'input_features': [
        {'name': 'plot_synopsis',
         'type': 'text',
         },
    ],
    'output_features': [
        {'name': 'tags', 'type': 'multiple_categorical', 'encoder_class': MultiHotEncoder}
    ],
}
predictor = Predictor(config)


def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
    print(
        'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}'.format(
            iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
            accuracy=predictor.train_accuracy, test_accuracy=test_accuracy))

predictor.learn(from_data=df_train,
                callback_on_iter=iter_function,
                eval_every_x_epochs=4)

predictions = predictor.predict(when_data=df_train)
train_tags = df_train.tags
predicted_tags = predictions['tags']['predictions']
train_tags_encoded = predictor._mixer.encoders['tags'].encode(train_tags)
pred_labels_encoded = predictor._mixer.encoders['tags'].encode(predicted_tags)
score = f1_score(train_tags_encoded, pred_labels_encoded, average='weighted')
print('Train f1 score', score)

# Why does it try to encode the missing column tags?
#predictions = predictor.predict(when_data=df_test.drop(['tags'], axis=1))
predictions = predictor.predict(when_data=df_test)

test_tags = df_test.tags
predicted_tags = predictions['tags']['predictions']

test_tags_encoded = predictor._mixer.encoders['tags'].encode(test_tags)
pred_labels_encoded = predictor._mixer.encoders['tags'].encode(predicted_tags)
score = f1_score(test_tags_encoded, pred_labels_encoded, average='weighted')
print('Test f1 score', score)