import pandas
import random
import lightwood
from lightwood import Predictor
import os


lightwood.config.config.CONFIG.HELPER_MIXERS = False

### Generate a dataset
n = 1000
op = '*'

# generate random numbers between -10 and 10
data_train = {'x': [random.randint(-15, 5) for i in range(n)],
        'y': [random.randint(-15, 5) for i in range(n)]}

data_test = {'x': [random.randint(-5, 15) for i in range(n)],
        'y': [random.randint(-5, 15) for i in range(n)]}

if op == '/':
    for i in range(n):
        if data_train['y'][i] == 0:
            data_train['y'][i] = 1
        if data_test['y'][i] == 0:
            data_test['y'][i] = 1

# target variable to be the multiplication of the two
data_train['z'] = eval(f"""[data_train['x'][i] {op} data_train['y'][i] for i in range(n)]""")
data_test['z'] = eval(f"""[data_test['x'][i] {op} data_test['y'][i] for i in range(n)]""")


df_train = pandas.DataFrame(data_train)
df_test = pandas.DataFrame(data_test)

predictor = Predictor(output=['z'])

def iter_function(epoch, training_error, test_error, test_error_gradient, test_accuracy):
    print(f'Epoch: {epoch}, Train Error: {training_error}, Test Error: {test_error}, Test Error Gradient: {test_error_gradient}, Test Accuracy: {test_accuracy}')

predictor.learn(from_data=df_train, callback_on_iter=iter_function, eval_every_x_epochs=20)
predictor.save('ok.pkl')

predictor = Predictor(load_from_path='ok.pkl')

'''
when = {'x': [0, 0, 1, -1, 1], 'y': [0, 1, -1, -1, 1]}
pred = predictor.predict(when_data=pandas.DataFrame(when))['z']['predictions']
print('Real values: ' + eval(f"""str([when['x'][i] {op} when['y'][i] for i in range(len(when['x']))])"""))
print('Pred values: ' + str(pred))

when = {'x': [0, 3, 1, -5, 1], 'y': [0, 1, -5, -4, 7]}
pred = predictor.predict(when_data=pandas.DataFrame(when))['z']['predictions']
print('Real values: ' + eval(f"""str([when['x'][i] {op} when['y'][i] for i in range(len(when['x']))])"""))
print('Pred values: ' + str(pred))
'''

print('Train accuracy: ', predictor.train_accuracy)
print('Test accuracy: ', predictor.calculate_accuracy(from_data=df_test))

predictions = predictor.predict(when_data=df_test)
print(list(df_test['z'])[30:60])
print(predictions['z']['predictions'][30:60])
