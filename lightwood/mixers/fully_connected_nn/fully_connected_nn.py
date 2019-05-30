
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import math

USE_CUDA = False

class FullyConnectedNet(nn.Module):

    def __init__(self, ds):

        """
        Here we define the basic building blocks of our model, in forward we define how we put it all together along wiht an input
        :param sample_batch: this is used to understand the characteristics of the input and target, it is an object of type utils.libs.data_types.batch.Batch
        """
        super(FullyConnectedNet, self).__init__()
        input_sample, output_sample = ds.__getitem__(0)
        input_size = len(input_sample)
        output_size = len(output_sample)

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            torch.nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size, int(math.ceil(input_size/2))),
            torch.nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(math.ceil(input_size/2)), output_size)
        )



        if USE_CUDA:
            self.net.cuda()



    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return:
        """

        if USE_CUDA:
            input.cuda()

        output = self.net(input)
        return output


class Transformation:

    def __init__(self, input_features, output_features):

        self.input_features = input_features
        self.output_features = output_features


    def __call__(self, sample):

        input_vector = []
        output_vector = []

        for input_feature in self.input_features:
            input_vector += sample['input_features'][input_feature].tolist()

        for output_feature in self.output_features:
            output_vector += sample['output_features'][output_feature].tolist()

        return torch.FloatTensor(input_vector),  torch.FloatTensor(output_vector)

class FullyConnectedNnMixer:

    def __init__(self, input_column_names=None, output_column_names=None):
        self.net = None
        self.criterion = nn.MSELoss()#CrossEntropyLoss()
        self.optimizer = None
        self.epochs = 2
        self.batch_size = 100
        self.input_column_names = None
        self.output_column_names = None
        self.data_loader = None

        pass

    def fit(self, ds= None, callback=None):
        ret = 0
        for i in self.iter_fit(ds):
            ret = i

        return ret

    def predict(self, when_data_source):
        """

        :param when_data_source:
        :return:
        """
        data_loader = DataLoader(ds, batch_size=len(when_data_source), shuffle=False, num_workers=1)
        data = next(iter(data_loader))
        inputs, labels = data
        outputs = self.net(inputs)
        return outputs

    def error(self, ds):
        """

        :param ds:
        :return:
        """
        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        running_loss = 0.0
        error = 0

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # print statistics
            running_loss += loss.item()
            error = running_loss / (i + 1)

        return error

    def iter_fit(self, ds):
        """

        :param ds:
        :return:
        """
        data_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.input_column_names = self.input_column_names if self.input_column_names is not None else ds.get_feature_names(
            'input_features')
        self.output_column_names = self.output_column_names if self.output_column_names is not None else ds.get_feature_names(
            'output_features')
        ds.transform = Transformation(self.input_column_names, self.output_column_names)

        self.net = FullyConnectedNet(ds)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(400):  # loop over the dataset multiple times
            running_loss = 0.0
            error = 0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                error = running_loss / (i + 1)

            yield error







if __name__ == "__main__":
    import random
    import pandas
    from lightwood.api.data_source import DataSource

    ###############
    # GENERATE DATA
    ###############

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            },
            {
                'name': 'y',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'categorical',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    ##For Classification
    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = ['low' if i < 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)

    # print(data_frame)

    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = FullyConnectedNnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(predict_input_ds)
    print(predictions)

    ##For Regression

    # GENERATE DATA
    ###############

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            },
            {
                'name': 'y',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = [i + 0.5 for i in range(10)]

    data_frame = pandas.DataFrame(data)
    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = FullyConnectedNnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    for i in  mixer.iter_fit(ds):
        print(i)

    predictions = mixer.predict(predict_input_ds)
    print(predictions)