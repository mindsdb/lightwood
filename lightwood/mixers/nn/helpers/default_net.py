import logging
from lightwood.config.config import CONFIG
from .shapes import *
import torch



class DefaultNet(torch.nn.Module):

    def __init__(self, ds, dynamic_parameters):
        device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE

        self.device = torch.device(device_str)

        self.dynamic_parameters = dynamic_parameters
        """
        Here we define the basic building blocks of our model, in forward we define how we put it all together along wiht an input
        :param sample_batch: this is used to understand the characteristics of the input and target, it is an object of type utils.libs.data_types.batch.Batch
        """
        super(DefaultNet, self).__init__()
        input_sample, output_sample = ds[0]
        input_size = len(input_sample)
        output_size = len(output_sample)

        # Select architecture

        # 1. Determine, based on the machines specification, if the input/output size are "large"
        if CONFIG.USE_CUDA or CONFIG.USE_DEVICE is not None:
            large_input = True if input_size > 4000 else False
            large_output = True if output_size > 400 else False
        else:
            large_input = True if input_size > 1000 else False
            large_output = True if output_size > 100 else False

        # 2. Determine in/out proportions
        # @TODO: Maybe provide a warning if the output is larger, this really shouldn't usually be the case (outside of very specific things, such as text to image)
        larger_output = True if output_size > input_size*2 else False
        larger_input = True if input_size > output_size*2 else False
        even_input_output = larger_input and large_output

        if 'network_depth' in self.dynamic_parameters:
            depth = self.dynamic_parameters['network_depth']
        else:
            depth = 5

        # 3. Determine shpae based on the sizes & propotions
        if (not large_input) and (not large_output):
            if larger_input:
                shape = rombus(input_size,output_size,depth,input_size*2)
            else:
                shape = rectangle(input_size,output_size,depth - 1)

        elif (not large_output) and large_input:
            shape = funnel(input_size,output_size,depth)

        elif (not large_input) and large_output:
            shape = rectangle(input_size,output_size,depth - 1)

        else:
            shape = rectangle(input_size,output_size,depth - 2)

        logging.info(f'Building network of shape: {shape}')
        rectifier = torch.nn.SELU  #alternative: torch.nn.ReLU

        layers = []
        for ind in range(len(shape) - 1):
            layers.append(torch.nn.Linear(shape[ind],shape[ind+1]))
            if ind < len(shape) - 2:
                layers.append(rectifier())


        self.net = torch.nn.Sequential(*layers)

        self.net = self.net.to(self.device)


    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return:
        """

        output = self.net(input)
        return output
