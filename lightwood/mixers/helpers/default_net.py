import logging
from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.shapes import *
from lightwood.mixers.helpers.plinear import PLinear
import torch



class DefaultNet(torch.nn.Module):

    def __init__(self, ds, dynamic_parameters):
        device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE

        if CONFIG.DETERMINISTIC:
            '''
                Seed that always has the same value on the same dataset plus setting the bellow CUDA options
                In order to make sure pytroch randomly generate number will be the same every time when training on the same dataset
            '''
            torch.manual_seed(len(ds))
            if device_str == 'cuda':
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False


        self.device = torch.device(device_str)

        self.dynamic_parameters = dynamic_parameters
        """
        Here we define the basic building blocks of our model, in forward we define how we put it all together along wiht an input
        :param sample_batch: this is used to understand the characteristics of the input and target, it is an object of type utils.libs.data_types.batch.Batch
        """
        super(DefaultNet, self).__init__()
        input_sample, output_sample = ds[0]

        self.input_size = len(input_sample)
        self.output_size = len(output_sample)

        # Select architecture

        # 1. Determine, based on the machines specification, if the input/output size are "large"
        if CONFIG.USE_CUDA or CONFIG.USE_DEVICE is not None:
            large_input = True if self.input_size > 4000 else False
            large_output = True if self.output_size > 400 else False
        else:
            large_input = True if self.input_size > 1000 else False
            large_output = True if self.output_size > 100 else False

        # 2. Determine in/out proportions
        # @TODO: Maybe provide a warning if the output is larger, this really shouldn't usually be the case (outside of very specific things, such as text to image)
        larger_output = True if self.output_size > self.input_size*2 else False
        larger_input = True if self.input_size > self.output_size*2 else False
        even_input_output = larger_input and large_output

        if 'network_depth' in self.dynamic_parameters:
            depth = self.dynamic_parameters['network_depth']
        else:
            depth = 5

        if (not large_input) and (not large_output):
            shape = rombus(self.input_size,self.output_size,depth,self.input_size*2)

        elif (not large_output) and large_input:
            shape = funnel(self.input_size,self.output_size,depth)

        elif (not large_input) and large_output:
            shape = rectangle(self.input_size,self.output_size,depth - 1)
        else:
            shape = rectangle(self.input_size,self.output_size,depth - 2)


        logging.info(f'Building network of shape: {shape}')
        rectifier = torch.nn.SELU  #alternative: torch.nn.ReLU

        layers = []
        for ind in range(len(shape) - 1):
            linear_function = PLinear  if CONFIG.USE_PROBABILISTIC_LINEAR else torch.nn.Linear
            layers.append(linear_function(shape[ind],shape[ind+1]))
            if ind < len(shape) - 2:
                layers.append(rectifier())


        self.net = torch.nn.Sequential(*layers)

        if CONFIG.DETERMINISTIC: # set initial weights based on a specific distribution if we have deterministic enabled
            for layer in self.net:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0., std=1 / math.sqrt(layer.out_features))
                    torch.nn.init.normal_(layer.bias, mean=0., std=0.1)

                elif isinstance(layer, PLinear):
                    torch.nn.init.normal_(layer.mean, mean=0., std=1 / math.sqrt(layer.out_features))
                    torch.nn.init.normal_(layer.bias, mean=0., std=0.1)

        self.net = self.net.to(self.device)


    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return:
        """

        output = self.net(input)
        return output
