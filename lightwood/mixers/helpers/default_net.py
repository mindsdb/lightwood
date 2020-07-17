import torch

from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.shapes import *
from lightwood.mixers.helpers.plinear import PLinear
from lightwood.helpers.device import get_devices


class DefaultNet(torch.nn.Module):

    def __init__(self, dynamic_parameters,
                     input_size=None,
                     output_size=None,
                     nr_outputs=None,
                     shape=None,
                     selfaware=False,
                     size_parameters={},
                     pretrained_net=None,
                     deterministic=False):
        self.input_size = input_size
        self.output_size = output_size
        self.nr_outputs = nr_outputs
        self.selfaware = selfaware
        # How many devices we can train this network on
        self.available_devices = 1
        self.max_variance = None

        self.device, _ = get_devices()

        if deterministic:
            '''
                Seed that always has the same value on the same dataset plus setting the bellow CUDA options
                In order to make sure pytorch randomly generate number will be the same every time
                when training on the same dataset
            '''
            torch.manual_seed(66)

            if 'cuda' in str(self.device):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.available_devices = torch.cuda.device_count()
            else:
                self.available_devices = 1

        self.dynamic_parameters = dynamic_parameters

        """
        Here we define the basic building blocks of our model,
        in forward we define how we put it all together along with an input
        """
        super(DefaultNet, self).__init__()

        if shape is None and pretrained_net is None:
            shape = [self.input_size, max([self.input_size*2,self.output_size*2,400]), self.output_size]

        if pretrained_net is None:
            logging.info(f'Building network of shape: {shape}')
            rectifier = torch.nn.SELU  #alternative: torch.nn.ReLU

            layers = []
            for ind in range(len(shape) - 1):
                linear_function = PLinear if CONFIG.USE_PROBABILISTIC_LINEAR else torch.nn.Linear
                layers.append(linear_function(shape[ind],shape[ind+1]))
                if ind < len(shape) - 2:
                    layers.append(rectifier())

            self.net = torch.nn.Sequential(*layers)
        else:
            self.net = pretrained_net
            for layer in self.net:
                if isinstance(layer, torch.nn.Linear):
                    if self.input_size is None:
                        self.input_size = layer.in_features
                    self.output_size = layer.out_features

        if self.selfaware:
            awareness_net_shape = [(self.input_size + self.output_size), max([int((self.input_size + self.output_size) * 1.5), 300]), self.nr_outputs]
            awareness_layers = []


            for ind in range(len(awareness_net_shape) - 1):
                rectifier = torch.nn.SELU  #alternative: torch.nn.ReLU
                awareness_layers.append(torch.nn.Linear(awareness_net_shape[ind], awareness_net_shape[ind + 1]))
                if ind < len(awareness_net_shape) - 2:
                    awareness_layers.append(rectifier())

            self.awareness_net = torch.nn.Sequential(*awareness_layers)

        if deterministic and pretrained_net is None: # set initial weights based on a specific distribution if we have deterministic enabled
            # lambda function so that we can do this for either awareness layer or the internal layers of net
            def reset_layer_params(layer):
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0., std=1 / math.sqrt(layer.out_features))
                    torch.nn.init.normal_(layer.bias, mean=0., std=0.1)

                elif isinstance(layer, PLinear):
                    torch.nn.init.normal_(layer.mean, mean=0., std=1 / math.sqrt(layer.out_features))
                    torch.nn.init.normal_(layer.bias, mean=0., std=0.1)

            if self.selfaware:
                for layer in self.awareness_net:
                    reset_layer_params(layer)

            for layer in self.net:
                reset_layer_params(layer)

        self.net = self.net.to(self.device)
        if self.available_devices > 1:
            self._foward_net = torch.nn.DataParallel(self.net)
        else:
            self._foward_net = self.net

        if self.selfaware:
            self.awareness_net = self.awareness_net.to(self.device)
            if self.available_devices > 1:
                self._foward_awareness_net = torch.nn.DataParallel(self.awareness_net)
            else:
                self._foward_awareness_net = self.awareness_net

    def to(self, device=None, available_devices=None):
        if device is None or available_devices is None:
            device, available_devices = get_devices()

        self.net = self.net.to(device)
        if self.selfaware:
            self.awareness_net = self.awareness_net.to(device)

        available_devices = 1
        if 'cuda' in str(device):
            available_devices = torch.cuda.device_count()

        if available_devices > 1:
            self._foward_net = torch.nn.DataParallel(self.net)
            if self.selfaware:
                self._foward_awareness_net = torch.nn.DataParallel(self.awareness_net)
        else:
            self._foward_net = self.net
            if self.selfaware:
                self._foward_awareness_net = self.awareness_net

        self.device = device
        self.available_devices = available_devices

        return self


    def calculate_overall_certainty(self):
        """
        Calculate overall certainty of the model
        :return: -1 means its unknown as it is using non probabilistic layers
        """
        mean_variance = 0
        count = 0

        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                continue
            elif isinstance(layer, PLinear):

                count += 1
                mean_variance += torch.mean(layer.sigma).tolist()

        if count == 0:
            return -1  # Unknown

        mean_variance = mean_variance / count
        self.max_variance = mean_variance if self.max_variance is None \
            else mean_variance if self.max_variance < mean_variance else self.max_variance

        return (self.max_variance - mean_variance) / self.max_variance

    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input

        :param input: a pytorch tensor with the input data of a batch
        :param return_awareness: This tells if we should return the awareness output

        :return: either just output or (output, awareness)
        """

        output = self._foward_net(input)

        if self.selfaware:
            interim = torch.cat((input, output), 1)
            awareness = self._foward_awareness_net(interim)

            return output, awareness

        return output
