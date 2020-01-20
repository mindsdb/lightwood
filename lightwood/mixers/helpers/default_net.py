import logging
from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.shapes import *
from lightwood.mixers.helpers.plinear import PLinear
import torch



class DefaultNet(torch.nn.Module):

    def __init__(self, ds, dynamic_parameters, shape=None, selfaware=False, size_parameters={}, pretrained_net=None):
        self.input_size = None
        self.output_size = None
        self.selfaware = selfaware
        # How many devices we can train this network on
        self.available_devices = 1
        self.max_variance = None

        device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE
        self.device = torch.device(device_str)

        if CONFIG.DETERMINISTIC:
            '''
                Seed that always has the same value on the same dataset plus setting the bellow CUDA options
                In order to make sure pytroch randomly generate number will be the same every time when training on the same dataset
            '''
            if ds is not None:
                torch.manual_seed(len(ds))
            else:
                torch.manual_seed(2)

            if device_str == 'cuda':
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.available_devices = torch.cuda.device_count()


        self.dynamic_parameters = dynamic_parameters
        """
        Here we define the basic building blocks of our model, in forward we define how we put it all together along wiht an input
        :param sample_batch: this is used to understand the characteristics of the input and target, it is an object of type utils.libs.data_types.batch.Batch
        """
        super(DefaultNet, self).__init__()

        if shape is None and pretrained_net is None:
            input_sample, output_sample = ds[0]

            self.input_size = len(input_sample)
            self.output_size = len(output_sample)

            '''
            small_input = True if self.input_size < 50 else False
            small_output = True if self.input_size < 50 else False
            large_input = True if self.input_size > 2000 else False
            large_output = True if self.output_size > 2000 else False

            # 2. Determine in/out proportions
            # @TODO: Maybe provide a warning if the output is larger, this really shouldn't usually be the case (outside of very specific things, such as text to image)
            larger_output = True if self.output_size > self.input_size*2 else False
            larger_input = True if self.input_size > self.output_size*2 else False
            even_input_output = larger_input and large_output

            if 'network_depth' in self.dynamic_parameters:
                depth = self.dynamic_parameters['network_depth']
            else:
                depth = 5

            if (small_input and small_output):
                shape = rombus(self.input_size,self.output_size,depth+1,800)
            elif (not large_input) and (not large_output):
                shape = rombus(self.input_size,self.output_size,depth,self.input_size*2)
            elif large_input and large_output:
                shape = rectangle(self.input_size,self.output_size,depth - 1)
            else:
                shape = funnel(self.input_size,self.output_size,depth)
            '''
            shape = [self.input_size, max([self.input_size*2,self.output_size*2,400]), self.output_size]

        if pretrained_net is None:
            logging.info(f'Building network of shape: {shape}')
            rectifier = torch.nn.SELU  #alternative: torch.nn.ReLU

            layers = []
            for ind in range(len(shape) - 1):
                linear_function = PLinear  if CONFIG.USE_PROBABILISTIC_LINEAR else torch.nn.Linear
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
            awareness_net_shape = funnel(self.input_size + self.output_size, self.output_size, 4)
            awareness_layers = []

            for ind in range(len(awareness_net_shape) - 1):
                awareness_layers.append(torch.nn.Linear(awareness_net_shape[ind],awareness_net_shape[ind+1]))
                if ind < len(awareness_layers) - 2:
                    awareness_layers.append(rectifier())

            self.awareness_net = torch.nn.Sequential(*awareness_layers)

        if CONFIG.DETERMINISTIC and pretrained_net is None: # set initial weights based on a specific distribution if we have deterministic enabled

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

                count +=1
                mean_variance += torch.mean(layer.sigma).tolist()

        if count == 0:
            return -1 # Unknown

        mean_variance = mean_variance / count
        self.max_variance = mean_variance if self.max_variance is None else mean_variance if self.max_variance < mean_variance else self.max_variance

        return (self.max_variance- mean_variance)/self.max_variance

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
