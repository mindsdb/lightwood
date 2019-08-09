from lightwood.config.config import CONFIG
import torch.nn as nn
import torch



class DefaultNet(nn.Module):

    def __init__(self, ds):
        if CONFIG.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
        Here we define the basic building blocks of our model, in forward we define how we put it all together along wiht an input
        :param sample_batch: this is used to understand the characteristics of the input and target, it is an object of type utils.libs.data_types.batch.Batch
        """
        super(DefaultNet, self).__init__()
        input_sample, output_sample = ds[0]
        input_size = len(input_sample)
        output_size = len(output_sample)

        if input_size < 3 * pow(10,3) and True:
            self.net = nn.Sequential(
                nn.Linear(input_size, 2*input_size),
                nn.SELU(),
                nn.Linear(2*input_size, 4*input_size),
                nn.SELU(),
                nn.Linear(4*input_size, 8*input_size),
                nn.SELU(),
                nn.Linear(8*input_size, 4*input_size),
                nn.SELU(),
                nn.Linear(4*input_size, 2*input_size),
                nn.SELU(),
                nn.Linear(2*input_size, output_size)
            )
        else:
            deep_layer_in = 128 * 4
            deep_layer_out = round(min(deep_layer_in,output_size*2))
            self.net = nn.Sequential(
                nn.Linear(input_size, deep_layer_in),
                nn.SELU(),
                nn.Linear(deep_layer_in, deep_layer_out),
                nn.SELU(),
                nn.Linear(deep_layer_out, output_size)
            )


        self.net = self.net.to(self.device)


    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return:
        """

        output = self.net(input)
        return output
