from lightwood.config.config import CONFIG
import torch.nn as nn



class DefaultNet(nn.Module):

    def __init__(self, ds):

        """
        Here we define the basic building blocks of our model, in forward we define how we put it all together along wiht an input
        :param sample_batch: this is used to understand the characteristics of the input and target, it is an object of type utils.libs.data_types.batch.Batch
        """
        super(DefaultNet, self).__init__()
        input_sample, output_sample = ds[0]
        input_size = len(input_sample)
        output_size = len(output_sample)

        self.net = nn.Sequential(

            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, output_size)

        )



        if CONFIG.USE_CUDA:
            self.net.cuda()



    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return:
        """

        if CONFIG.USE_CUDA:
            input.cuda()

        output = self.net(input)
        return output
