import torch
import torch.nn as nn
import torchvision.models as models
from lightwood.helpers.device import get_devices
from lightwood.config.config import CONFIG


class ChannelPoolAdaptiveAvg1d(torch.nn.AdaptiveAvgPool1d):
    def forward(self, input):
        n, c, _, _ = input.size()
        input = input.view(n,c,1).permute(0,2,1)
        pooled =  torch.nn.functional.adaptive_avg_pool1d(input, self.output_size)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c)

class Img2Vec(nn.Module):

    def __init__(self, model, layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        super(Img2Vec, self).__init__()

        self.device, _ = get_devices()
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model = self._get_model_and_layer(model, layer)
        self.model = self.model.to(self.device).train()


    def to(self, device, available_devices):
        self.device = device
        self.model = self.model.to(self.device)
        return self

    def forward(self, image, batch=True):
        embedding = self.model(image.to(self.device))

        if self.model_name in ('resnext-50-small'):
            if batch:
                return embedding
            return embedding[0, :]
        else:
            if batch:
                return embedding[:, :, 0, 0]
            return embedding[0, :, 0, 0]


    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name == 'resnet-18':
            self.layer_output_size = 512
            model = torch.nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])

        elif model_name == 'resnext-50-small':
            self.layer_output_size = 512
            model = torch.nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-1] , ChannelPoolAdaptiveAvg1d(output_size=512))

        elif model_name == 'resnext-50':
            self.layer_output_size = 2048
            model = torch.nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-1])

        else:
            raise Exception(f'Image encoding model {model_name} was not found')

        return model
