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

    def __init__(self, model):
        """ Img2Vec
        :param model: name of the model to use
        """
        super(Img2Vec, self).__init__()

        self.device, _ = get_devices()
        self.model_name = model

        self.model = self._get_model()
        self.model = self.model.to(self.device)


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


    def _get_model(self):
        if self.model_name == 'resnet-18':
            model = torch.nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])

        elif self.model_name == 'resnext-50-small':
            model = torch.nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-1] , ChannelPoolAdaptiveAvg1d(output_size=512))

        elif self.model_name == 'resnext-50':
            model = torch.nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-1])

        else:
            raise Exception('Image encoding model ' + self.model_name + ' was not found')

        return model
