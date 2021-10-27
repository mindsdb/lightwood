import torch
import torch.nn as nn
import torchvision.models as models
from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast


class ChannelPoolAdaptiveAvg1d(torch.nn.AdaptiveAvgPool1d):
    def forward(self, input):
        with LightwoodAutocast():
            n, c, _, _ = input.size()
            input = input.view(n, c, 1).permute(0, 2, 1)
            pooled = torch.nn.functional.adaptive_avg_pool1d(input, self.output_size)
            _, _, c = pooled.size()
            pooled = pooled.permute(0, 2, 1)
            return pooled.view(n, c)


class Img2Vec(nn.Module):
    def __init__(self):
        """ Img2Vec
        :param model: name of the model to use
        """
        super(Img2Vec, self).__init__()

        self.device, _ = get_devices()
        self.output_size = 512
        self.model = torch.nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[: -1],
                                         ChannelPoolAdaptiveAvg1d(output_size=self.output_size))
        self.model = self.model.to(self.device)

    def to(self, device, available_devices):
        self.device = device
        self.model = self.model.to(self.device)
        return self

    def forward(self, image, batch=True):
        with LightwoodAutocast():
            embedding = self.model(image.to(self.device))

            if batch:
                return embedding
            return embedding[0, :]
