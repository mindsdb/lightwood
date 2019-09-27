import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from lightwood.config.config import CONFIG


class ChannelPoolAdaptiveAvg1d(torch.nn.AdaptiveAvgPool1d):
    def forward(self, input):
        n, c = input.size()
        input = input.view(n,c,1).permute(0,2,1)
        pooled =  torch.nn.functional.adaptive_avg_pool1d(input, self.output_size)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c)

class Img2Vec():

    def __init__(self, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE

        self.device = torch.device(device_str)
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        if self.model_name in ('alexnet', 'mobilenet', 'resnext-50-small'):
            my_embedding = torch.zeros(1, self.layer_output_size)
        elif self.model_name in ('resnet-18', 'resnext-50'):
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            if self.model_name in ('alexnet', 'mobilenet', 'resnext-50-small'):
                return my_embedding.numpy()[0, :]
            elif self.model_name in ('resnet-18', 'resnext-50'):
                return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name == 'resnext-50-small':
            model = models.resnext50_32x4d(pretrained=True)
            if layer == 'default':
                #b = torch.nn.AvgPool2d(kernel_size=(8,8),stride=(4,4))
                #a = torch.nn.AvgPool2d(kernel_size=(2,2),stride=2)
                #model.avgpool = b
                #model.fc = nn.Identity()
                #layer = model.avgpool
                model.fc = ChannelPoolAdaptiveAvg1d(output_size=512)
                layer = model.fc
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        if model_name == 'resnext-50':
            model = models.resnext50_32x4d(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        # @TODO: Fix or remove, this is both slow and inaccurate, not sure where we'd use it
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        # @TODO: Fix or remove, this is slow and not quite as accurate as resnet18, it's a failed experiment trying to end the encoder with the output from an FC rather than output from the pooling layer, might work on it later, if 1 month from now it stays the same, just remove it
        if model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('classifier')
                self.layer_output_size = 1000
            else:
                layer = model._modules.get(layer)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)
