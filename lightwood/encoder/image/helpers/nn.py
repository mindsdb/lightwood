import os
import shutil

import PIL
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import requests
from io import BytesIO

from lightwood.helpers.torch import LightwoodAutocast


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 128, 128)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128 * 128, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 128 * 128), nn.Tanh())

    def forward(self, x):
        with LightwoodAutocast():
            x = self.encoder(x)
            x = self.decoder(x)
        return x


class NnEncoderHelper:

    def __init__(self, images):
        """

        :param images: List of images
        """
        self._train_model(images)

    def encode(self, images):
        """
        Encode all the images from the list of paths(to images)

        :param images: List of images
        :return: a torch.floatTensor
        """
        data_source = []
        for image in self._transform_images(images):
            img = image.view(image.size(0), -1)
            img = Variable(img).cpu()
            data_source.append(self.model.encoder(img))
        return data_source

    def decode(self, encoded_values_tensor, save_to_path):
        """
        Decoded the encoded list of image tensors and write the decoded images to give path

        :param encoded_values_tensor:  List of encoded images tensors
        :param save_to_path: Path to store decoded images
        :return: a torch.floatTensor
        """
        decoded_values = []
        for i, encoded_image in enumerate(encoded_values_tensor):
            decoded = self.model.decoder(encoded_image)
            pic = to_img(decoded.cpu().data[0:-2])
            path_to_img = os.path.join(save_to_path, 'output_{}.png'.format(i))
            save_image(pic, path_to_img)
            decoded_values.append(os.path.abspath(path_to_img))
        return decoded_values

    def _train_model(self, images):
        """

        :param images: List of images paths
        """
        data_source = self._transform_images(images)
        self.model = autoencoder().cpu()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        for epoch in range(1, num_epochs + 1):
            for i, data in enumerate(data_source):
                img = data
                img = img.view(img.size(0), -1)
                img = Variable(img).cpu()
                # ===================forward=====================
                with LightwoodAutocast():
                    output = self.model(img)
                    loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ===================log========================
                print('epoch [{}/{}], loss:{:.4f}'
                      .format(epoch + 1, num_epochs, loss.data))
                if epoch % num_epochs == 0:
                    pic = to_img(output.cpu().data[0:-2])
                    if not os.path.exists('./mlp_img'):
                        os.mkdir('./mlp_img')
                    save_image(pic, './mlp_img/image_{}.png'.format(i))
        shutil.rmtree('./mlp_img')
        torch.save(self.model.state_dict(), './sim_autoencoder.pth')

    def _transform_images(self, images):
        """
            Transform the images

        :param images: List of images paths
        :return: a torch.floatTensor
        """
        data_source = []
        for image in images:
            if image is not None:
                if image.startswith('http'):
                    response = requests.get(image)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image)
                resized_image = img.resize((128, 128), PIL.Image.ANTIALIAS)
                transformed_img = transforms.ToTensor()(resized_image)
            else:
                transformed_img = [0] * self._encoded_length

            if self._encoded_length is None:
                self._encoded_length = len(transformed_img)

            data_source.append(transformed_img)
        return data_source
