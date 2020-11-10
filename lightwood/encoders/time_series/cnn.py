import numpy as np
import torch
import torch.nn as nn
from torch import optim

from lightwood.encoders.time_series.helpers.cnn_helpers import *
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder

""" 
Causal 1D CAE using dilated convolutions loosely based on: https://arxiv.org/pdf/1611.05267.pdf
https://arxiv.org/pdf/1803.01271.pdf used for initial parameters as similar task. 
Overall idea is similar to a simplified WaveNet. 

Encoder is made up of a series of Temporal Convolution blocks - typical ResNet units with 
zero padding of front of tensor to ensure causality.
Residual connection 1x1 convolved to ensure it matches shape of block output
Dilation d*i at block i to increase receptive field
Maxpool halves sequence length after each block
Unable to use pretrained standard ResNet as it requires square input and is not causal.

Size of network depends on 'blocks' variable (list):
length = number of temporal convolution blocks, 
each element refers to number of filters applied during each convolution

Shape of encoded tensor:
(blocks[-1] , original max sequence length / 2^(len(blocks)))
Hence user can alter output size depending on application by tuning blocks variable
Fully connected layers could instead be used at ouput as in 
https://github.com/hsinyilin19/ResNetVAE, to set encoded length to desired value more easily.

Decoder works in reverse, dilation decreases with each block, interpolate/ upsample is used to 
restore to original length.

For now, AE encodes input and just aims to reproduce it. 
This could be adapted to produce n future predictions by training with a sliding window method 


In conclusion, this simple model has its merits.
the temporal dilated convolutions are necessary, and being able to scale depth and 
complexity of the model depending on the task is useful.
However, this process requires significant trial and error to ensure encoded output
is sensible shape etc. It could make more sense to have a large chunk at the start 
which purely extracts features with limited downsampling, then adjusting its output down 
to required latent size using fully connected layers. Making use of a deeper ResNet 
structure (current model has only 8 convolutional layers in encoder compared to 18 in
ResNet-18) is an option which would not require much adjustment. A complete WaveNet is 
another option but has not been applied as an autoencoder.
"""

class CnnEncoder(BaseEncoder):
    def __init__(self, blocks, batch_size, train_epochs, kernel_size=3, stop_on_error=0.01, learning_rate=0.001,
                 is_target=False, ts_n_dims=1):
        super().__init__(is_target)
        self.device, _ = get_devices()
        self._n_dims = ts_n_dims     # expected dimensionality of time series
        self._max_ts_length = 0
        self._sos = 0.0              # start of sequence for decoding
        self._eos = 0.0              # end of input sequence -- padding value for batches
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._kernel_size = kernel_size
        self._blocks = blocks        # user defined array giving nodes in each layer (last element = encoded size)
        self._batch_size = batch_size
        self._train_epochs = train_epochs
        self._pytorch_wrapper = torch.FloatTensor
        self._encoder = EncoderCNNts(kernel_size=self._kernel_size, input_dims=self._n_dims, blocks=self._blocks).to(self.device)
        self._decoder = DecoderCNNts(kernel_size=self._kernel_size, output_dims=self._n_dims, blocks=self._blocks).to(self.device)
        self._parameters = list(self._encoder.parameters()) + list(self._decoder.parameters())
        self._optimizer = optim.Adam(self._parameters, lr=self._learning_rate)
        self._criterion = nn.MSELoss()
        self._prepared = False

    def to(self, device, available_devices):
        # Device (GPU/CPU) decided by get_devices
        self.device = device
        self._encoder = self._encoder.to(self.device)
        return self

    def prepare_encoder(self, priming_data, batch_size, feedback_hoop_function=None):
        """
        Run this on the initial training data for the encoder
        :param priming_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...]
        :param batch_size: number of time series samples to be processed at once
        :return:
        """
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        if self._kernel_size % 2 == 0:
            raise Exception('Kernel size must be odd')

        # determine time_series length
        for data_points in priming_data:
            for dp in data_points:
                l = len(dp)
                self._max_ts_length = max(l, self._max_ts_length)

        # decrease for small datasets
        if batch_size >= len(priming_data):
            batch_size = len(priming_data) // 2

        self._encoder.train()
        for epoch in range(self._train_epochs):
            epoch_loss = float(0)
            data_idx = 0
            
            while data_idx < len(priming_data):
                # batch building
                data_points = priming_data[data_idx:min(data_idx + batch_size, len(priming_data))]
                batch = []
                for dp in data_points:
                    data_tensor = tensor_from_series(dp, self.device, self._n_dims, self._eos, self._max_ts_length)
                    batch.append(data_tensor)

                # Stack and transpose to shape: (batch_size, n_dims, timesteps)
                batch = torch.cat(batch, dim=0)
                batch = torch.torch.transpose(batch, 1, 2).to(self.device)
                data_idx += batch_size
                                    
                # setup optimizer
                self._optimizer.zero_grad()
                        
                # encode and decode
                encoded = self._encoder.forward(batch)
                output = self._decoder.forward(encoded)
                
                loss = self._criterion(output, batch)
                loss.backward()
                self._optimizer.step()    

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(priming_data)

            if epoch_loss < self._stop_on_error:
                break

            if feedback_hoop_function is not None:
                feedback_hoop_function('Epoch: {epoch}/ {total}\tLoss: {loss}'.format(
                    epoch=epoch+1, total=self._train_epochs, loss=epoch_loss))
        
        self._prepared = True
        return epoch_loss
        
    
    def encode(self, data):
        """
        This method encodes a list of time series data
        :param data: multidimensional time series as list of lists [[dim1_data], [dim2_data], ...]
                        (dim_data: string with format "x11, x12, ... x1n")
        :return:  encoded data
        """

        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        self._encoder.eval()
        with torch.no_grad():
            data_tensor = tensor_from_series(data, self.device, self._n_dims, self._eos, self._max_ts_length)
            data_tensor = torch.transpose(data_tensor, 1, 2)

            encoded_data = self._encoder.forward(data_tensor)
            return self._pytorch_wrapper(encoded_data)

    def decode(self, encoded):
        """ Decode a list of embedded multidimensional time series
        :param encoded: tensor of embeddings output from encoder
        :return: a list of reconstructed time series
        """
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        self._decoder.eval()
        with torch.no_grad():
            decoded = self._decoder.forward(encoded)
            return decoded



