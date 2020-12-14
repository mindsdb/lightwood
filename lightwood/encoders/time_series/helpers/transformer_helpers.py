import math
import torch
import torch.nn as nn
from lightwood.helpers.torch import LightwoodAutocast


def len_to_mask(lengths, zeros):
    """
    :param lengths: list of ints with the lengths of the sequences
    :param zeros: bool. If false, the first lengths[i] values will be True and the rest will be false.
            If true, the first values will be False and the rest True
    :return: Boolean tensor of dimension (L, T) with L = len(lenghts) and T = lengths.max(), where with rows with lengths[i] True values followed by lengths.max()-lengths[i] False values. The True and False values are inverted if `zeros == True`
    """
    # Clean trick from:
    # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    device = lengths.device
    mask = torch.arange(lengths.max()).to(device)[None, :] < lengths[:, None]
    if zeros:
        mask = ~mask  # Logical not
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with LightwoodAutocast():
            x = x + self.pe[: x.size(0), :]
            return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        # Lezcano: This could be an embedding if the data is in a range [0, R-1]
        self.encoder = nn.Linear(1, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, 1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # Take the logarithm
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src, lengths):
        device = src.device
        with LightwoodAutocast():
            if self.src_mask is None or self.src_mask.size(0) != src.size(0):
                # Attention mask to avoid attending to upcoming parts of the sequence
                self.src_mask = self._generate_square_subsequent_mask(src.size(0)).to(
                    device
                )
            src = self.encoder(src)
            src = self.pos_encoder(src)
            # The lengths_mask has to be of size [batch, lengths]
            lengths_mask = len_to_mask(lengths, zeros=True).to(device)
            output = self.transformer_encoder(
                src, mask=self.src_mask, src_key_padding_mask=lengths_mask
            )
            output = self.decoder(output)
        return output
