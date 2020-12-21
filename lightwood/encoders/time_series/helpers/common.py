import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def len_to_mask(lengths, zeros):
    """
    :param lengths: list of ints with the lengths of the sequences
    :param zeros: bool. If false, the first lengths[i] values will be True and the rest will be false.
            If true, the first values will be False and the rest True
    :return: Boolean tensor of dimension (L, T) with L = len(lenghts) and T = lengths.max(),
    where with rows with lengths[i] True values followed by lengths.max()-lengths[i] False values.
    The True and False values are inverted if `zeros == True`
    """
    # Clean trick from:
    # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    mask = torch.arange(lengths.max(), device=lengths.device)[None, :] < lengths[:, None]
    if zeros:
        mask = ~mask  # Logical not
    return mask


def get_chunk(source, source_lengths, start, step):
    """Source is 3D tensor, shaped (batch_size, timesteps, n_dimensions), assuming static sequence length"""
    # Compute the lengths of the sequences (-1 due to the last element being used as target but not as data!
    trunc_seq_len = int(source_lengths[0].item() - 1)
    lengths = torch.zeros(source.shape[0]).fill_(trunc_seq_len).to(source.device)

    # This is necessary for MultiHeadedAttention to work
    end = min(start + step, trunc_seq_len)
    data = source[:, start:end, :]
    target = source[:, start+1:end+1, :]

    return data, target, lengths


class MinMaxNormalizer:
    def __init__(self, factor=1):
        self.scaler = MinMaxScaler()
        self.factor = factor

    def prepare(self, x):
        X = np.array([j for i in x for j in i]).reshape(-1, 1)
        self.scaler.fit(X)

    def encode(self, y):
        if not isinstance(y[0], list):
            y = y.reshape(-1, 1)
        return torch.Tensor(self.scaler.transform(y))

    def decode(self, y):
        return self.scaler.inverse_transform(y)


class CatNormalizer:
    def __init__(self):
        self.scaler = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.unk = "<UNK>"

    def prepare(self, x):
        X = []
        for i in x:
            for j in i:
                X.append(j if j is not None else self.unk)
        self.scaler.fit(np.array(X).reshape(-1, 1))

    def encode(self, Y):
        y = np.array([[j if j is not None else self.unk for j in i] for i in Y])
        out = []
        for i in y:
            out.append(self.scaler.transform(i.reshape(-1, 1)))
        return torch.Tensor(out)

    def decode(self, y):
        return [[i[0] for i in self.scaler.inverse_transform(o)] for o in y]
