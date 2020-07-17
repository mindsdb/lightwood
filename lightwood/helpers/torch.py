import torch
from torch.nn.functional import pad


def concat_vectors_and_pad(vec_list, max_):
    assert len(vec_list) > 0
    assert len(vec_list) <= max_
    assert max_ > 0

    cat_vec = torch.cat(list(vec_list), dim=0)

    pad_size = max_ - len(vec_list)
    padding = (0, pad_size * vec_list[0].size(0))
    padded = pad(cat_vec[None], padding, 'constant', 0)[0]

    return padded


def average_vectors(vec_list):
    assert len(vec_list) > 0
    return torch.cat([emb[None] for emb in vec_list], dim=0).mean(0)
