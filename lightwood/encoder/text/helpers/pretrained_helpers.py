"""
2021.03.05

Basic helper functions for PretrainedLangEncoder
"""
import torch


class TextEmbed(torch.utils.data.Dataset):
    """
    Dataset class for quick embedding/label retrieval.
    Labels should be in the index space.

    If the labels provided are not in torch form, will convert them.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
