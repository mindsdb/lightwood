import torch
from lightwood.helpers.torch import LightwoodAutocast


# Basically cross entropy loss that does the one hot decoding of the targets inside of it... useful for code-logic reasons to have it setup like this # noqa
class TransformCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, target):
        with LightwoodAutocast():
            cat_labels = target.max(1).indices
            return self.cross_entropy_loss(preds, cat_labels)
