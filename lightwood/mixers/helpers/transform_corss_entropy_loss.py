import torch

# Basically cross entropy loss that does the one hot decoding of the targets inside of it... useful for code-logic reasons to have it setup like this
class TransformCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, input, target):
        cat_labels = target.max(1).indices
        return self.cross_entropy_loss(outputs, cat_labels)
