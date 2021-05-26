import torch
from lightwood.helpers.torch import LightwoodAutocast


# Basically cross entropy loss that does the one hot decoding of the targets inside of it... useful for code-logic reasons to have it setup like this
class TransformCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, target):
        with LightwoodAutocast():
            cat_labels = target.max(1).indices
            return self.cross_entropy_loss(preds, cat_labels)

    def estimate_confidence(self, preds, maximum_confidence=None):
        confidences = []
        for pred in preds:
            conf = float(pred.max(0).values)/float(sum([float(x) if x > 0 else 0.000001 for x in pred]))
            if maximum_confidence is not None:
                conf = conf/maximum_confidence
            confidences.append(min(conf,1))
        return confidences
