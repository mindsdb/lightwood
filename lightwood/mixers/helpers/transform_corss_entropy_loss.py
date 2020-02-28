import torch

# Basically cross entropy loss that does the one hot decoding of the targets inside of it... useful for code-logic reasons to have it setup like this
class TransformCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.maximum_confidence = 0.000000001
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, target):
        confidences = self.estimate_confidence(self, preds)
        self.maximum_confidence = max(self.maximum_confidence, max(confidences))

        cat_labels = target.max(1).indices
        return self.cross_entropy_loss(preds, cat_labels)

    def estimate_confidence(self, preds):
        confidences = []
        for pred in preds:
            conf = float(pred.max(0).values)/float(sum([x if x > 0 else 0 for x in preds.sum(0)]))
            conf = conf/self.maximum_confidence
            confidences.append(conf)
        return confidences
