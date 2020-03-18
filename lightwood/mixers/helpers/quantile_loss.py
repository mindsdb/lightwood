import torch


class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles, reduce='mean', **kwargs):
        super().__init__()
        self.quantiles = quantiles
        self.reduce = reduce

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[:, i*2:i*2+1] - preds[:, i*2:i*2+1]
            losses.append(
                torch.max(
                   (q-1) * errors,
                   q * errors
            ).unsqueeze(1))

        loss = torch.sum(torch.cat(losses, dim=1), dim=1)

        if self.reduce is False:
            return loss
        if self.reduce == 'mean':
            return torch.mean(loss)
