import torch


class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles, target_index, index_offset, reduce='mean', **kwargs):
        super().__init__()
        self.quantiles = quantiles
        self.reduce = reduce
        self.index_offset = index_offset
        self.target_index = target_index

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[:, self.target_index] - preds[:, self.index_offset+i]
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
