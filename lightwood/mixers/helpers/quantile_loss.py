import torch


class QuantileLoss(torch.nn.Module):
    def __init__(self, reduce='mean', **kwargs):
        super().__init__()
        self.reduce = reduce

    def forward(self, preds, target):
        main_mse_loss = (preds[:,:3] - target[:,:3]) ** 2

        lowe_range_loss = []
        for i in range(len(preds)):
            if preds[i,3] > preds[i,2]:
                lowe_range_loss.append(torch.Tensor([1000]))
            elif preds[i,3] > target[i,2]*0.95:
                lowe_range_loss.append(torch.Tensor([0]))
            else:
                lowe_range_loss.append(torch.Tensor([preds[i,3] - target[i,2]*0.95]) ** 2)

        lowe_range_loss = torch.Tensor(lowe_range_loss)

        upper_range_loss = []
        for i in range(len(preds)):
            if preds[i,4] < preds[i,2]:
                upper_range_loss.append(torch.Tensor([1000]))
            elif preds[i,4] < target[i,2]*1.05:
                upper_range_loss.append(torch.Tensor([0]))
            else:
                upper_range_loss.append(torch.Tensor([preds[i,4] - target[i,2]*1.05]) ** 2)

        upper_range_loss = torch.Tensor(lowe_range_loss)

        loss = torch.cat([main_mse_loss, lowe_range_loss, upper_range_loss])

        if reduce is False:
            return loss
        if reduce == 'mean':
            return torch.mean(loss)

        return torch.mean(loss)

'''
class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles=[0.85,0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i+3]
            losses.append(
                torch.max(
                   (q-1) * errors,
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
'''
